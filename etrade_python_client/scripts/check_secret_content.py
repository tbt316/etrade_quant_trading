#!/usr/bin/env python3
"""Reject credential material in exact Git blobs and release archives.

The Git scanners read object bytes named by the index or an exact tree. They
never read a tracked path through the mutable working tree. Archive scanners
apply bounded, non-extracting ZIP/TAR parsing before inspecting regular member
bytes. Diagnostics contain only stable rule IDs and SHA-256 fingerprints.

This is a current-content release gate. It does not inspect Git history,
runtime state, logs, external caches, or revoke credentials that were already
published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


MAX_GIT_BLOBS = 10_000
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_TOTAL_GIT_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4_096
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 250

UNSAFE_GIT_ENVIRONMENT_VARIABLES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)

_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_PRIVATE_KEY_BEGIN = re.compile(
    rb"-----BEGIN[ \t]+(?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED)[ \t]+)?"
    rb"PRIVATE[ \t]+KEY-----|-----BEGIN[ \t]+PGP[ \t]+PRIVATE[ \t]+"
    rb"KEY[ \t]+BLOCK-----",
    re.IGNORECASE,
)
_PRIVATE_KEY_END = re.compile(
    rb"-----END[ \t]+(?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED)[ \t]+)?"
    rb"PRIVATE[ \t]+KEY-----|-----END[ \t]+PGP[ \t]+PRIVATE[ \t]+"
    rb"KEY[ \t]+BLOCK-----",
    re.IGNORECASE,
)
_KNOWN_TOKEN_RULES = (
    (
        "TOKEN_GITHUB_CLASSIC",
        re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,255}"),
    ),
    (
        "TOKEN_GITHUB_FINE_GRAINED",
        re.compile(rb"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{50,255}"),
    ),
    (
        "TOKEN_AWS_ACCESS_KEY_ID",
        re.compile(
            rb"(?<![A-Z0-9])(?:AKIA|ASIA|AIDA|AIPA|ANPA|ANVA|AROA|AGPA)"
            rb"[A-Z0-9]{16}(?![A-Z0-9])"
        ),
    ),
    (
        "TOKEN_SLACK",
        re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,200}"),
    ),
    (
        "TOKEN_STRIPE_LIVE",
        re.compile(rb"(?<![A-Za-z0-9])(?:sk|rk)_live_[A-Za-z0-9]{16,200}"),
    ),
    (
        "TOKEN_GOOGLE_API",
        re.compile(rb"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
    ),
    (
        "TOKEN_OPENAI",
        re.compile(
            rb"(?<![A-Za-z0-9])sk-(?:(?:proj|svcacct)-)?"
            rb"[A-Za-z0-9_-]{20,200}"
        ),
    ),
    (
        "TOKEN_BEARER_LITERAL",
        re.compile(
            rb"(?i)\b(?:authorization[ \t]*[:=][ \t]*[\"']?)?"
            rb"bearer[ \t]+[A-Za-z0-9._~+/-]{20,512}"
        ),
    ),
)
_CREDENTIAL_NAME = (
    rb"(?:api[_-]?(?:key|secret)|consumer[_-]?(?:key|secret)|"
    rb"client[_-]?secret|access[_-]?(?:key(?:[_-]?id)?|token)|"
    rb"secret[_-]?access[_-]?key|refresh[_-]?token|"
    rb"oauth[_-]?(?:token|secret|verifier)|private[_-]?key|"
    rb"password|passwd|action[_-]?pin|dashboard[_-]?(?:password|pin)|"
    rb"signing[_-]?secret|hmac[_-]?key|webhook[_-]?secret|"
    rb"database[_-]?(?:url|password)|secret)"
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?im)^[ \t]*(?:export[ \t]+)?[\"']?(?P<name>"
    + _CREDENTIAL_NAME
    + rb")[\"']?[ \t]*(?P<separator>=|:)[ \t]*(?P<value>[^\r\n]*)"
)
_INLINE_QUOTED_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?i)[{\[(,][ \t]*[\"']?(?P<name>"
    + _CREDENTIAL_NAME
    + rb")[\"']?[ \t]*(?:=|:)[ \t]*(?P<value>"
    + rb"\"(?:\\.|[^\"\\\r\n]){1,4096}\""
    + rb"|'(?:\\.|[^'\\\r\n]){1,4096}')"
)
_URL = re.compile(
    rb"(?i)\b(?:https?|postgres(?:ql)?|mysql|redis|amqps?)://"
    rb"[^\x00-\x20<>\"']{1,4096}"
)
_SENSITIVE_QUERY_NAME = re.compile(
    r"(?i)^(?:api[_-]?(?:key|secret)|consumer[_-]?(?:key|secret)|"
    r"client[_-]?secret|access[_-]?token|refresh[_-]?token|"
    r"oauth[_-]?(?:token|secret|verifier)|password|passwd|token|secret)$"
)
_PLACEHOLDER_MARKERS = (
    b"placeholder",
    b"replace-me",
    b"replace_me",
    b"replace-this",
    b"replace_this",
    b"changeme",
    b"change-me",
    b"change_me",
    b"example-only",
    b"example_only",
    b"dummy-value",
    b"dummy_value",
    b"fake-value",
    b"fake_value",
    b"redacted",
    b"not-a-secret",
    b"not_a_secret",
    b"do-not-use",
    b"do_not_use",
    b"your-api-key",
    b"your_api_key",
    b"your-secret",
    b"your_secret",
    b"<secret",
    b"<token",
    b"<password",
    b"${",
)
_PLACEHOLDER_EXACT = frozenset(
    {
        b"",
        b"none",
        b"null",
        b"unset",
        b"unused",
        b"disabled",
        b"placeholder",
        b"example",
        b"sample",
        b"dummy",
        b"fake",
        b"redacted",
        b"changeme",
    }
)
_PEM_PLACEHOLDER_PAYLOADS = frozenset(
    {
        b"placeholder",
        b"placeholderonly",
        b"explicitplaceholder",
        b"explicitplaceholderonly",
        b"privatekeyplaceholder",
        b"redacted",
        b"intentionallyredacted",
        b"replaceme",
        b"yourprivatekeyhere",
        b"exampleonly",
        b"dummy",
    }
)
_NON_LITERAL_PREFIXES = (
    b"$",
    b"%",
    b"os.",
    b"env.",
    b"getenv",
    b"config.",
    b"settings.",
    b"secret(",
    b"resolve_",
)
_NON_LITERAL_WORDS = frozenset(
    {
        b"str",
        b"bytes",
        b"int",
        b"bool",
        b"object",
        b"any",
        b"optional",
        b"required",
        b"true",
        b"false",
    }
)
# Exact path-and-value hashes for reviewed, non-secret examples that predate
# this gate and cannot be rewritten in this isolated checkpoint. New fixtures
# should contain an explicit placeholder token instead of extending this set.
_REVIEWED_PLACEHOLDER_SHA256 = frozenset(
    {
        (
            "etrade_python_client/docs/runtime_configuration.md",
            "0bf9de24d9714afa5c4ba504f8668668bfcb3fcaf55f2e9e6f230f48bd9cb737",
        ),
        (
            "etrade_python_client/docs/runtime_configuration.md",
            "108f2eb8ce654f7108c1968a856449af88449258393ddddc15958b3594114abe",
        ),
        (
            "etrade_python_client/tests/test_contract_reference_cache_truth.py",
            "2ceac6f36363c6246a64cca805cd43ca7a01b14eb2fcc532ceec3f60f2f7df1c",
        ),
        (
            "etrade_python_client/tests/test_etrade_broker_reader.py",
            "3f16bed7089f4653e5ef21bfd2824d7f3aaaecc7a598e7e89c580e1606a9cc52",
        ),
        (
            "etrade_python_client/tests/test_etrade_broker_transport.py",
            "3f16bed7089f4653e5ef21bfd2824d7f3aaaecc7a598e7e89c580e1606a9cc52",
        ),
        (
            "etrade_python_client/tests/test_etrade_broker_transport.py",
            "5ba3ed8c96af9806fec7a3cde1848c841d5353f57d7c32b78b23f6a4f78ab4fa",
        ),
        (
            "etrade_python_client/tests/test_etrade_cancel_transport.py",
            "3f16bed7089f4653e5ef21bfd2824d7f3aaaecc7a598e7e89c580e1606a9cc52",
        ),
        (
            "etrade_python_client/tests/test_etrade_cancel_transport.py",
            "4a7972d6a7442c7a4b8061f5a24084488ebc652c3af4ae863d8a08d7079811f7",
        ),
        (
            "etrade_python_client/tests/test_etrade_order_gateway.py",
            "3f16bed7089f4653e5ef21bfd2824d7f3aaaecc7a598e7e89c580e1606a9cc52",
        ),
        (
            "etrade_python_client/tests/test_read_only_dashboard.py",
            "87cbebfeebc05f7c54ac9336c4b4bbec831227a641951a4bde7edd56020f8590",
        ),
        (
            "etrade_python_client/tests/test_read_only_dashboard.py",
            "8810ad581e59f2bc3928b261707a71308f7e139eb04820366dc4d5c18d980225",
        ),
        (
            "etrade_python_client/tests/test_regime_evidence_store.py",
            "f3035e6ee3097be11e6a30229371f29cf25813a029cbf05ce60e999c1454fd49",
        ),
        (
            "etrade_python_client/tests/test_regime_market_data_gateway.py",
            "14f2817043b30fcb9b1187d0d19e12efa9deab7ceb4b062091693baecb689bb7",
        ),
        (
            "etrade_python_client/tests/test_runtime_config.py",
            "d9cc3fbf816c43a0d7427da2f11f769829a07edbd13bbbe37fc08fe4a25579e3",
        ),
        (
            "etrade_python_client/tests/test_runtime_config.py",
            "87cbebfeebc05f7c54ac9336c4b4bbec831227a641951a4bde7edd56020f8590",
        ),
        (
            "etrade_python_client/tests/test_runtime_composition.py",
            "87cbebfeebc05f7c54ac9336c4b4bbec831227a641951a4bde7edd56020f8590",
        ),
        (
            "etrade_python_client/tests/test_runtime_composition.py",
            "e9ec2a11fa0d058c22d8a510bb6d4d8268268b77011f9774e8543953accaaa3f",
        ),
        (
            "etrade_python_client/tests/test_runtime_safety.py",
            "998bf2cd967eadd354bd6e5df56903ffd20294b5c2f7917b51c79b302a5bc0f8",
        ),
        (
            "etrade_python_client/tests/test_runtime_safety.py",
            "9e4dfc4565c18f77be8d9e7926cabaaf3a0bdc8cfa10275cf4a1d748047a59f6",
        ),
        (
            "etrade_python_client/tests/test_spy_position_tracker_sync.py",
            "5ba3ed8c96af9806fec7a3cde1848c841d5353f57d7c32b78b23f6a4f78ab4fa",
        ),
    }
)


class SecretScanError(RuntimeError):
    """The requested exact source could not be scanned safely."""


@dataclass(frozen=True, order=True)
class SecretFinding:
    """One redacted content-policy finding."""

    rule_id: str
    path_fingerprint: str
    match_fingerprint: str
    review_path: str = field(compare=False, repr=False)

    def render(self, *, include_review_path: bool = False) -> str:
        rendered = (
            f"{self.rule_id} path_fingerprint={self.path_fingerprint} "
            f"match_fingerprint={self.match_fingerprint}"
        )
        if include_review_path:
            rendered += " path=" + json.dumps(self.review_path, ensure_ascii=True)
        return rendered


@dataclass(frozen=True)
class SecretScanReport:
    """Bounded scan totals plus deterministic redacted findings."""

    scanned_items: int
    scanned_bytes: int
    findings: tuple[SecretFinding, ...]


def _fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:20]


def _finding(rule_id: str, logical_path: str, matched: bytes) -> SecretFinding:
    return SecretFinding(
        rule_id=rule_id,
        path_fingerprint=_fingerprint(logical_path.encode("utf-8", errors="surrogatepass")),
        match_fingerprint=_fingerprint(rule_id.encode("ascii") + b"\0" + matched),
        review_path=logical_path,
    )


def _is_placeholder(value: bytes) -> bool:
    candidate = value.strip().strip(b"\"'`").strip().lower()
    candidate = urllib.parse.unquote_to_bytes(candidate.decode("ascii", errors="ignore"))
    candidate = candidate.strip().lower()
    if candidate in _PLACEHOLDER_EXACT:
        return True
    if any(marker in candidate for marker in _PLACEHOLDER_MARKERS):
        return True
    if (
        len(candidate) >= 2
        and candidate[:1] == b"{"
        and candidate[-1:] == b"}"
    ):
        return True
    compact = re.sub(rb"[^a-z0-9]", b"", candidate)
    if len(compact) >= 6 and len(set(compact)) == 1:
        return True
    return False


def _is_placeholder_private_key_payload(payload: bytes) -> bool:
    """Accept only a complete PEM body that is itself an explicit marker."""

    if not payload or len(payload) > 4096 or b"\0" in payload:
        return False
    normalized = re.sub(rb"[^a-z0-9]+", b"", payload.lower())
    return normalized in _PEM_PLACEHOLDER_PAYLOADS


def _is_reviewed_placeholder(logical_path: str, value: bytes) -> bool:
    return (
        logical_path,
        hashlib.sha256(value).hexdigest(),
    ) in _REVIEWED_PLACEHOLDER_SHA256


def _literal_value(raw: bytes) -> bytes | None:
    value = raw.strip()
    if not value:
        return None
    quoted = value[:1] in {b"\"", b"'", b"`"}
    if quoted:
        quote = value[:1]
        closing: int | None = None
        index = 1
        while index < len(value):
            character = value[index : index + 1]
            if character == b"\\":
                index += 2
                continue
            if character == quote:
                closing = index
                break
            index += 1
        if closing is None:
            return None
        value = value[1:closing]
    else:
        value = re.split(rb"[ \t,;#]", value, maxsplit=1)[0]
    value = value.strip()
    folded = value.lower()
    if (
        not value
        or folded in _NON_LITERAL_WORDS
        or any(folded.startswith(prefix) for prefix in _NON_LITERAL_PREFIXES)
    ):
        return None
    if not quoted:
        if any(character in value for character in b"()[]{}"):
            return None
        if re.fullmatch(
            rb"[A-Za-z_][A-Za-z0-9_.]*[|&*+][A-Za-z_][A-Za-z0-9_.]*",
            value,
        ):
            return None
        if re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_.]*", value):
            if (
                b"." in value
                or b"_" in value
                or value.isupper()
                or any(
                    marker in folded
                    for marker in (
                        b"key",
                        b"token",
                        b"secret",
                        b"password",
                        b"credential",
                    )
                )
            ):
                return None
    return value


def _credible_assignment(name: bytes, value: bytes) -> bool:
    if len(value) > 4096 or _is_placeholder(value):
        return False
    folded_name = name.lower().replace(b"-", b"_")
    if b"password" in folded_name or folded_name.endswith(b"_pin"):
        return len(value) >= 4
    if len(value) < 8:
        return False
    if b"://" in value:
        return True
    if any(character in value for character in b" \t"):
        return len(value) >= 12
    has_alpha = any(
        (65 <= character <= 90) or (97 <= character <= 122)
        for character in value
    )
    has_digit = any(48 <= character <= 57 for character in value)
    has_symbol = any(
        not (
            (65 <= character <= 90)
            or (97 <= character <= 122)
            or (48 <= character <= 57)
        )
        for character in value
    )
    return has_alpha and (has_digit or has_symbol)


def scan_bytes(logical_path: str, content: bytes) -> tuple[SecretFinding, ...]:
    """Scan one bounded logical file without retaining matched secret values."""

    findings: set[SecretFinding] = set()

    for match in _PRIVATE_KEY_BEGIN.finditer(content):
        end_match = _PRIVATE_KEY_END.search(content, match.end())
        context_end = (
            min(len(content), end_match.end())
            if end_match is not None
            else min(len(content), match.end() + 64 * 1024)
        )
        context = content[match.start() : context_end]
        if not (
            end_match is not None
            and (
                _is_placeholder_private_key_payload(
                    content[match.end() : end_match.start()]
                )
                or _is_reviewed_placeholder(logical_path, context)
            )
        ):
            findings.add(
                _finding("PRIVATE_KEY_PEM", logical_path, match.group(0))
            )

    for rule_id, pattern in _KNOWN_TOKEN_RULES:
        for match in pattern.finditer(content):
            value = match.group(0)
            if not (
                _is_placeholder(value)
                or _is_reviewed_placeholder(logical_path, value)
            ):
                findings.add(_finding(rule_id, logical_path, value))

    for assignment_pattern in (
        _CREDENTIAL_ASSIGNMENT,
        _INLINE_QUOTED_CREDENTIAL_ASSIGNMENT,
    ):
        for match in assignment_pattern.finditer(content):
            name = match.group("name")
            value = _literal_value(match.group("value"))
            if (
                value is not None
                and _credible_assignment(name, value)
                and not _is_reviewed_placeholder(logical_path, value)
            ):
                findings.add(
                    _finding(
                        "LITERAL_CREDENTIAL_ASSIGNMENT",
                        logical_path,
                        name.lower() + b"\0" + value,
                    )
                )

    for match in _URL.finditer(content):
        raw_url = match.group(0).rstrip(b").,;]")
        try:
            parsed = urllib.parse.urlsplit(raw_url.decode("ascii"))
        except (UnicodeError, ValueError):
            continue
        if parsed.username is not None and parsed.password is not None:
            userinfo = (
                urllib.parse.unquote_to_bytes(parsed.username)
                + b"\0"
                + urllib.parse.unquote_to_bytes(parsed.password)
            )
            decoded_password = urllib.parse.unquote_to_bytes(parsed.password)
            if (
                _credible_assignment(b"password", decoded_password)
                and not _is_reviewed_placeholder(logical_path, decoded_password)
            ):
                findings.add(
                    _finding("URL_USERINFO_CREDENTIAL", logical_path, userinfo)
                )
        try:
            query_items = urllib.parse.parse_qsl(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=False,
                max_num_fields=128,
            )
        except ValueError:
            query_items = []
        for query_name, query_value in query_items:
            if (
                _SENSITIVE_QUERY_NAME.fullmatch(query_name)
                and query_value
                and _credible_assignment(
                    query_name.encode("utf-8"),
                    query_value.encode("utf-8"),
                )
                and not _is_reviewed_placeholder(
                    logical_path,
                    query_value.encode("utf-8"),
                )
            ):
                findings.add(
                    _finding(
                        "URL_QUERY_CREDENTIAL",
                        logical_path,
                        query_name.casefold().encode("utf-8")
                        + b"\0"
                        + query_value.encode("utf-8"),
                    )
                )

    return tuple(sorted(findings))


def _git_environment() -> dict[str, str]:
    overridden = sorted(
        name for name in UNSAFE_GIT_ENVIRONMENT_VARIABLES if name in os.environ
    )
    if overridden:
        raise SecretScanError("unsafe Git repository override is set")
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }


def _git(cwd: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            check=False,
            env=_git_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise SecretScanError("could not execute Git") from error
    if completed.returncode != 0:
        raise SecretScanError("Git could not provide the requested exact content")
    return completed.stdout


def _git_root(start: Path) -> Path:
    start = start.resolve()
    if start.is_file():
        start = start.parent
    raw = _git(start, "rev-parse", "--show-toplevel")
    try:
        root = Path(raw.decode("utf-8").rstrip("\r\n"))
    except UnicodeError as error:
        raise SecretScanError("Git root is not valid UTF-8") from error
    if not root.is_absolute():
        root = start / root
    return root.resolve()


def _safe_prefix(prefix: str) -> str:
    path = PurePosixPath(prefix)
    if (
        not prefix
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != prefix
        or "\\" in prefix
    ):
        raise SecretScanError("tree prefix is not a normalized relative path")
    return prefix


def _decode_git_path(raw_path: bytes) -> str:
    try:
        path = raw_path.decode("utf-8")
    except UnicodeError as error:
        raise SecretScanError("Git path is not valid UTF-8") from error
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        raise SecretScanError("Git returned an unsafe path")
    return path


def _index_objects(root: Path) -> tuple[tuple[str, str], ...]:
    payload = _git(root, "ls-files", "--stage", "-z")
    objects: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = header.decode("ascii").split()
        except (UnicodeError, ValueError) as error:
            raise SecretScanError("Git index record is malformed") from error
        path = _decode_git_path(raw_path)
        if (
            stage != "0"
            or mode not in {"100644", "100755"}
            or not _OBJECT_ID.fullmatch(object_id)
            or path in seen
        ):
            raise SecretScanError("Git index is not a unique regular-file snapshot")
        seen.add(path)
        objects.append((path, object_id))
    return tuple(sorted(objects))


def _tree_objects(
    root: Path,
    object_id: str,
    *,
    prefix: str = "",
) -> tuple[tuple[str, str], ...]:
    if not _OBJECT_ID.fullmatch(object_id):
        raise SecretScanError("tree identity is not canonical")
    arguments = ["ls-tree", "-r", "-z", object_id]
    if prefix:
        arguments.extend(("--", _safe_prefix(prefix)))
    payload = _git(root, *arguments)
    objects: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, blob_id = header.decode("ascii").split()
        except (UnicodeError, ValueError) as error:
            raise SecretScanError("Git tree record is malformed") from error
        path = _decode_git_path(raw_path)
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or not _OBJECT_ID.fullmatch(blob_id)
            or path in seen
        ):
            raise SecretScanError("Git tree is not a unique regular-blob snapshot")
        seen.add(path)
        objects.append((path, blob_id))
    return tuple(sorted(objects))


def _scan_git_objects(
    root: Path,
    objects: Sequence[tuple[str, str]],
) -> SecretScanReport:
    if len(objects) > MAX_GIT_BLOBS:
        return SecretScanReport(
            scanned_items=0,
            scanned_bytes=0,
            findings=(
                _finding("GIT_BLOB_COUNT_LIMIT", "git-object-set", str(len(objects)).encode()),
            ),
        )
    findings: set[SecretFinding] = set()
    scanned_items = 0
    scanned_bytes = 0
    for path, object_id in objects:
        raw_size = _git(root, "cat-file", "-s", object_id)
        try:
            size = int(raw_size.decode("ascii").strip())
        except (UnicodeError, ValueError) as error:
            raise SecretScanError("Git blob size is invalid") from error
        if size < 0 or size > MAX_BLOB_BYTES:
            findings.add(
                _finding("GIT_BLOB_SIZE_LIMIT", path, str(size).encode("ascii"))
            )
            continue
        if scanned_bytes + size > MAX_TOTAL_GIT_BYTES:
            findings.add(
                _finding(
                    "GIT_TOTAL_SIZE_LIMIT",
                    path,
                    str(scanned_bytes + size).encode("ascii"),
                )
            )
            break
        content = _git(root, "cat-file", "blob", object_id)
        if len(content) != size:
            raise SecretScanError("Git blob length changed during exact-object read")
        scanned_items += 1
        scanned_bytes += size
        findings.update(scan_bytes(path, content))
    return SecretScanReport(
        scanned_items=scanned_items,
        scanned_bytes=scanned_bytes,
        findings=tuple(sorted(findings)),
    )


def scan_git_index(start: Path) -> SecretScanReport:
    """Scan exact stage-zero regular blobs currently named by the Git index."""

    root = _git_root(start)
    return _scan_git_objects(root, _index_objects(root))


def scan_git_tree(
    start: Path,
    object_id: str,
    *,
    prefix: str = "",
) -> SecretScanReport:
    """Scan regular blobs from one exact Git tree/commit identity."""

    root = _git_root(start)
    return _scan_git_objects(
        root,
        _tree_objects(root, object_id, prefix=prefix),
    )


def _safe_archive_path(name: str) -> str | None:
    candidate = name[:-1] if name.endswith("/") else name
    if (
        not candidate
        or "\\" in candidate
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
    ):
        return None
    path = PurePosixPath(candidate)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != candidate:
        return None
    return path.as_posix()


def _archive_limit_finding(rule_id: str, logical_path: str, value: int) -> SecretFinding:
    return _finding(rule_id, logical_path, str(value).encode("ascii"))


def _scan_zip(path: Path) -> SecretScanReport:
    findings: set[SecretFinding] = set()
    scanned_items = 0
    scanned_bytes = 0
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            findings.add(
                _archive_limit_finding(
                    "ARCHIVE_MEMBER_COUNT_LIMIT",
                    "zip-member-set",
                    len(infos),
                )
            )
            return SecretScanReport(0, 0, tuple(sorted(findings)))
        for info in infos:
            raw_name = info.orig_filename
            logical_name = f"zip:{raw_name}"
            normalized = (
                _safe_archive_path(info.filename)
                if info.orig_filename == info.filename
                else None
            )
            if normalized is None:
                findings.add(
                    _finding(
                        "ARCHIVE_UNSAFE_PATH",
                        logical_name,
                        raw_name.encode("utf-8", errors="surrogatepass"),
                    )
                )
                continue
            logical_name = f"zip:{normalized}"
            folded = normalized.casefold()
            if normalized in seen or folded in seen_casefolded:
                findings.add(
                    _finding(
                        "ARCHIVE_DUPLICATE_PATH",
                        logical_name,
                        normalized.encode("utf-8"),
                    )
                )
                continue
            seen.add(normalized)
            seen_casefolded.add(folded)
            if info.is_dir():
                continue
            unix_mode = info.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if (
                info.flag_bits & 0x1
                or file_type not in {0, stat.S_IFREG}
                or info.file_size < 0
                or info.compress_size < 0
            ):
                findings.add(
                    _finding(
                        "ARCHIVE_NONREGULAR_MEMBER",
                        logical_name,
                        str(info.external_attr).encode("ascii"),
                    )
                )
                continue
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_MEMBER_SIZE_LIMIT",
                        logical_name,
                        info.file_size,
                    )
                )
                continue
            if scanned_bytes + info.file_size > MAX_ARCHIVE_TOTAL_BYTES:
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_TOTAL_SIZE_LIMIT",
                        logical_name,
                        scanned_bytes + info.file_size,
                    )
                )
                break
            if (
                info.file_size > 1024 * 1024
                and info.compress_size > 0
                and info.file_size
                > info.compress_size * MAX_ARCHIVE_COMPRESSION_RATIO
            ):
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_COMPRESSION_RATIO_LIMIT",
                        logical_name,
                        info.file_size // info.compress_size,
                    )
                )
                continue
            with archive.open(info, "r") as member:
                content = member.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
            if len(content) != info.file_size:
                raise SecretScanError("ZIP member length differs from its descriptor")
            scanned_items += 1
            scanned_bytes += len(content)
            findings.update(scan_bytes(logical_name, content))
    return SecretScanReport(
        scanned_items=scanned_items,
        scanned_bytes=scanned_bytes,
        findings=tuple(sorted(findings)),
    )


def _scan_tar(path: Path) -> SecretScanReport:
    findings: set[SecretFinding] = set()
    scanned_items = 0
    scanned_bytes = 0
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    with tarfile.open(path, mode="r:*") as archive:
        member_count = 0
        for member in archive:
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_MEMBER_COUNT_LIMIT",
                        "tar-member-set",
                        member_count,
                    )
                )
                break
            logical_name = f"tar:{member.name}"
            normalized = _safe_archive_path(member.name)
            if normalized is None:
                findings.add(
                    _finding(
                        "ARCHIVE_UNSAFE_PATH",
                        logical_name,
                        member.name.encode("utf-8", errors="surrogatepass"),
                    )
                )
                continue
            logical_name = f"tar:{normalized}"
            folded = normalized.casefold()
            if normalized in seen or folded in seen_casefolded:
                findings.add(
                    _finding(
                        "ARCHIVE_DUPLICATE_PATH",
                        logical_name,
                        normalized.encode("utf-8"),
                    )
                )
                continue
            seen.add(normalized)
            seen_casefolded.add(folded)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                findings.add(
                    _finding(
                        "ARCHIVE_NONREGULAR_MEMBER",
                        logical_name,
                        str(member.type).encode("ascii", errors="replace"),
                    )
                )
                continue
            if member.size > MAX_ARCHIVE_MEMBER_BYTES:
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_MEMBER_SIZE_LIMIT",
                        logical_name,
                        member.size,
                    )
                )
                continue
            if scanned_bytes + member.size > MAX_ARCHIVE_TOTAL_BYTES:
                findings.add(
                    _archive_limit_finding(
                        "ARCHIVE_TOTAL_SIZE_LIMIT",
                        logical_name,
                        scanned_bytes + member.size,
                    )
                )
                break
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SecretScanError("TAR member could not be read")
            content = extracted.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
            if len(content) != member.size:
                raise SecretScanError("TAR member length differs from its descriptor")
            scanned_items += 1
            scanned_bytes += len(content)
            findings.update(scan_bytes(logical_name, content))
    return SecretScanReport(
        scanned_items=scanned_items,
        scanned_bytes=scanned_bytes,
        findings=tuple(sorted(findings)),
    )


def scan_archive(path: Path) -> SecretScanReport:
    """Scan bounded regular members of one wheel/ZIP or sdist/TAR archive."""

    path = Path(os.path.abspath(path))
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SecretScanError("archive could not be inspected") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise SecretScanError("archive is not a regular file")
    if metadata.st_size > MAX_ARCHIVE_BYTES:
        return SecretScanReport(
            scanned_items=0,
            scanned_bytes=0,
            findings=(
                _archive_limit_finding(
                    "ARCHIVE_FILE_SIZE_LIMIT",
                    "archive-file",
                    metadata.st_size,
                ),
            ),
        )
    lower_name = path.name.casefold()
    try:
        if lower_name.endswith((".whl", ".zip")):
            return _scan_zip(path)
        if lower_name.endswith((".tar.gz", ".tgz", ".tar")):
            return _scan_tar(path)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as error:
        raise SecretScanError("archive parsing failed closed") from error
    raise SecretScanError("archive extension is not a supported wheel or sdist")


def combine_reports(reports: Iterable[SecretScanReport]) -> SecretScanReport:
    reports = tuple(reports)
    return SecretScanReport(
        scanned_items=sum(report.scanned_items for report in reports),
        scanned_bytes=sum(report.scanned_bytes for report in reports),
        findings=tuple(
            sorted(
                {
                    finding
                    for report in reports
                    for finding in report.findings
                }
            )
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=Path,
        default=Path.cwd(),
        help="path inside the repository for index/tree scans",
    )
    parser.add_argument(
        "--tree",
        help="scan one exact 40- or 64-hex Git tree/commit instead of the index",
    )
    parser.add_argument(
        "--tree-prefix",
        default="",
        help="normalized literal repository-relative prefix used with --tree",
    )
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        default=[],
        help="bounded wheel/sdist archive to scan; may be repeated",
    )
    parser.add_argument(
        "--review-paths",
        action="store_true",
        help=(
            "include JSON-escaped repository/member paths for local review; "
            "matched values remain fingerprint-only"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.archive and (args.tree or args.tree_prefix):
        print("SECRET_GATE_ARGUMENT_ERROR", file=sys.stderr)
        return 2
    if args.tree_prefix and not args.tree:
        print("SECRET_GATE_ARGUMENT_ERROR", file=sys.stderr)
        return 2
    try:
        if args.archive:
            report = combine_reports(scan_archive(path) for path in args.archive)
        elif args.tree:
            report = scan_git_tree(
                args.start,
                args.tree,
                prefix=args.tree_prefix,
            )
        else:
            report = scan_git_index(args.start)
    except SecretScanError as error:
        print(
            "SECRET_GATE_SCAN_ERROR "
            f"error_fingerprint={_fingerprint(str(error).encode('utf-8'))}",
            file=sys.stderr,
        )
        return 2

    if report.findings:
        for finding in report.findings:
            print(
                finding.render(include_review_path=args.review_paths),
                file=sys.stderr,
            )
        print(
            f"SECRET_GATE_REJECTED findings={len(report.findings)}",
            file=sys.stderr,
        )
        return 1
    print(
        "SECRET_GATE_PASSED "
        f"items={report.scanned_items} bytes={report.scanned_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
