#!/usr/bin/env python3
"""Validate release archives against the exact committed Git payload."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - CPython 3.10
    import tomli as tomllib


EXPECTED_PACKAGES = frozenset(
    {
        "accounts",
        "ai_agents",
        "backtesting",
        "core_api",
        "data_and_research",
        "live_trading",
        "market",
        "order",
        "polygonio",
        "strategies",
    }
)
EXPECTED_PACKAGE_DATA = frozenset(
    {
        "backtesting/strategies/baseline_put_spread.yaml",
        "backtesting/strategies/baseline_put_spread_catchup_refill.yaml",
        "backtesting/strategies/put_call_credit_spread.yaml",
        "live_trading/dashboard_template.html",
    }
)
TRACKED_SDIST_FILES = frozenset(
    {
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
    }
)
EXPECTED_WHEEL_METADATA = frozenset(
    {
        "METADATA",
        "RECORD",
        "WHEEL",
        "top_level.txt",
    }
)
EXPECTED_EGG_INFO = frozenset(
    {
        "etrade_python_client/etrade_quant_trading.egg-info/PKG-INFO",
        "etrade_python_client/etrade_quant_trading.egg-info/SOURCES.txt",
        "etrade_python_client/etrade_quant_trading.egg-info/dependency_links.txt",
        "etrade_python_client/etrade_quant_trading.egg-info/requires.txt",
        "etrade_python_client/etrade_quant_trading.egg-info/top_level.txt",
    }
)
EXPECTED_SDIST_METADATA = frozenset(
    {
        *TRACKED_SDIST_FILES,
        "PKG-INFO",
        "setup.cfg",
        *EXPECTED_EGG_INFO,
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        "__pycache__",
        "scratch",
        "tests",
        "yfinance",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "config.ini",
        "credentials.json",
        "etrade_session.json",
        "live_trading_settings.json",
        "production_arm.json",
        "session.json",
    }
)
GENERATED_SETUP_CFG = b"[egg_info]\ntag_build = \ntag_date = 0\n\n"
MAX_GENERATED_METADATA_BYTES = 2 * 1024 * 1024


class ArtifactContractError(RuntimeError):
    """A built distribution violated the reviewed payload contract."""


def _safe_path(name: str) -> PurePosixPath:
    candidate = name[:-1] if name.endswith("/") else name
    if (
        not candidate
        or "\\" in candidate
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
    ):
        raise ArtifactContractError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(candidate)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != candidate
    ):
        raise ArtifactContractError(f"unsafe archive path: {name!r}")
    return path


def _safe_zip_path(info: zipfile.ZipInfo) -> PurePosixPath:
    if info.orig_filename != info.filename:
        raise ArtifactContractError(
            f"wheel path was truncated while parsing: {info.orig_filename!r}"
        )
    return _safe_path(info.filename)


def _forbidden_path(path: PurePosixPath) -> bool:
    return (
        bool(FORBIDDEN_PARTS.intersection(path.parts))
        or path.name in FORBIDDEN_NAMES
        or path.suffix in {".pyc", ".pyo"}
    )


def _git(repository_root: Path, *arguments: str) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ArtifactContractError(f"could not execute Git: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ArtifactContractError(f"Git command failed: {detail}")
    return result.stdout


def _resolve_commit(repository_root: Path, revision: str) -> str:
    if not revision or revision.startswith("-"):
        raise ArtifactContractError("Git revision must be an explicit non-option")
    commit = _git(
        repository_root,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
    ).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ArtifactContractError(f"Git returned an invalid commit id: {commit!r}")
    return commit


def _git_blob(repository_root: Path, object_id: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
        raise ArtifactContractError(f"Git returned an invalid object id: {object_id!r}")
    return _git(repository_root, "cat-file", "blob", object_id)


def _git_path_blob(repository_root: Path, commit: str, path: str) -> bytes:
    safe_path = _safe_path(path).as_posix()
    return _git(repository_root, "show", f"{commit}:{safe_path}")


def _expected_git_payload(
    repository_root: Path,
    revision: str,
) -> tuple[str, dict[str, bytes], dict[str, bytes]]:
    """Return the committed package and tracked sdist bytes.

    The expectation comes from ``git ls-tree``, not ``git archive`` or the
    mutable worktree, so ``export-ignore`` cannot silently shrink a release.
    """

    commit = _resolve_commit(repository_root, revision)
    raw_tree = _git(
        repository_root,
        "ls-tree",
        "-rz",
        commit,
        "--",
        "etrade_python_client",
    )
    objects: dict[str, tuple[str, str]] = {}
    casefolded: set[str] = set()
    prefix = "etrade_python_client/"
    for raw_record in raw_tree.split(b"\0"):
        if not raw_record:
            continue
        try:
            raw_metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = raw_metadata.decode("ascii").split()
            tracked_path = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as error:
            raise ArtifactContractError("Git tree contains an invalid record") from error
        if not tracked_path.startswith(prefix):
            continue
        relative = _safe_path(tracked_path[len(prefix) :])
        if not relative.parts or relative.parts[0] not in EXPECTED_PACKAGES:
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            continue
        is_python = relative.suffix == ".py"
        is_package_data = relative.as_posix() in EXPECTED_PACKAGE_DATA
        if is_python and len(relative.parts) != 2:
            raise ArtifactContractError(
                f"tracked Python module is outside the package contract: {relative}"
            )
        if not is_python and not is_package_data:
            continue
        normalized = relative.as_posix()
        if normalized in objects or normalized.casefold() in casefolded:
            raise ArtifactContractError(
                f"Git payload contains a duplicate or case collision: {normalized}"
            )
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise ArtifactContractError(
                f"Git payload is not a regular blob: {normalized} ({mode} {object_type})"
            )
        objects[normalized] = (object_id, mode)
        casefolded.add(normalized.casefold())

    required = {
        *(f"{package}/__init__.py" for package in EXPECTED_PACKAGES),
        *EXPECTED_PACKAGE_DATA,
    }
    missing = sorted(required - objects.keys())
    if missing:
        raise ArtifactContractError(f"committed package payload is missing: {missing}")
    payload = {
        name: _git_blob(repository_root, object_id)
        for name, (object_id, _mode) in objects.items()
    }
    tracked_sdist = {
        name: _git_path_blob(repository_root, commit, name)
        for name in TRACKED_SDIST_FILES
    }
    return commit, payload, tracked_sdist


def _metadata_contract(pyproject_bytes: bytes) -> dict[str, object]:
    try:
        configuration = tomllib.loads(pyproject_bytes.decode("utf-8"))
        project = configuration["project"]
        build_requires = configuration["build-system"]["requires"]
        name = str(project["name"])
        version = str(project["version"])
        summary = str(project["description"])
        requires_python = str(project["requires-python"])
        dependencies = tuple(str(value) for value in project["dependencies"])
        optional = {
            str(extra): tuple(str(value) for value in values)
            for extra, values in project.get("optional-dependencies", {}).items()
        }
    except (KeyError, TypeError, UnicodeError, ValueError) as error:
        raise ArtifactContractError("committed pyproject metadata is invalid") from error
    setuptools_pins = [
        value.split("==", 1)[1]
        for value in build_requires
        if str(value).startswith("setuptools==")
    ]
    if len(setuptools_pins) != 1:
        raise ArtifactContractError("build backend must pin exactly one setuptools version")
    requirements = {
        *dependencies,
        *(
            f'{requirement}; extra == "{extra}"'
            for extra, values in optional.items()
            for requirement in values
        ),
    }
    return {
        "name": name,
        "normalized_name": re.sub(r"[-_.]+", "_", name),
        "version": version,
        "summary": summary,
        "requires_python": requires_python,
        "dependencies": dependencies,
        "optional": optional,
        "requirements": requirements,
        "setuptools_version": setuptools_pins[0],
    }


def _release_artifacts(
    dist_dir: Path,
    contract: Mapping[str, object],
) -> tuple[Path, Path]:
    basename = f"{contract['normalized_name']}-{contract['version']}"
    expected_names = {
        f"{basename}-py3-none-any.whl",
        f"{basename}.tar.gz",
    }
    entries = {path.name: path for path in dist_dir.iterdir()}
    if set(entries) != expected_names:
        raise ArtifactContractError(
            f"release artifact names differ: {sorted(entries)}"
        )
    for name, path in entries.items():
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o644
        ):
            raise ArtifactContractError(
                f"release artifact is not an owner-writable regular file: {name}"
            )
    return (
        entries[f"{basename}-py3-none-any.whl"],
        entries[f"{basename}.tar.gz"],
    )


def _required_header(message, name: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1:
        raise ArtifactContractError(
            f"metadata must contain exactly one {name} header"
        )
    return str(values[0])


def _verify_metadata(
    metadata_bytes: bytes,
    contract: Mapping[str, object],
    readme_bytes: bytes,
) -> None:
    message = BytesParser(policy=policy.default).parsebytes(metadata_bytes)
    expected_header_names = {
        "Description-Content-Type",
        "Metadata-Version",
        "Name",
        "Provides-Extra",
        "Requires-Dist",
        "Requires-Python",
        "Summary",
        "Version",
    }
    if set(message.keys()) != expected_header_names:
        raise ArtifactContractError("wheel metadata header set differs")
    expected_headers = {
        "Metadata-Version": "2.4",
        "Name": contract["name"],
        "Version": contract["version"],
        "Summary": contract["summary"],
        "Description-Content-Type": "text/markdown",
    }
    for name, expected in expected_headers.items():
        actual = _required_header(message, name)
        if actual != expected:
            raise ArtifactContractError(
                f"metadata {name} differs: expected={expected!r}, actual={actual!r}"
            )
    actual_python = {
        value.strip()
        for value in _required_header(message, "Requires-Python").split(",")
    }
    expected_python = {
        value.strip()
        for value in str(contract["requires_python"]).split(",")
    }
    if actual_python != expected_python:
        raise ArtifactContractError("metadata Requires-Python differs")
    if set(message.get_all("Requires-Dist", [])) != contract["requirements"]:
        raise ArtifactContractError("wheel dependency metadata differs from pyproject")
    if set(message.get_all("Provides-Extra", [])) != set(contract["optional"]):
        raise ArtifactContractError("wheel extra metadata differs from pyproject")
    try:
        readme = readme_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ArtifactContractError("committed README is not UTF-8") from error
    if message.get_payload() != readme:
        raise ArtifactContractError("wheel long description differs from committed README")


def _verify_wheel_descriptor(
    descriptor_bytes: bytes,
    contract: Mapping[str, object],
) -> None:
    message = BytesParser(policy=policy.default).parsebytes(descriptor_bytes)
    expected = {
        "Wheel-Version": "1.0",
        "Generator": f"setuptools ({contract['setuptools_version']})",
        "Root-Is-Purelib": "true",
        "Tag": "py3-none-any",
    }
    if set(message.keys()) != set(expected) or message.get_payload():
        raise ArtifactContractError("wheel descriptor structure differs")
    for name, value in expected.items():
        if _required_header(message, name) != value:
            raise ArtifactContractError(f"wheel descriptor {name} differs")


def _verify_record(contents: Mapping[str, bytes], record_name: str) -> None:
    try:
        rows = list(
            csv.reader(
                io.StringIO(contents[record_name].decode("utf-8"), newline="")
            )
        )
    except (KeyError, UnicodeError, csv.Error) as error:
        raise ArtifactContractError("wheel RECORD is unreadable") from error
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise ArtifactContractError("wheel RECORD row must contain three fields")
        name, digest, size = row
        normalized = _safe_path(name).as_posix()
        if normalized in recorded:
            raise ArtifactContractError(f"wheel RECORD repeats {normalized}")
        recorded.add(normalized)
        if normalized not in contents:
            raise ArtifactContractError(f"wheel RECORD names missing file {normalized}")
        if normalized == record_name:
            if digest or size:
                raise ArtifactContractError("wheel RECORD self-entry must be unhashed")
            continue
        data = contents[normalized]
        expected_digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).rstrip(b"=").decode("ascii")
        if digest != f"sha256={expected_digest}" or size != str(len(data)):
            raise ArtifactContractError(f"wheel RECORD integrity differs for {normalized}")
    if recorded != set(contents):
        raise ArtifactContractError("wheel RECORD does not cover every archive file")


def _inspect_wheel(
    wheel_path: Path,
    expected_payload: Mapping[str, bytes],
    contract: Mapping[str, object],
    readme_bytes: bytes,
    expected_epoch: int,
) -> bytes:
    contents: dict[str, bytes] = {}
    dist_info_roots: set[str] = set()
    metadata_files: set[str] = set()
    seen_casefolded: set[str] = set()
    with zipfile.ZipFile(wheel_path) as archive:
        if archive.comment:
            raise ArtifactContractError("wheel archive comment must be empty")
        for info in archive.infolist():
            path = _safe_zip_path(info)
            if info.is_dir():
                raise ArtifactContractError(
                    f"wheel contains an unexpected directory entry: {info.filename}"
                )
            normalized = path.as_posix()
            if normalized in contents or normalized.casefold() in seen_casefolded:
                raise ArtifactContractError(
                    f"wheel contains a duplicate or case-colliding path: {info.filename}"
                )
            seen_casefolded.add(normalized.casefold())
            if (
                info.comment
                or info.extra
                or info.flag_bits != 0
                or info.create_system != 3
                or info.compress_type != zipfile.ZIP_DEFLATED
            ):
                raise ArtifactContractError(
                    f"wheel member metadata differs: {info.filename}"
                )
            expected_permissions = (
                0o664
                if path.parts[0].endswith(".dist-info")
                and path.name == "RECORD"
                else 0o644
            )
            expected_external_attr = (
                stat.S_IFREG | expected_permissions
            ) << 16
            if info.external_attr != expected_external_attr:
                raise ArtifactContractError(
                    f"wheel member type or permissions differ: {info.filename}"
                )
            expected_timestamp = time.gmtime(expected_epoch - expected_epoch % 2)[:6]
            if info.date_time != expected_timestamp:
                raise ArtifactContractError(
                    f"wheel member timestamp differs: {info.filename}"
                )
            if _forbidden_path(path):
                raise ArtifactContractError(
                    f"wheel contains forbidden path: {info.filename}"
                )
            if path.parts[0].endswith(".dist-info"):
                dist_info_roots.add(path.parts[0])
                metadata_files.add(PurePosixPath(*path.parts[1:]).as_posix())
                if info.file_size > MAX_GENERATED_METADATA_BYTES:
                    raise ArtifactContractError(
                        f"wheel metadata is unexpectedly large: {info.filename}"
                    )
            elif normalized in expected_payload:
                if info.file_size != len(expected_payload[normalized]):
                    raise ArtifactContractError(
                        f"wheel payload size differs for {normalized}"
                    )
            contents[normalized] = archive.read(info)

    expected_dist_info = (
        f"{contract['normalized_name']}-{contract['version']}.dist-info"
    )
    if dist_info_roots != {expected_dist_info}:
        raise ArtifactContractError(
            f"wheel .dist-info root differs: {sorted(dist_info_roots)}"
        )
    if metadata_files != EXPECTED_WHEEL_METADATA:
        raise ArtifactContractError(
            f"wheel metadata differs: {sorted(metadata_files)}"
        )
    files = {
        name: data
        for name, data in contents.items()
        if not PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if set(files) != set(expected_payload):
        missing = sorted(set(expected_payload) - set(files))
        unexpected = sorted(set(files) - set(expected_payload))
        raise ArtifactContractError(
            f"wheel payload mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, expected_bytes in expected_payload.items():
        if files[name] != expected_bytes:
            raise ArtifactContractError(
                f"wheel bytes differ from committed Git blob: {name}"
            )
    roots = {PurePosixPath(name).parts[0] for name in files}
    if roots != EXPECTED_PACKAGES:
        raise ArtifactContractError(f"wheel package roots differ: {sorted(roots)}")

    metadata_name = f"{expected_dist_info}/METADATA"
    _verify_metadata(contents[metadata_name], contract, readme_bytes)
    _verify_wheel_descriptor(
        contents[f"{expected_dist_info}/WHEEL"],
        contract,
    )
    expected_top_level = (
        "\n".join(sorted(EXPECTED_PACKAGES)) + "\n"
    ).encode("utf-8")
    if contents[f"{expected_dist_info}/top_level.txt"] != expected_top_level:
        raise ArtifactContractError("wheel top-level package metadata differs")
    _verify_record(contents, f"{expected_dist_info}/RECORD")
    return contents[metadata_name]


def _expected_requires_txt(contract: Mapping[str, object]) -> bytes:
    lines = [*contract["dependencies"]]
    for extra, requirements in contract["optional"].items():
        lines.extend(["", f"[{extra}]", *requirements])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _verify_sources_list(
    sources_bytes: bytes,
    expected_payload: Mapping[str, bytes],
) -> None:
    try:
        lines = sources_bytes.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ArtifactContractError("sdist SOURCES.txt is not UTF-8") from error
    if len(lines) != len(set(lines)):
        raise ArtifactContractError("sdist SOURCES.txt contains duplicate entries")
    for line in lines:
        _safe_path(line)
    expected = {
        *TRACKED_SDIST_FILES,
        *EXPECTED_EGG_INFO,
        *(f"etrade_python_client/{name}" for name in expected_payload),
    }
    if set(lines) != expected:
        raise ArtifactContractError("sdist SOURCES.txt differs from full payload")


def _inspect_sdist(
    sdist_path: Path,
    expected_payload: Mapping[str, bytes],
    tracked_sdist: Mapping[str, bytes],
    contract: Mapping[str, object],
    wheel_metadata: bytes,
    expected_epoch: int,
) -> None:
    with sdist_path.open("rb") as compressed:
        gzip_header = compressed.read(10)
    if (
        len(gzip_header) != 10
        or gzip_header[:4] != b"\x1f\x8b\x08\x00"
        or int.from_bytes(gzip_header[4:8], "little") != expected_epoch
        or gzip_header[8:] != b"\x02\xff"
    ):
        raise ArtifactContractError("sdist gzip header is not deterministic")
    expected_sdist_files = EXPECTED_SDIST_METADATA.union(
        f"etrade_python_client/{name}" for name in expected_payload
    )
    expected_root = f"{contract['normalized_name']}-{contract['version']}"
    expected_directories = {expected_root}
    for relative_name in expected_sdist_files:
        parent = PurePosixPath(relative_name).parent
        while parent.as_posix() != ".":
            expected_directories.add(f"{expected_root}/{parent.as_posix()}")
            parent = parent.parent

    relative_contents: dict[str, bytes] = {}
    directories: set[str] = set()
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        if archive.pax_headers:
            raise ArtifactContractError("sdist global PAX headers must be empty")
        members = archive.getmembers()
        member_names = [_safe_path(member.name).as_posix() for member in members]
        if member_names != sorted(
            member_names,
            key=lambda name: (len(PurePosixPath(name).parts), name),
        ):
            raise ArtifactContractError("sdist member order is not deterministic")
        roots = {_safe_path(member.name).parts[0] for member in members}
        if roots != {expected_root}:
            raise ArtifactContractError(f"sdist root differs: {sorted(roots)}")
        for member in members:
            path = _safe_path(member.name)
            normalized = path.as_posix()
            if normalized in seen or normalized.casefold() in seen_casefolded:
                raise ArtifactContractError(
                    f"sdist contains a duplicate or case-colliding path: {member.name}"
                )
            seen.add(normalized)
            seen_casefolded.add(normalized.casefold())
            unexpected_pax = set(member.pax_headers) - {"mtime", "path"}
            if unexpected_pax:
                raise ArtifactContractError(
                    f"sdist member has unexpected PAX metadata: "
                    f"{member.name} {sorted(unexpected_pax)}"
                )
            if (
                "path" in member.pax_headers
                and member.pax_headers["path"] != member.name
            ):
                raise ArtifactContractError(
                    f"sdist PAX path differs: {member.name}"
                )
            if "mtime" in member.pax_headers:
                try:
                    pax_mtime = float(member.pax_headers["mtime"])
                except ValueError as error:
                    raise ArtifactContractError(
                        f"sdist PAX mtime is invalid: {member.name}"
                    ) from error
                if pax_mtime != member.mtime:
                    raise ArtifactContractError(
                        f"sdist PAX mtime differs: {member.name}"
                    )
            if not member.isdir() and not member.isfile():
                raise ArtifactContractError(
                    f"sdist contains a non-regular member: {member.name}"
                )
            expected_mode = 0o755 if member.isdir() else 0o644
            if stat.S_IMODE(member.mode) != expected_mode:
                raise ArtifactContractError(
                    f"sdist member mode differs: {member.name}"
                )
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
                or member.mtime != expected_epoch
            ):
                raise ArtifactContractError(
                    f"sdist deterministic metadata differs: {member.name}"
                )
            relative = PurePosixPath(*path.parts[1:])
            if _forbidden_path(relative):
                raise ArtifactContractError(
                    f"sdist contains forbidden path: {member.name}"
                )
            if member.isdir():
                directories.add(normalized)
                continue
            relative_name = relative.as_posix()
            package_name = (
                PurePosixPath(*relative.parts[1:]).as_posix()
                if relative.parts[:1] == ("etrade_python_client",)
                else None
            )
            if package_name in expected_payload:
                if member.size != len(expected_payload[package_name]):
                    raise ArtifactContractError(
                        f"sdist payload size differs for {package_name}"
                    )
            elif member.size > MAX_GENERATED_METADATA_BYTES:
                raise ArtifactContractError(
                    f"sdist metadata is unexpectedly large: {relative_name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ArtifactContractError(f"could not read sdist member: {member.name}")
            relative_contents[relative_name] = extracted.read()

    if directories != expected_directories:
        missing = sorted(expected_directories - directories)
        unexpected = sorted(directories - expected_directories)
        raise ArtifactContractError(
            f"sdist directory set differs; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if set(relative_contents) != expected_sdist_files:
        missing = sorted(expected_sdist_files - set(relative_contents))
        unexpected = sorted(set(relative_contents) - expected_sdist_files)
        raise ArtifactContractError(
            f"sdist full payload mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, expected_bytes in expected_payload.items():
        packaged_name = f"etrade_python_client/{name}"
        if relative_contents[packaged_name] != expected_bytes:
            raise ArtifactContractError(
                f"sdist bytes differ from committed Git blob: {name}"
            )
    for name, expected_bytes in tracked_sdist.items():
        if relative_contents[name] != expected_bytes:
            raise ArtifactContractError(
                f"sdist bytes differ from committed Git blob: {name}"
            )

    egg_root = "etrade_python_client/etrade_quant_trading.egg-info"
    if relative_contents["PKG-INFO"] != wheel_metadata:
        raise ArtifactContractError("sdist PKG-INFO differs from wheel METADATA")
    if relative_contents[f"{egg_root}/PKG-INFO"] != wheel_metadata:
        raise ArtifactContractError("sdist egg-info PKG-INFO differs from wheel")
    if relative_contents["setup.cfg"] != GENERATED_SETUP_CFG:
        raise ArtifactContractError("sdist generated setup.cfg differs")
    expected_top_level = (
        "\n".join(sorted(EXPECTED_PACKAGES)) + "\n"
    ).encode("utf-8")
    if relative_contents[f"{egg_root}/top_level.txt"] != expected_top_level:
        raise ArtifactContractError("sdist top-level package metadata differs")
    if relative_contents[f"{egg_root}/dependency_links.txt"] != b"\n":
        raise ArtifactContractError("sdist dependency link metadata differs")
    if relative_contents[f"{egg_root}/requires.txt"] != _expected_requires_txt(contract):
        raise ArtifactContractError("sdist dependency metadata differs from pyproject")
    _verify_sources_list(
        relative_contents[f"{egg_root}/SOURCES.txt"],
        expected_payload,
    )


def inspect_release(
    dist_dir: Path,
    repository_root: Path,
    git_revision: str = "HEAD",
) -> tuple[Path, Path]:
    """Inspect the single wheel and sdist against a committed Git revision."""

    commit, expected_payload, tracked_sdist = _expected_git_payload(
        repository_root,
        git_revision,
    )
    try:
        expected_epoch = int(
            _git(
                repository_root,
                "show",
                "-s",
                "--format=%ct",
                commit,
            ).decode("ascii").strip()
        )
    except (UnicodeError, ValueError) as error:
        raise ArtifactContractError("Git commit timestamp is invalid") from error
    contract = _metadata_contract(tracked_sdist["pyproject.toml"])
    wheel, sdist = _release_artifacts(dist_dir, contract)
    wheel_metadata = _inspect_wheel(
        wheel,
        expected_payload,
        contract,
        tracked_sdist["README.md"],
        expected_epoch,
    )
    _inspect_sdist(
        sdist,
        expected_payload,
        tracked_sdist,
        contract,
        wheel_metadata,
        expected_epoch,
    )
    return wheel, sdist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        required=True,
        help="Directory containing exactly one wheel and one .tar.gz sdist.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Git repository containing the canonical package source.",
    )
    parser.add_argument(
        "--git-revision",
        default="HEAD",
        help="Committed revision whose exact bytes must be distributed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        wheel, sdist = inspect_release(
            args.dist_dir.resolve(),
            args.repository_root.resolve(),
            args.git_revision,
        )
    except (
        ArtifactContractError,
        OSError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"release artifact check failed: {error}", file=sys.stderr)
        return 1
    print(
        "release artifact check passed: "
        f"{wheel.name}, {sdist.name}, {len(EXPECTED_PACKAGES)} package roots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
