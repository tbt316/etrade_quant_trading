"""Fail-closed runtime controls for the order-capable E*TRADE process."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import fcntl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, NoReturn


class RuntimeSafetyError(RuntimeError):
    """Raised before broker construction when a runtime safety invariant fails."""


class LegacyExecutionDisabled(RuntimeSafetyError):
    """Raised when obsolete code attempts to reach a broker mutation surface."""


def reject_legacy_execution(surface: str) -> NoReturn:
    """Unconditionally reject a legacy execution surface.

    Legacy paths have no operator, configuration, or environment override.
    Broker mutations must be routed through the durable E*TRADE order gateway.
    """

    raise LegacyExecutionDisabled(
        f"legacy execution surface is quarantined and disabled: {surface}; "
        "route broker mutations through ETradeOrderGateway"
    )


ARM_SCHEMA_VERSION = 1
MAX_PRODUCTION_ARM_LIFETIME = timedelta(minutes=15)
MAX_PRODUCTION_ARM_FUTURE_SKEW = timedelta(seconds=60)
_ARM_SIGNED_FIELDS = (
    "version",
    "environment",
    "expected_account_id",
    "expected_account_id_key",
    "expected_institution_type",
    "issued_at",
    "expires_at",
)
_ARM_FIELDS = frozenset((*_ARM_SIGNED_FIELDS, "signature"))


def _required_text(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeSafetyError(f"{name} is required")
    return value.strip()


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeSafetyError(f"production arm {label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeSafetyError(f"production arm {label} is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeSafetyError(f"production arm {label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _arm_payload(document: dict[str, Any]) -> bytes:
    if not isinstance(document, dict) or set(document) - {"signature"} != set(_ARM_SIGNED_FIELDS):
        raise RuntimeSafetyError("production arm schema is invalid")
    return json.dumps(
        {field: document[field] for field in _ARM_SIGNED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def production_arm_signature(document: dict[str, Any], secret: str) -> str:
    """Return the signature required in an offline production arm document."""

    return hmac.new(secret.encode("utf-8"), _arm_payload(document), hashlib.sha256).hexdigest()


def _require_arm_secret(secret: str | None) -> str:
    result = _required_text(secret, "ETRADE_PRODUCTION_ARMING_SECRET")
    if len(result) < 32:
        raise RuntimeSafetyError("ETRADE_PRODUCTION_ARMING_SECRET is too short")
    return result


def _validate_arm_schema(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != _ARM_FIELDS:
        raise RuntimeSafetyError("production arm schema is invalid")
    if type(document["version"]) is not int or document["version"] != ARM_SCHEMA_VERSION:
        raise RuntimeSafetyError("production arm version is invalid")
    if document["environment"] != "production":
        raise RuntimeSafetyError("production arm environment is invalid")
    for field in _ARM_SIGNED_FIELDS:
        if field in {"version", "environment", "issued_at", "expires_at"}:
            continue
        _required_text(document[field], field)
    if not isinstance(document["signature"], str):
        raise RuntimeSafetyError("production arm proof is invalid")
    return document


def _trusted_parent_descriptor(path: Path) -> int:
    parent = path.parent
    try:
        before = os.lstat(parent)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise RuntimeSafetyError("runtime file parent is unsafe")
        descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeSafetyError("runtime file parent is unsafe") from exc
    after = os.fstat(descriptor)
    if after.st_dev != before.st_dev or after.st_ino != before.st_ino:
        os.close(descriptor)
        raise RuntimeSafetyError("runtime file parent is unsafe")
    return descriptor


def _write_owner_only_atomic(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    parent_descriptor = _trusted_parent_descriptor(target)
    temporary_name = None
    descriptor = -1
    try:
        for _ in range(16):
            candidate = f".{target.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if descriptor < 0 or temporary_name is None:
            raise RuntimeSafetyError("could not create runtime file")
        os.fchmod(descriptor, 0o600)
        payload_offset = 0
        while payload_offset < len(payload):
            written = os.write(descriptor, payload[payload_offset:])
            if written <= 0:
                raise RuntimeSafetyError("could not write runtime file")
            payload_offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_name, target.name, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def write_owner_only_json(path: str | Path, value: Any) -> None:
    """Atomically replace a sensitive JSON document without following links."""

    _write_owner_only_atomic(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_owner_only_json(path: str | Path, *, label: str = "runtime file") -> Any:
    """Read JSON only from an owner-only regular file under a trusted parent."""

    target = Path(path)
    parent_descriptor = _trusted_parent_descriptor(target)
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > 64 * 1024
        ):
            raise RuntimeSafetyError(f"{label} must be an owner-only regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSafetyError(f"{label} is invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def secure_lock_file(path: str | Path):
    """Open and non-blockingly lock an owner-only runtime lock file."""

    target = Path(path)
    parent_descriptor = _trusted_parent_descriptor(target)
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeSafetyError("runtime lock file is unsafe")
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeSafetyError("another trading instance is already running") from exc
        handle = os.fdopen(descriptor, "r+")
        descriptor = -1
        return handle
    except OSError as exc:
        raise RuntimeSafetyError("runtime lock file is unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _open_owner_only_log(path: str | Path) -> int:
    """Open a regular owner-only log through a verified parent descriptor."""

    target = Path(path)
    parent_descriptor = _trusted_parent_descriptor(target)
    try:
        descriptor = os.open(
            target.name,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise RuntimeSafetyError("log file is unsafe") from exc
    finally:
        os.close(parent_descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise RuntimeSafetyError("log file is unsafe")
    return descriptor


def issue_production_arm(
    *,
    output: str | Path,
    expected_account_id: str,
    expected_account_id_key: str,
    expected_institution_type: str,
    ttl_seconds: int,
    secret: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a bounded, signed, owner-only production arm document."""

    signing_secret = _require_arm_secret(secret)
    if type(ttl_seconds) is not int or not 0 < ttl_seconds <= int(MAX_PRODUCTION_ARM_LIFETIME.total_seconds()):
        raise RuntimeSafetyError("production arm lifetime is invalid")
    issued_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    document = {
        "version": ARM_SCHEMA_VERSION,
        "environment": "production",
        "expected_account_id": _required_text(expected_account_id, "expected account id"),
        "expected_account_id_key": _required_text(expected_account_id_key, "expected account key"),
        "expected_institution_type": _required_text(expected_institution_type, "expected institution type"),
        "issued_at": issued_at.isoformat(),
        "expires_at": (issued_at + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    document["signature"] = production_arm_signature(document, signing_secret)
    _write_owner_only_atomic(output, json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return document


def secure_append_text(path: str | Path, text: str) -> None:
    """Append a log record without following links or accepting unsafe files."""

    descriptor = _open_owner_only_log(path)
    try:
        payload = text.encode("utf-8")
        payload_offset = 0
        while payload_offset < len(payload):
            written = os.write(descriptor, payload[payload_offset:])
            if written <= 0:
                raise RuntimeSafetyError("could not append dashboard log")
            payload_offset += written
    finally:
        os.close(descriptor)


class OwnerOnlyRotatingFileHandler(RotatingFileHandler):
    """A rotating handler that only opens owner-only regular log files."""

    def _open(self):
        descriptor = _open_owner_only_log(self.baseFilename)
        return os.fdopen(descriptor, self.mode, encoding=self.encoding, errors=self.errors)


def configure_owner_only_logger(name: str, filename: str = "python_client.log") -> logging.Logger:
    """Attach one owner-only rotating file handler to a shared client logger."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    expected_filename = os.path.abspath(filename)
    for existing in logger.handlers:
        if isinstance(existing, OwnerOnlyRotatingFileHandler) and existing.baseFilename == expected_filename:
            return logger
    handler = OwnerOnlyRotatingFileHandler(filename, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)-15s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"))
    handler.addFilter(_SafeClientLogFilter())
    logger.addHandler(handler)
    return logger


class _SafeClientLogFilter(logging.Filter):
    """Defence in depth for older client call sites that log request objects."""

    def filter(self, record: logging.LogRecord) -> bool:
        raw_message = str(record.msg)
        message = raw_message.lower()
        if "request header" in message or "response headers" in message:
            if record.args:
                values = record.args if isinstance(record.args, tuple) else (record.args,)
                record.args = tuple(redact_http_headers(value) for value in values)
        elif (
            "request payload" in message
            or "response" in message
            or "request url" in message
            or "order placed" in message
        ) and record.args:
            values = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(content_fingerprint(value) for value in values)
        elif raw_message.lstrip().startswith(("{", "[", "<")):
            record.msg = "Structured client detail withheld: %s"
            record.args = (content_fingerprint(raw_message),)
        return True


def redact_http_headers(headers: Any) -> dict[str, str]:
    """Return only header names; no HTTP header values are retained in logs."""

    try:
        items = headers.items()
    except AttributeError:
        return {}
    return {str(name): "[REDACTED]" for name, _value in items}


def payload_fingerprint(payload: str | bytes) -> dict[str, int | str]:
    """Provide audit-safe payload metadata without retaining order XML."""

    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def content_fingerprint(value: Any) -> dict[str, int | str]:
    """Fingerprint arbitrary client detail without retaining account or order data."""

    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            encoded = type(value).__name__.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def resolve_etrade_consumer_key(
    use_sandbox: bool,
    *,
    consumer_key: str | None = None,
    config_value: str | None = None,
    required: bool = True,
) -> str | None:
    """Resolve one consumer key from an explicit value, env, then local config."""

    if consumer_key is not None:
        return consumer_key
    environment_name = "ETRADE_SANDBOX_CONSUMER_KEY" if use_sandbox else "ETRADE_LIVE_CONSUMER_KEY"
    value = os.getenv(environment_name) or config_value
    if isinstance(value, str) and value.strip():
        return value.strip()
    if required:
        environment = "sandbox" if use_sandbox else "production"
        raise RuntimeSafetyError(f"Missing E*TRADE {environment} consumer key; set {environment_name} or local config.ini")
    return None


@dataclass(frozen=True)
class RuntimeSafetyBoundary:
    environment: str
    expected_account_id: str | None
    expected_account_id_key: str | None
    expected_institution_type: str | None
    arm_issued_at: datetime | None = None
    arm_expires_at: datetime | None = None

    @property
    def use_sandbox(self) -> bool:
        return self.environment == "sandbox"

    @property
    def account_selection_kwargs(self) -> dict[str, str | int | None]:
        if self.use_sandbox and self.expected_account_id_key is None:
            return {"selected_account_id": 1}
        return {
            "selected_account_id": None,
            "expected_account_id_key": self.expected_account_id_key,
            "expected_account_id": self.expected_account_id,
            "expected_institution_type": self.expected_institution_type,
        }

    def assert_current(self, now: datetime | None = None) -> None:
        if self.environment != "production":
            return
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self.arm_issued_at is None or self.arm_expires_at is None:
            raise RuntimeSafetyError("production arm proof is unavailable")
        if current_time < self.arm_issued_at - MAX_PRODUCTION_ARM_FUTURE_SKEW:
            raise RuntimeSafetyError("production arm proof is not yet valid")
        if current_time >= self.arm_expires_at:
            raise RuntimeSafetyError("production arm proof is expired")

    def verify_account(self, account: Any, *, now: datetime | None = None) -> None:
        self.assert_current(now)
        if self.use_sandbox and self.expected_account_id_key is None:
            return
        if not isinstance(account, dict):
            raise RuntimeSafetyError("broker did not return a selected account")
        checks = {
            "accountIdKey": self.expected_account_id_key,
            "accountId": self.expected_account_id,
            "institutionType": self.expected_institution_type,
        }
        for field, expected in checks.items():
            if account.get(field) != expected:
                raise RuntimeSafetyError(f"broker account {field} does not match the armed identity")


def _read_production_arm(
    path: str | Path,
    secret: str,
    now: datetime,
) -> tuple[dict[str, Any], datetime, datetime]:
    arm_path = Path(path)
    try:
        descriptor = os.open(
            arm_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeSafetyError("production arm document is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > 64 * 1024
        ):
            raise RuntimeSafetyError("production arm document must be an owner-only regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSafetyError("production arm document is invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    document = _validate_arm_schema(document)
    signature = document["signature"]
    expected_signature = production_arm_signature(document, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise RuntimeSafetyError("production arm proof is invalid")
    issued_at = _parse_timestamp(document["issued_at"], "issued_at")
    expiry = _parse_timestamp(document["expires_at"], "expires_at")
    if issued_at > now + MAX_PRODUCTION_ARM_FUTURE_SKEW:
        raise RuntimeSafetyError("production arm issued_at is too far in the future")
    if expiry <= issued_at:
        raise RuntimeSafetyError("production arm lifetime is invalid")
    if expiry - issued_at > MAX_PRODUCTION_ARM_LIFETIME:
        raise RuntimeSafetyError("production arm lifetime is invalid")
    if expiry <= now:
        raise RuntimeSafetyError("production arm proof is expired")
    return document, issued_at, expiry


def build_runtime_safety_boundary(
    *,
    environment: str | None,
    legacy_sandbox: bool | None,
    expected_account_id: str | None,
    expected_account_id_key: str | None,
    expected_institution_type: str | None,
    production_arm_file: str | Path | None = None,
    production_arm_secret: str | None = None,
    now: datetime | None = None,
) -> RuntimeSafetyBoundary:
    """Resolve one explicit mode and verify production's independent arm proof.

    The legacy ``--sandbox`` flag remains an explicit sandbox-only compatibility
    path. Missing or conflicting mode input never falls through to production.
    """

    if environment not in {None, "sandbox", "production"}:
        raise RuntimeSafetyError("environment must be sandbox or production")
    if environment is None:
        if legacy_sandbox is True:
            resolved = "sandbox"
        elif legacy_sandbox is False:
            resolved = "production"
        else:
            raise RuntimeSafetyError("an explicit --environment or --sandbox/--no-sandbox is required")
    else:
        resolved = environment
        if legacy_sandbox is not None and legacy_sandbox != (resolved == "sandbox"):
            raise RuntimeSafetyError("--environment conflicts with --sandbox/--no-sandbox")

    supplied_identity = (expected_account_id, expected_account_id_key, expected_institution_type)
    if resolved == "sandbox" and not any(supplied_identity):
        return RuntimeSafetyBoundary("sandbox", None, None, None)
    account_id = _required_text(expected_account_id, "expected account id")
    account_key = _required_text(expected_account_id_key, "expected account key")
    institution_type = _required_text(expected_institution_type, "expected institution type")

    if resolved == "sandbox":
        return RuntimeSafetyBoundary("sandbox", account_id, account_key, institution_type)

    secret = _require_arm_secret(production_arm_secret)
    if production_arm_file is None:
        raise RuntimeSafetyError("--production-arm-file is required for production")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    document, issued_at, expiry = _read_production_arm(production_arm_file, secret, current_time)
    required = {
        "environment": "production",
        "expected_account_id": account_id,
        "expected_account_id_key": account_key,
        "expected_institution_type": institution_type,
    }
    if any(document.get(field) != value for field, value in required.items()):
        raise RuntimeSafetyError("production arm identity does not match runtime identity")
    return RuntimeSafetyBoundary(
        "production",
        account_id,
        account_key,
        institution_type,
        issued_at,
        expiry,
    )


def validate_dashboard_credentials(settings: dict[str, Any]) -> None:
    """Reject credentials that make an order-capable dashboard unsafe to start."""

    if not isinstance(settings, dict):
        raise RuntimeSafetyError("live settings are invalid")
    username = _required_text(settings.get("dashboard_user"), "dashboard username")
    password = _required_text(settings.get("dashboard_pass"), "dashboard password")
    pin = _required_text(settings.get("pin"), "dashboard action PIN")
    if username.lower() in {"admin", "user", "etrade"}:
        raise RuntimeSafetyError("dashboard username is a reserved default")
    if len(password) < 16 or password.lower() in {"password", "changeme", "default"}:
        raise RuntimeSafetyError("dashboard password is weak")
    if (
        len(pin) < 8
        or len(pin) > 64
        or not pin.isdigit()
        or pin in {"00000000", "12345678", "1234"}
    ):
        raise RuntimeSafetyError("dashboard action PIN is weak")


def main(argv: list[str] | None = None) -> int:
    """Issue a short-lived production arm without accepting a CLI secret."""

    parser = argparse.ArgumentParser(description="Runtime safety operator commands")
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue-production-arm")
    issue.add_argument("--output", required=True, type=Path)
    issue.add_argument("--expected-account-id", required=True)
    issue.add_argument("--expected-account-id-key", required=True)
    issue.add_argument("--expected-institution-type", required=True)
    issue.add_argument("--ttl-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if args.command == "issue-production-arm":
            issue_production_arm(
                output=args.output,
                expected_account_id=args.expected_account_id,
                expected_account_id_key=args.expected_account_id_key,
                expected_institution_type=args.expected_institution_type,
                ttl_seconds=args.ttl_seconds,
                secret=os.getenv("ETRADE_PRODUCTION_ARMING_SECRET"),
            )
            return 0
    except RuntimeSafetyError as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable")


__all__ = [
    "RuntimeSafetyBoundary",
    "RuntimeSafetyError",
    "ARM_SCHEMA_VERSION",
    "MAX_PRODUCTION_ARM_LIFETIME",
    "OwnerOnlyRotatingFileHandler",
    "build_runtime_safety_boundary",
    "configure_owner_only_logger",
    "content_fingerprint",
    "issue_production_arm",
    "main",
    "production_arm_signature",
    "payload_fingerprint",
    "redact_http_headers",
    "read_owner_only_json",
    "resolve_etrade_consumer_key",
    "secure_lock_file",
    "secure_append_text",
    "validate_dashboard_credentials",
    "write_owner_only_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
