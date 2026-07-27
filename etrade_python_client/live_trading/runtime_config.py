"""Strict, side-effect-free runtime configuration for live services.

Configuration describes operational intent.  It is never authority to mutate
an E*TRADE account: schema version 1 rejects mutation enablement in every mode.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA_VERSION = 1
SECRET_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_PATH_LENGTH = 4096
MAX_SIGNED_SQLITE_INTEGER = (1 << 63) - 1
RUNTIME_MODES = frozenset({"sandbox", "shadow", "paper", "live"})

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "runtime_root",
        "strategy",
        "data",
        "model",
        "execution",
        "risk",
    }
)
_STRATEGY_FIELDS = frozenset({"enabled", "strategy_id", "symbols"})
_DATA_FIELDS = frozenset(
    {"require_complete_snapshots", "max_snapshot_age_seconds"}
)
_MODEL_FIELDS = frozenset(
    {"enabled", "required_for_entry", "max_signal_age_seconds"}
)
_EXECUTION_FIELDS = frozenset(
    {
        "selected_account_id_key",
        "account_allowlist",
        "broker_mutations_enabled",
    }
)
_ACCOUNT_FIELDS = frozenset(
    {"account_id", "account_id_key", "institution_type"}
)
_URLSAFE_32_BYTE_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_RISK_FIELDS = frozenset(
    {
        "max_order_contracts",
        "max_order_loss_cents",
        "max_account_open_risk_cents",
        "max_daily_loss_cents",
        "max_quote_age_seconds",
    }
)
_SECRET_FIELDS = frozenset(
    {
        "schema_version",
        "etrade_consumer_key",
        "etrade_consumer_secret",
        "etrade_username",
        "etrade_password",
        "dashboard_username",
        "dashboard_password",
        "dashboard_pin",
        "dashboard_session_secret",
    }
)
_SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]{0,14}")
_STRATEGY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_ACCOUNT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_INSTITUTION_PATTERN = re.compile(r"[A-Z][A-Z0-9_ -]{0,63}")
_PLACEHOLDER_SECRETS = frozenset(
    {
        "admin",
        "changeme",
        "default",
        "default_live_key",
        "default_live_secret",
        "default_sandbox_key",
        "default_sandbox_secret",
        "password",
        "replace-me",
        "replace_me",
        "secret",
    }
)


class RuntimeConfigError(ValueError):
    """Raised when runtime configuration or secrets violate the contract."""


@dataclass(frozen=True, slots=True, repr=False)
class AccountIdentity:
    account_id: str
    account_id_key: str
    institution_type: str

    def __repr__(self) -> str:
        return "AccountIdentity([REDACTED])"


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    enabled: bool
    strategy_id: str
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DataConfig:
    require_complete_snapshots: bool
    max_snapshot_age_seconds: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    enabled: bool
    required_for_entry: bool
    max_signal_age_seconds: int


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    selected_account_id_key: str | None = field(repr=False)
    account_allowlist: tuple[AccountIdentity, ...]
    broker_mutations_enabled: bool


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_order_contracts: int
    max_order_loss_cents: int
    max_account_open_risk_cents: int
    max_daily_loss_cents: int
    max_quote_age_seconds: int


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Absolute paths derived from the single configured runtime root."""

    root: Path
    state_dir: Path
    cache_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    execution_dir: Path
    data_dir: Path
    model_dir: Path
    secrets_file: Path
    oauth_file: Path
    production_arm_file: Path
    runtime_lock_file: Path
    live_settings_file: Path
    trade_status_file: Path
    manual_trade_status_file: Path
    spy_tracker_file: Path
    spy_gains_cache_file: Path
    order_intent_ledger_file: Path
    client_log_file: Path
    dashboard_log_file: Path
    order_audit_log_file: Path
    positions_artifact_file: Path
    regime_shadow_file: Path
    spy_vix_price_cache_file: Path

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.state_dir,
            self.cache_dir,
            self.logs_dir,
            self.artifacts_dir,
            self.execution_dir,
            self.data_dir,
            self.model_dir,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    schema_version: int
    mode: str
    paths: RuntimePaths
    strategy: StrategyConfig
    data: DataConfig
    model: ModelConfig
    execution: ExecutionConfig
    risk: RiskConfig
    source_path: Path
    source_sha256: str

    @property
    def broker_environment(self) -> str | None:
        if self.mode == "sandbox":
            return "sandbox"
        if self.mode in {"shadow", "live"}:
            return "production"
        return None

    @property
    def broker_mutations_authorized(self) -> bool:
        """Configuration is not a broker-mutation capability."""

        return False

    @property
    def starts_armed(self) -> bool:
        return False

    @property
    def selected_account(self) -> AccountIdentity | None:
        selected_key = self.execution.selected_account_id_key
        if selected_key is None:
            return None
        for account in self.execution.account_allowlist:
            if account.account_id_key == selected_key:
                return account
        raise AssertionError("validated selected account is unavailable")

    def verify_broker_account(
        self,
        account: Mapping[str, Any],
    ) -> AccountIdentity:
        """Require the broker-selected account to match the exact triple."""

        selected = self.selected_account
        if selected is None or not isinstance(account, Mapping):
            raise RuntimeConfigError(
                "runtime mode does not have a verifiable broker account"
            )
        expected = {
            "accountId": selected.account_id,
            "accountIdKey": selected.account_id_key,
            "institutionType": selected.institution_type,
        }
        if any(account.get(name) != value for name, value in expected.items()):
            raise RuntimeConfigError(
                "broker-selected account does not match the configured identity"
            )
        return selected


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeSecrets:
    etrade_consumer_key: str | None
    etrade_consumer_secret: str | None
    etrade_username: str | None
    etrade_password: str | None
    dashboard_username: str
    dashboard_password: str
    dashboard_pin: str
    dashboard_session_secret: str
    source: str

    def __repr__(self) -> str:
        configured = tuple(
            field
            for field in (
                "etrade_consumer_key",
                "etrade_consumer_secret",
                "etrade_username",
                "etrade_password",
                "dashboard_username",
                "dashboard_password",
                "dashboard_pin",
                "dashboard_session_secret",
            )
            if getattr(self, field) is not None
        )
        return f"RuntimeSecrets(configured_fields={configured!r}, source={self.source!r})"


@dataclass(frozen=True, slots=True, repr=False)
class DashboardSecrets:
    """Least-privilege credentials for the read-only operator process."""

    username: str
    password: str
    session_secret: str
    source: str

    def __repr__(self) -> str:
        return f"DashboardSecrets(source={self.source!r}, values=[REDACTED])"


def _reject_constant(value: str) -> None:
    raise RuntimeConfigError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except RuntimeConfigError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeConfigError(f"{label} is not valid UTF-8 JSON") from exc


def _trusted_file_bytes(
    path: str | Path,
    *,
    label: str,
    owner_only: bool,
) -> tuple[Path, bytes]:
    try:
        candidate = Path(path).expanduser().absolute()
    except (OSError, RuntimeError, UnicodeError) as exc:
        raise RuntimeConfigError(f"{label} path is invalid") from exc
    _validate_existing_runtime_components(
        candidate.parent,
        allow_sticky_final=True,
    )
    canonical_candidate = (
        Path(os.path.realpath(candidate.parent)) / candidate.name
    )
    parent_descriptor = -1
    descriptor = -1
    try:
        try:
            parent_before = os.lstat(candidate.parent)
            parent_mode = stat.S_IMODE(parent_before.st_mode)
            if (
                stat.S_ISLNK(parent_before.st_mode)
                or not stat.S_ISDIR(parent_before.st_mode)
                or parent_before.st_uid not in {0, os.geteuid()}
                or (
                    parent_mode & 0o022
                    and not (
                        parent_before.st_uid == 0
                        and parent_mode & stat.S_ISVTX
                    )
                )
            ):
                raise RuntimeConfigError(f"{label} parent is unsafe")
            parent_descriptor = os.open(
                candidate.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_after = os.fstat(parent_descriptor)
            if (
                parent_after.st_dev != parent_before.st_dev
                or parent_after.st_ino != parent_before.st_ino
            ):
                raise RuntimeConfigError(
                    f"{label} parent changed while opening"
                )
            before = os.lstat(candidate)
            forbidden_mode = 0o077 if owner_only else 0o022
            accepted_owners = (
                {os.geteuid()} if owner_only else {0, os.geteuid()}
            )
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_uid not in accepted_owners
                or stat.S_IMODE(before.st_mode) & forbidden_mode
                or before.st_size > MAX_CONFIG_BYTES
            ):
                qualifier = "owner-only " if owner_only else ""
                raise RuntimeConfigError(
                    f"{label} must be a safe {qualifier}regular file"
                )
            descriptor = os.open(
                candidate.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise RuntimeConfigError(f"{label} is unavailable") from exc

        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or not stat.S_ISREG(after.st_mode)
        ):
            raise RuntimeConfigError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = MAX_CONFIG_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_CONFIG_BYTES:
            raise RuntimeConfigError(f"{label} is too large")
        final = os.fstat(descriptor)
        if (
            final.st_dev != after.st_dev
            or final.st_ino != after.st_ino
            or final.st_size != after.st_size
            or final.st_mtime_ns != after.st_mtime_ns
            or final.st_ctime_ns != after.st_ctime_ns
            or len(payload) != final.st_size
        ):
            raise RuntimeConfigError(f"{label} changed while reading")
        return canonical_candidate, payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if unknown:
            detail.append(f"unknown={unknown}")
        raise RuntimeConfigError(f"{label} fields are invalid ({', '.join(detail)})")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeConfigError(f"{label} must be a boolean")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RuntimeConfigError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _text(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
    maximum: int = 256,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise RuntimeConfigError(
            f"{label} must be text with length in [{minimum}, {maximum}]"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeConfigError(f"{label} is not valid Unicode text") from exc
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise RuntimeConfigError(f"{label} contains forbidden whitespace or control characters")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise RuntimeConfigError(f"{label} has an invalid format")
    return value


def _optional_text(
    value: Any,
    label: str,
    *,
    maximum: int = 256,
    pattern: re.Pattern[str] | None = None,
) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum=maximum, pattern=pattern)


def _runtime_paths(raw_root: Any, config_path: Path) -> RuntimePaths:
    root_text = _text(raw_root, "runtime_root", maximum=MAX_PATH_LENGTH)
    if "\x00" in root_text or "\\" in root_text:
        raise RuntimeConfigError("runtime_root contains a forbidden character")
    slash_parts = root_text.split("/")
    if (
        "" in slash_parts[1:]
        or (not root_text.startswith("/") and "" in slash_parts)
        or any(part in {".", ".."} for part in slash_parts)
    ):
        raise RuntimeConfigError(
            "runtime_root contains an empty or traversal path component"
        )
    raw_path = PurePath(root_text)
    if any(part in {".", ".."} for part in raw_path.parts):
        raise RuntimeConfigError("runtime_root may not contain '.' or '..'")
    if raw_path.is_absolute():
        root = Path(root_text)
    else:
        root = config_path.parent / root_text
    root = Path(os.path.abspath(root))
    if root == Path(root.anchor) or len(str(root)) > MAX_PATH_LENGTH:
        raise RuntimeConfigError("runtime_root is too broad or too long")
    _validate_existing_runtime_components(root)
    root = Path(os.path.realpath(root))

    state = root / "state"
    cache = root / "cache"
    logs = root / "logs"
    artifacts = root / "artifacts"
    execution = root / "execution"
    data = root / "data"
    model = root / "model"
    return RuntimePaths(
        root=root,
        state_dir=state,
        cache_dir=cache,
        logs_dir=logs,
        artifacts_dir=artifacts,
        execution_dir=execution,
        data_dir=data,
        model_dir=model,
        secrets_file=state / "secrets.json",
        oauth_file=state / "oauth.json",
        production_arm_file=state / "production-arm.json",
        runtime_lock_file=state / "runtime.lock",
        live_settings_file=state / "legacy-settings.json",
        trade_status_file=state / "trade-status.json",
        manual_trade_status_file=state / "manual-trade-status.json",
        spy_tracker_file=state / "spy-tracking.json",
        spy_gains_cache_file=cache / "spy-gains.json",
        order_intent_ledger_file=execution / "order-intents.sqlite3",
        client_log_file=logs / "client.log",
        dashboard_log_file=logs / "dashboard.log",
        order_audit_log_file=logs / "order-audit.csv",
        positions_artifact_file=artifacts / "positions.html",
        regime_shadow_file=model / "regime-v2-shadow.json",
        spy_vix_price_cache_file=cache / "spy-vix-prices.json",
    )


def _validate_existing_runtime_components(
    path: Path,
    *,
    allow_sticky_final: bool = False,
) -> None:
    """Reject user-controlled symlinks and unsafe existing directories."""

    current = Path(path.anchor)
    shared_sticky_ancestor = False
    protected_after_shared_ancestor = False
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if (
                shared_sticky_ancestor
                and not protected_after_shared_ancestor
            ):
                raise RuntimeConfigError(
                    "runtime_root lacks a private directory below a shared parent"
                )
            return
        except (OSError, UnicodeError) as exc:
            raise RuntimeConfigError("runtime_root cannot be inspected safely") from exc
        if stat.S_ISLNK(metadata.st_mode):
            # POSIX symlink mode bits are not access controls.  A root-owned
            # alias (for example macOS /var -> /private/var) is trusted;
            # service-user-controlled aliases are not.
            if current != path and metadata.st_uid == 0:
                continue
            raise RuntimeConfigError(
                "runtime_root may not traverse an untrusted symbolic link"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeConfigError("runtime_root has a non-directory component")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise RuntimeConfigError(
                "runtime_root traverses a directory owned by another user"
            )
        mode = stat.S_IMODE(metadata.st_mode)
        trusted_sticky_directory = (
            metadata.st_uid == 0
            and bool(mode & stat.S_ISVTX)
            and (current != path or allow_sticky_final)
        )
        if mode & 0o022 and not trusted_sticky_directory:
            raise RuntimeConfigError(
                "runtime_root traverses a group- or world-writable directory"
            )
        if trusted_sticky_directory:
            shared_sticky_ancestor = True
            protected_after_shared_ancestor = False
        elif shared_sticky_ancestor:
            protected_after_shared_ancestor = True
    if (
        shared_sticky_ancestor
        and not protected_after_shared_ancestor
        and not allow_sticky_final
    ):
        raise RuntimeConfigError(
            "runtime_root lacks a private directory below a shared parent"
        )


def _parse_strategy(value: Any) -> StrategyConfig:
    raw = _object(value, _STRATEGY_FIELDS, "strategy")
    enabled = _boolean(raw["enabled"], "strategy.enabled")
    strategy_id = _text(
        raw["strategy_id"],
        "strategy.strategy_id",
        maximum=64,
        pattern=_STRATEGY_ID_PATTERN,
    )
    if not isinstance(raw["symbols"], list):
        raise RuntimeConfigError("strategy.symbols must be an array")
    symbols = tuple(
        _text(
            symbol,
            f"strategy.symbols[{index}]",
            maximum=15,
            pattern=_SYMBOL_PATTERN,
        )
        for index, symbol in enumerate(raw["symbols"])
    )
    if len(set(symbols)) != len(symbols):
        raise RuntimeConfigError("strategy.symbols must be unique")
    if enabled and (not symbols or strategy_id == "disabled"):
        raise RuntimeConfigError("enabled strategy requires an id and symbols")
    if not enabled and (symbols or strategy_id != "disabled"):
        raise RuntimeConfigError("disabled strategy must use id 'disabled' and no symbols")
    return StrategyConfig(enabled=enabled, strategy_id=strategy_id, symbols=symbols)


def _parse_data(value: Any) -> DataConfig:
    raw = _object(value, _DATA_FIELDS, "data")
    complete = _boolean(
        raw["require_complete_snapshots"],
        "data.require_complete_snapshots",
    )
    if not complete:
        raise RuntimeConfigError("schema v1 requires complete snapshots")
    return DataConfig(
        require_complete_snapshots=complete,
        max_snapshot_age_seconds=_integer(
            raw["max_snapshot_age_seconds"],
            "data.max_snapshot_age_seconds",
            1,
            86_400,
        ),
    )


def _parse_model(value: Any) -> ModelConfig:
    raw = _object(value, _MODEL_FIELDS, "model")
    enabled = _boolean(raw["enabled"], "model.enabled")
    required = _boolean(
        raw["required_for_entry"],
        "model.required_for_entry",
    )
    if required and not enabled:
        raise RuntimeConfigError("required entry model must be enabled")
    return ModelConfig(
        enabled=enabled,
        required_for_entry=required,
        max_signal_age_seconds=_integer(
            raw["max_signal_age_seconds"],
            "model.max_signal_age_seconds",
            1,
            86_400,
        ),
    )


def _parse_account(value: Any, index: int) -> AccountIdentity:
    raw = _object(
        value,
        _ACCOUNT_FIELDS,
        f"execution.account_allowlist[{index}]",
    )
    return AccountIdentity(
        account_id=_text(
            raw["account_id"],
            f"execution.account_allowlist[{index}].account_id",
            maximum=128,
            pattern=_ACCOUNT_ID_PATTERN,
        ),
        account_id_key=_text(
            raw["account_id_key"],
            f"execution.account_allowlist[{index}].account_id_key",
            maximum=128,
            pattern=_ACCOUNT_ID_PATTERN,
        ),
        institution_type=_text(
            raw["institution_type"],
            f"execution.account_allowlist[{index}].institution_type",
            maximum=64,
            pattern=_INSTITUTION_PATTERN,
        ),
    )


def _parse_execution(value: Any, mode: str) -> ExecutionConfig:
    raw = _object(value, _EXECUTION_FIELDS, "execution")
    selected_key = _optional_text(
        raw["selected_account_id_key"],
        "execution.selected_account_id_key",
        maximum=128,
        pattern=_ACCOUNT_ID_PATTERN,
    )
    if not isinstance(raw["account_allowlist"], list):
        raise RuntimeConfigError("execution.account_allowlist must be an array")
    accounts = tuple(
        _parse_account(account, index)
        for index, account in enumerate(raw["account_allowlist"])
    )
    keys = tuple(account.account_id_key for account in accounts)
    if len(set(keys)) != len(keys):
        raise RuntimeConfigError("account allowlist keys must be unique")
    mutations = _boolean(
        raw["broker_mutations_enabled"],
        "execution.broker_mutations_enabled",
    )
    if mutations:
        raise RuntimeConfigError("schema v1 never authorizes broker mutations")
    if mode == "paper":
        if selected_key is not None or accounts:
            raise RuntimeConfigError("paper mode may not select a broker account")
    elif selected_key is None or not accounts:
        raise RuntimeConfigError(f"{mode} mode requires an exact account allowlist")
    elif keys.count(selected_key) != 1:
        raise RuntimeConfigError("selected account key is not uniquely allowlisted")
    return ExecutionConfig(
        selected_account_id_key=selected_key,
        account_allowlist=accounts,
        broker_mutations_enabled=mutations,
    )


def _parse_risk(value: Any) -> RiskConfig:
    raw = _object(value, _RISK_FIELDS, "risk")
    return RiskConfig(
        max_order_contracts=_integer(
            raw["max_order_contracts"],
            "risk.max_order_contracts",
            0,
            1_000_000,
        ),
        max_order_loss_cents=_integer(
            raw["max_order_loss_cents"],
            "risk.max_order_loss_cents",
            0,
            MAX_SIGNED_SQLITE_INTEGER,
        ),
        max_account_open_risk_cents=_integer(
            raw["max_account_open_risk_cents"],
            "risk.max_account_open_risk_cents",
            0,
            MAX_SIGNED_SQLITE_INTEGER,
        ),
        max_daily_loss_cents=_integer(
            raw["max_daily_loss_cents"],
            "risk.max_daily_loss_cents",
            0,
            MAX_SIGNED_SQLITE_INTEGER,
        ),
        max_quote_age_seconds=_integer(
            raw["max_quote_age_seconds"],
            "risk.max_quote_age_seconds",
            1,
            300,
        ),
    )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load and validate one immutable schema-v1 configuration document.

    Loading reads only ``path``.  It does not create runtime directories,
    resolve secrets, import a broker client, or touch the network.
    """

    source_path, payload = _trusted_file_bytes(
        path,
        label="runtime configuration",
        owner_only=False,
    )
    document = _object(
        _decode_json(payload, "runtime configuration"),
        _TOP_LEVEL_FIELDS,
        "runtime configuration",
    )
    schema_version = _integer(
        document["schema_version"],
        "schema_version",
        CONFIG_SCHEMA_VERSION,
        CONFIG_SCHEMA_VERSION,
    )
    mode = _text(document["mode"], "mode", maximum=16)
    if mode not in RUNTIME_MODES:
        raise RuntimeConfigError(f"mode must be one of {sorted(RUNTIME_MODES)}")
    paths = _runtime_paths(document["runtime_root"], source_path)
    strategy = _parse_strategy(document["strategy"])
    data = _parse_data(document["data"])
    model = _parse_model(document["model"])
    execution = _parse_execution(document["execution"], mode)
    risk = _parse_risk(document["risk"])
    if risk.max_order_loss_cents > risk.max_account_open_risk_cents:
        raise RuntimeConfigError(
            "max order loss may not exceed account opening-risk capacity"
        )
    if strategy.enabled and (
        risk.max_order_contracts == 0
        or risk.max_order_loss_cents == 0
        or risk.max_account_open_risk_cents == 0
        or risk.max_daily_loss_cents == 0
    ):
        raise RuntimeConfigError("enabled strategy requires positive risk limits")
    return RuntimeConfig(
        schema_version=schema_version,
        mode=mode,
        paths=paths,
        strategy=strategy,
        data=data,
        model=model,
        execution=execution,
        risk=risk,
        source_path=source_path,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _validate_runtime_directory_set(
    directories: tuple[Path, ...],
) -> None:
    for directory in directories:
        try:
            metadata = os.lstat(directory)
        except OSError as exc:
            raise RuntimeConfigError(
                f"runtime directory is unavailable: {directory}"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RuntimeConfigError(
                f"runtime directory must be owner-only and non-symbolic: {directory}"
            )


def validate_runtime_directories(paths: RuntimePaths) -> None:
    """Require every pre-created owner-only runtime directory."""

    _validate_runtime_directory_set(paths.directories)


def validate_read_only_runtime_directories(paths: RuntimePaths) -> None:
    """Validate only directories exposed to the broker-isolated dashboard."""

    _validate_runtime_directory_set(
        (
            paths.root,
            paths.artifacts_dir,
            paths.model_dir,
        )
    )


def _environment_secret_values(
    config: RuntimeConfig,
    environ: Mapping[str, str],
) -> dict[str, str | None]:
    consumer_prefix = (
        "SANDBOX" if config.mode == "sandbox" else "LIVE"
    )
    broker_mode = config.broker_environment is not None
    return {
        "etrade_consumer_key": (
            environ.get(f"ETRADE_{consumer_prefix}_CONSUMER_KEY")
            if broker_mode
            else None
        ),
        "etrade_consumer_secret": (
            environ.get(f"ETRADE_{consumer_prefix}_CONSUMER_SECRET")
            if broker_mode
            else None
        ),
        "etrade_username": environ.get("ETRADE_USER") if broker_mode else None,
        "etrade_password": environ.get("ETRADE_PASS") if broker_mode else None,
        "dashboard_username": environ.get("ETRADE_DASHBOARD_USER"),
        "dashboard_password": environ.get("ETRADE_DASHBOARD_PASSWORD"),
        "dashboard_pin": environ.get("ETRADE_DASHBOARD_PIN"),
        "dashboard_session_secret": environ.get(
            "ETRADE_DASHBOARD_SESSION_SECRET"
        ),
    }


def _required_secret_names(config: RuntimeConfig) -> tuple[str, ...]:
    dashboard = (
        "dashboard_username",
        "dashboard_password",
        "dashboard_pin",
        "dashboard_session_secret",
    )
    if config.broker_environment is None:
        return dashboard
    return (
        "etrade_consumer_key",
        "etrade_consumer_secret",
        "etrade_username",
        "etrade_password",
        *dashboard,
    )


def _secret_file_values(path: Path) -> dict[str, str | None]:
    _source, payload = _trusted_file_bytes(
        path,
        label="runtime secret fallback",
        owner_only=True,
    )
    raw = _object(
        _decode_json(payload, "runtime secret fallback"),
        _SECRET_FIELDS,
        "runtime secret fallback",
    )
    _integer(
        raw["schema_version"],
        "runtime secret schema_version",
        SECRET_SCHEMA_VERSION,
        SECRET_SCHEMA_VERSION,
    )
    return {
        name: (
            None
            if raw[name] is None
            else _text(raw[name], f"runtime secret {name}", maximum=4096)
        )
        for name in _SECRET_FIELDS - {"schema_version"}
    }


def _validate_secret_values(
    config: RuntimeConfig,
    values: Mapping[str, str | None],
) -> None:
    required = _required_secret_names(config)
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise RuntimeConfigError(
            f"required runtime secrets are missing: {sorted(missing)}"
        )
    _validate_secret_strings(values, required)
    _validate_dashboard_credentials(values)
    pin = values["dashboard_pin"]
    assert pin is not None
    if (
        not 8 <= len(pin) <= 64
        or len(set(pin)) == 1
        or pin.lower()
        in {
            "00000000",
            "12345678",
            "87654321",
            "changeme",
            "password",
        }
    ):
        raise RuntimeConfigError("dashboard PIN is weak")
    if config.broker_environment is not None:
        for name in (
            "etrade_consumer_key",
            "etrade_consumer_secret",
            "etrade_username",
            "etrade_password",
        ):
            value = values[name]
            assert value is not None
            if len(value) < 8:
                raise RuntimeConfigError(f"runtime secret {name} is too short")


def _validate_secret_strings(
    values: Mapping[str, str | None],
    names: tuple[str, ...],
) -> None:
    for name in names:
        value = values[name]
        assert value is not None
        _text(value, f"runtime secret {name}", maximum=4096)
        if value.strip().lower() in _PLACEHOLDER_SECRETS:
            raise RuntimeConfigError(f"runtime secret {name} is a placeholder")


def _validate_dashboard_credentials(
    values: Mapping[str, str | None],
) -> None:
    username = values["dashboard_username"]
    password = values["dashboard_password"]
    session_secret = values["dashboard_session_secret"]
    assert username is not None
    assert password is not None
    assert session_secret is not None
    if username.lower() in {"admin", "etrade", "user"}:
        raise RuntimeConfigError("dashboard username is a reserved default")
    if len(password) < 16 or len(set(password)) < 8:
        raise RuntimeConfigError("dashboard password is weak")
    if len(session_secret) < 43 or len(set(session_secret)) < 12:
        raise RuntimeConfigError("dashboard session secret is weak")


def resolve_dashboard_secrets(
    *,
    environ: Mapping[str, str] | None = None,
) -> DashboardSecrets:
    """Resolve only credentials needed by the broker-isolated dashboard.

    This intentionally has no owner-file fallback: the read-only process never
    opens the combined broker secret document or reads broker environment keys.
    """

    environment = os.environ if environ is None else environ
    values = {
        "dashboard_username": environment.get("ETRADE_DASHBOARD_USER"),
        "dashboard_password": environment.get("ETRADE_DASHBOARD_PASSWORD"),
        "dashboard_session_secret": environment.get(
            "ETRADE_DASHBOARD_SESSION_SECRET"
        ),
    }
    required = (
        "dashboard_username",
        "dashboard_password",
        "dashboard_session_secret",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise RuntimeConfigError(
            f"required dashboard secrets are missing: {sorted(missing)}"
        )
    _validate_secret_strings(values, required)
    _validate_dashboard_credentials(values)
    return DashboardSecrets(
        username=values["dashboard_username"] or "",
        password=values["dashboard_password"] or "",
        session_secret=values["dashboard_session_secret"] or "",
        source="environment",
    )


def resolve_positions_artifact_hmac_key(
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the environment-only key shared by publisher and reader."""

    environment = os.environ if environ is None else environ
    value = environment.get("ETRADE_POSITIONS_ARTIFACT_HMAC_KEY")
    values = {"positions_artifact_hmac_key": value}
    required = ("positions_artifact_hmac_key",)
    if not value:
        raise RuntimeConfigError(
            "required positions artifact signing secret is missing"
        )
    _validate_secret_strings(values, required)
    try:
        decoded = base64.b64decode(
            value + "=",
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise RuntimeConfigError(
            "positions artifact signing secret must be a canonical "
            "32-byte URL-safe token"
        ) from exc
    periodic = any(
        all(
            character == value[index % width]
            for index, character in enumerate(value)
        )
        for width in range(1, len(value) // 2 + 1)
    )
    if (
        _URLSAFE_32_BYTE_TOKEN.fullmatch(value) is None
        or len(decoded) != 32
        or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
        != value
        or len(set(value)) < 12
        or periodic
    ):
        raise RuntimeConfigError(
            "positions artifact signing secret must be a canonical, "
            "non-repeating 32-byte URL-safe token"
        )
    return value


def validate_positions_artifact_key_separation(
    artifact_key: str,
    dashboard_secrets: DashboardSecrets,
) -> None:
    """Reject reuse of browser-facing authentication secrets."""

    if type(artifact_key) is not str or not artifact_key:
        raise TypeError("artifact_key must be a non-empty exact string")
    if type(dashboard_secrets) is not DashboardSecrets:
        raise TypeError(
            "dashboard_secrets must be exact DashboardSecrets"
        )
    for name, value in (
        ("dashboard password", dashboard_secrets.password),
        ("dashboard session secret", dashboard_secrets.session_secret),
    ):
        if hmac.compare_digest(
            artifact_key.encode("utf-8"),
            value.encode("utf-8"),
        ):
            raise RuntimeConfigError(
                "positions artifact signing secret must not reuse "
                f"the {name}"
            )


def positions_artifact_runtime_binding(config: RuntimeConfig) -> str:
    """Hash runtime, environment, and account identity without exposing it."""

    if type(config) is not RuntimeConfig:
        raise TypeError("config must be an exact RuntimeConfig")
    account = config.selected_account
    canonical = {
        "account": (
            None
            if account is None
            else {
                "account_id": account.account_id,
                "account_id_key": account.account_id_key,
                "institution_type": account.institution_type,
            }
        ),
        "broker_environment": config.broker_environment,
        "mode": config.mode,
        "runtime_root": str(config.paths.root),
        "schema_version": config.schema_version,
        "source_sha256": config.source_sha256,
    }
    return hashlib.sha256(
        b"etrade-positions-runtime-binding.v1\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def resolve_runtime_secrets(
    config: RuntimeConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeSecrets:
    """Resolve service-injected secrets, with a local owner-only fallback.

    Live mode never reads the fallback file.  The independent production
    arming secret is intentionally not accepted by this API.
    """

    environment = os.environ if environ is None else environ
    values = _environment_secret_values(config, environment)
    required = _required_secret_names(config)
    source = "environment"
    if any(not values.get(name) for name in required):
        if config.mode == "live":
            raise RuntimeConfigError(
                "live mode requires service-injected environment secrets"
            )
        fallback = _secret_file_values(config.paths.secrets_file)
        values = {
            name: values.get(name) or fallback.get(name)
            for name in values
        }
        source = "environment+owner-only-file"
    if config.broker_environment is None:
        for name in (
            "etrade_consumer_key",
            "etrade_consumer_secret",
            "etrade_username",
            "etrade_password",
        ):
            values[name] = None
    _validate_secret_values(config, values)
    return RuntimeSecrets(
        etrade_consumer_key=values["etrade_consumer_key"],
        etrade_consumer_secret=values["etrade_consumer_secret"],
        etrade_username=values["etrade_username"],
        etrade_password=values["etrade_password"],
        dashboard_username=values["dashboard_username"] or "",
        dashboard_password=values["dashboard_password"] or "",
        dashboard_pin=values["dashboard_pin"] or "",
        dashboard_session_secret=values["dashboard_session_secret"] or "",
        source=source,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate immutable E*TRADE runtime configuration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--check-directories", action="store_true")
    validate.add_argument("--check-secrets", action="store_true")
    validate.add_argument("--check-dashboard-secrets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        if args.check_directories:
            validate_runtime_directories(config.paths)
        if args.check_secrets:
            resolve_runtime_secrets(config)
        if args.check_dashboard_secrets:
            dashboard_secrets = resolve_dashboard_secrets()
            artifact_key = resolve_positions_artifact_hmac_key()
            validate_positions_artifact_key_separation(
                artifact_key,
                dashboard_secrets,
            )
    except RuntimeConfigError as exc:
        print(f"runtime configuration invalid: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": config.schema_version,
                "mode": config.mode,
                "broker_environment": config.broker_environment,
                "broker_mutations_authorized": False,
                "runtime_root": str(config.paths.root),
                "source_sha256": config.source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "AccountIdentity",
    "DashboardSecrets",
    "DataConfig",
    "ExecutionConfig",
    "ModelConfig",
    "RiskConfig",
    "RuntimeConfig",
    "RuntimeConfigError",
    "RuntimePaths",
    "RuntimeSecrets",
    "StrategyConfig",
    "load_runtime_config",
    "main",
    "positions_artifact_runtime_binding",
    "resolve_dashboard_secrets",
    "resolve_positions_artifact_hmac_key",
    "resolve_runtime_secrets",
    "validate_positions_artifact_key_separation",
    "validate_read_only_runtime_directories",
    "validate_runtime_directories",
]
