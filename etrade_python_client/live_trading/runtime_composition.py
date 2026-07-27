"""Typed, read-only runtime composition.

This module is the only startup seam that combines the static runtime
configuration with runtime secrets.  It deliberately constructs no OAuth
client, broker session, order gateway, or mutable legacy settings object.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from live_trading.positions_artifact import (
    PositionsArtifactReader,
    PositionsArtifactSigningKey,
)
from live_trading.regime_shadow_store import RegimeShadowReader
from live_trading.runtime_config import (
    DashboardSecrets,
    RuntimeConfig,
    load_runtime_config,
    positions_artifact_runtime_binding,
    resolve_dashboard_secrets,
    resolve_positions_artifact_hmac_key,
    validate_positions_artifact_key_separation,
    validate_read_only_runtime_directories,
)


SESSION_SCHEMA = "dashboard-session.v1"
MIN_SESSION_SECONDS = 60
MAX_SESSION_SECONDS = 7 * 24 * 60 * 60


class RuntimeCompositionError(RuntimeError):
    """Raised when independently valid runtime inputs cannot be composed."""


def _utc_timestamp(value: datetime | None) -> int:
    if value is None:
        return int(datetime.now(timezone.utc).timestamp())
    if type(value) is not datetime or value.tzinfo is None:
        raise TypeError("now must be an exact timezone-aware datetime")
    return int(value.astimezone(timezone.utc).timestamp())


def _exact_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class DashboardAuth:
    """Immutable dashboard credentials and stateless-session signer."""

    username: str
    password: str
    session_secret: str
    audience: str

    @classmethod
    def from_dashboard_secrets(
        cls,
        secrets: DashboardSecrets,
        *,
        audience: str,
    ) -> "DashboardAuth":
        if type(secrets) is not DashboardSecrets:
            raise TypeError(
                "secrets must be an exact DashboardSecrets instance"
            )
        return cls(
            username=_exact_text(secrets.username, "dashboard username"),
            password=_exact_text(secrets.password, "dashboard password"),
            session_secret=_exact_text(
                secrets.session_secret,
                "dashboard session secret",
            ),
            audience=_exact_text(audience, "dashboard audience"),
        )

    def __post_init__(self) -> None:
        _exact_text(self.username, "dashboard username")
        _exact_text(self.password, "dashboard password")
        _exact_text(self.session_secret, "dashboard session secret")
        _exact_text(self.audience, "dashboard audience")

    def __repr__(self) -> str:
        return "DashboardAuth([REDACTED])"

    def verify_login(self, username: object, password: object) -> bool:
        """Compare both login fields without exposing either value."""

        supplied_username = username if type(username) is str else ""
        supplied_password = password if type(password) is str else ""
        username_ok = hmac.compare_digest(
            supplied_username.encode("utf-8"),
            self.username.encode("utf-8"),
        )
        password_ok = hmac.compare_digest(
            supplied_password.encode("utf-8"),
            self.password.encode("utf-8"),
        )
        return username_ok and password_ok

    def issue_session(
        self,
        *,
        now: datetime | None = None,
        lifetime_seconds: int = MAX_SESSION_SECONDS,
    ) -> str:
        if (
            type(lifetime_seconds) is not int
            or not MIN_SESSION_SECONDS <= lifetime_seconds <= MAX_SESSION_SECONDS
        ):
            raise ValueError(
                "session lifetime must be an exact integer between "
                f"{MIN_SESSION_SECONDS} and {MAX_SESSION_SECONDS} seconds"
            )
        expires_at = _utc_timestamp(now) + lifetime_seconds
        payload = self._session_payload(expires_at)
        signature = hmac.new(
            self.session_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return f"{SESSION_SCHEMA}|{expires_at}|{signature}"

    def verify_session(
        self,
        value: object,
        *,
        now: datetime | None = None,
    ) -> bool:
        if type(value) is not str or len(value) > 256:
            return False
        try:
            schema, expires_raw, supplied_signature = value.split("|", 2)
            if schema != SESSION_SCHEMA or not expires_raw.isascii():
                return False
            expires_at = int(expires_raw)
        except (TypeError, ValueError):
            return False
        current = _utc_timestamp(now)
        if expires_at <= current or expires_at > current + MAX_SESSION_SECONDS:
            return False
        expected = hmac.new(
            self.session_secret.encode("utf-8"),
            self._session_payload(expires_at),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(supplied_signature, expected)

    def _session_payload(self, expires_at: int) -> bytes:
        password_binding = hashlib.sha256(
            self.password.encode("utf-8")
        ).hexdigest()
        return (
            f"{SESSION_SCHEMA}\0{self.username}\0"
            f"{password_binding}\0{self.audience}\0{expires_at}"
        ).encode("utf-8")


@dataclass(frozen=True, slots=True, repr=False)
class ReadOnlyRuntimeSettings:
    """Non-sensitive projection of the full operational configuration."""

    schema_version: int
    mode: str
    source_sha256: str
    instance_id: str
    broker_environment: str | None
    max_snapshot_age_seconds: int
    positions_artifact_path_sha256: str
    positions_runtime_binding: str

    @classmethod
    def from_runtime_config(
        cls,
        config: RuntimeConfig,
    ) -> "ReadOnlyRuntimeSettings":
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be an exact RuntimeConfig instance")
        if (
            config.execution.broker_mutations_enabled
            or config.broker_mutations_authorized
            or config.starts_armed
        ):
            raise RuntimeCompositionError(
                "read-only runtime cannot accept broker mutation authority"
            )
        instance_id = hashlib.sha256(
            (
                "read-only-runtime.v1\0"
                f"{config.source_sha256}\0{config.paths.root}"
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            schema_version=config.schema_version,
            mode=config.mode,
            source_sha256=config.source_sha256,
            instance_id=instance_id,
            broker_environment=config.broker_environment,
            max_snapshot_age_seconds=(
                config.data.max_snapshot_age_seconds
            ),
            positions_artifact_path_sha256=hashlib.sha256(
                (
                    "positions-artifact-path.v1\0"
                    f"{config.paths.positions_artifact_file}"
                ).encode("utf-8")
            ).hexdigest(),
            positions_runtime_binding=(
                positions_artifact_runtime_binding(config)
            ),
        )

    def __repr__(self) -> str:
        return (
            "ReadOnlyRuntimeSettings("
            f"mode={self.mode!r}, "
            f"schema_version={self.schema_version}, "
            f"source_sha256={self.source_sha256!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ReadOnlyRuntimeContext:
    """Fully validated runtime inputs for processes that cannot mutate a broker."""

    settings: ReadOnlyRuntimeSettings
    dashboard_auth: DashboardAuth
    positions_artifact_reader: PositionsArtifactReader
    regime_shadow_reader: RegimeShadowReader

    def __post_init__(self) -> None:
        if type(self.settings) is not ReadOnlyRuntimeSettings:
            raise TypeError(
                "settings must be an exact ReadOnlyRuntimeSettings instance"
            )
        if type(self.dashboard_auth) is not DashboardAuth:
            raise TypeError("dashboard_auth must be an exact DashboardAuth instance")
        if type(self.positions_artifact_reader) is not PositionsArtifactReader:
            raise TypeError(
                "positions_artifact_reader must be an exact "
                "PositionsArtifactReader instance"
            )
        if type(self.regime_shadow_reader) is not RegimeShadowReader:
            raise TypeError(
                "regime_shadow_reader must be an exact RegimeShadowReader instance"
            )
        reader = self.positions_artifact_reader
        path_sha256 = hashlib.sha256(
            (
                "positions-artifact-path.v1\0"
                f"{reader.path}"
            ).encode("utf-8")
        ).hexdigest()
        if (
            reader.max_age_seconds
            != self.settings.max_snapshot_age_seconds
            or reader.expected_broker_environment
            != self.settings.broker_environment
            or reader.expected_runtime_binding
            != self.settings.positions_runtime_binding
            or path_sha256
            != self.settings.positions_artifact_path_sha256
            or reader.enabled
            != (self.settings.broker_environment is not None)
        ):
            raise RuntimeCompositionError(
                "positions artifact reader is not bound to runtime settings"
            )

    def __repr__(self) -> str:
        return (
            "ReadOnlyRuntimeContext("
            f"mode={self.settings.mode!r}, "
            f"schema_version={self.settings.schema_version}, "
            f"source_sha256={self.settings.source_sha256!r}, "
            "secrets=[REDACTED])"
        )

    @property
    def mode(self) -> str:
        return self.settings.mode

    @property
    def broker_environment(self) -> str | None:
        return self.settings.broker_environment

    @property
    def broker_reads_enabled(self) -> bool:
        return self.broker_environment is not None

    @property
    def broker_mutations_enabled(self) -> bool:
        return False

    @property
    def source_sha256(self) -> str:
        return self.settings.source_sha256

    @property
    def max_snapshot_age_seconds(self) -> int:
        return self.settings.max_snapshot_age_seconds

    @property
    def session_cookie_name(self) -> str:
        return f"etrade_read_only_{self.settings.instance_id[:16]}"


def load_read_only_runtime(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReadOnlyRuntimeContext:
    """Load all read-only startup inputs before any external collaborator."""

    config = load_runtime_config(config_path)
    validate_read_only_runtime_directories(config.paths)
    settings = ReadOnlyRuntimeSettings.from_runtime_config(config)
    runtime_environment = os.environ if environ is None else environ
    dashboard_secrets = resolve_dashboard_secrets(
        environ=runtime_environment,
    )
    artifact_key_text = resolve_positions_artifact_hmac_key(
        environ=runtime_environment,
    )
    validate_positions_artifact_key_separation(
        artifact_key_text,
        dashboard_secrets,
    )
    artifact_signing_key = PositionsArtifactSigningKey.from_text(
        artifact_key_text
    )
    auth = DashboardAuth.from_dashboard_secrets(
        dashboard_secrets,
        audience=settings.instance_id,
    )
    return ReadOnlyRuntimeContext(
        settings=settings,
        dashboard_auth=auth,
        positions_artifact_reader=PositionsArtifactReader(
            config.paths.positions_artifact_file,
            max_age_seconds=config.data.max_snapshot_age_seconds,
            signing_key=artifact_signing_key,
            expected_broker_environment=config.broker_environment,
            expected_runtime_binding=(
                settings.positions_runtime_binding
            ),
        ),
        regime_shadow_reader=RegimeShadowReader(
            config.paths.regime_shadow_file
        ),
    )


__all__ = [
    "DashboardAuth",
    "MAX_SESSION_SECONDS",
    "MIN_SESSION_SECONDS",
    "ReadOnlyRuntimeContext",
    "ReadOnlyRuntimeSettings",
    "RuntimeCompositionError",
    "SESSION_SCHEMA",
    "load_read_only_runtime",
]
