"""Atomic, sanitized persistence for the dashboard's V2 shadow signal.

The store accepts only a sealed :class:`RegimeSignal` envelope. It neither
obtains market data nor imports account, provider, EV, or order modules.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from live_trading.regime_market_data import regime_market_schedule
from live_trading.regime_signal import RegimeSignal, RegimeSignalError


MAX_ENVELOPE_BYTES = 256 * 1024
_PUBLIC_REASONS = frozenset({
    "shock_evidence_unavailable",
    "vix_daily_change_extreme",
    "spy_downside_tail_move",
    "shock_decay_window",
    "slow_vix_absolute_stress",
    "realized_vol_absolute_stress",
    "drawdown_absolute_stress",
    "background_score_high",
    "background_score_elevated",
    "persistent_score_confirmed",
    "no_stress_evidence",
})
_PUBLIC_ABSTAINS = frozenset({
    "r4_shadow_non_authoritative",
    "research_only",
    "execution_ineligible",
    "legacy_calibration_provenance_unverified",
    "prospective_holdout_incomplete",
    "calibration_artifact_unverified",
    "calibration_lineage_missing",
    "shadow_signal_missing",
    "shadow_signal_invalid",
    "shadow_signal_stale",
    "shadow_signal_not_yet_available",
})
_FAILURE_PUBLIC_REASON = {
    "shadow_signal_missing": "missing",
    "shadow_store_parent_missing": "missing",
    "shadow_signal_stale": "stale",
    "shadow_signal_future": "not_yet_available",
}


class RegimeShadowStoreError(RuntimeError):
    """Safe failure from the local V2 shadow-signal read model."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _utc_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("now must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _safe_open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _validate_trusted_directory(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RegimeShadowStoreError("shadow_store_parent_unsafe")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RegimeShadowStoreError("shadow_store_parent_unsafe")


def _validate_signal_file(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RegimeShadowStoreError("shadow_signal_path_unsafe")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise RegimeShadowStoreError("shadow_signal_path_unsafe")
    if metadata.st_nlink != 1:
        raise RegimeShadowStoreError("shadow_signal_path_unsafe")
    if metadata.st_size > MAX_ENVELOPE_BYTES:
        raise RegimeShadowStoreError("shadow_signal_schema_invalid")


def _public_codes(values: tuple[str, ...], allowed: frozenset[str]) -> list[str]:
    return [value for value in values if value in allowed]


def unavailable_dashboard_payload(internal_code: str) -> dict[str, Any]:
    """Return a redacted, closed-enum public status for every failure."""

    reason = _FAILURE_PUBLIC_REASON.get(internal_code, "invalid")
    return {
        "available": False,
        "status": "unavailable",
        "source_family": "v2_shadow",
        "as_of_session": None,
        "effective_session": None,
        "background_state": "unavailable",
        "shock_state": "unavailable",
        "composite_label": "unavailable+unavailable",
        "availability": "unavailable",
        "reason_codes": [reason],
        "abstain_reasons": [reason, "r4_shadow_non_authoritative"],
        "stale": reason == "stale",
        "may_authorize_execution": False,
    }


class RegimeShadowStore:
    """Write and read one canonical V2 signal envelope at a configured path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name or self.path.name in {".", ".."}:
            raise ValueError("path must identify a signal file")

    def _open_trusted_parent(self) -> int:
        try:
            before = os.lstat(self.path.parent)
        except FileNotFoundError as exc:
            raise RegimeShadowStoreError("shadow_store_parent_missing") from exc
        except OSError as exc:
            raise RegimeShadowStoreError("shadow_store_parent_unsafe") from exc
        if stat.S_ISLNK(before.st_mode):
            raise RegimeShadowStoreError("shadow_store_parent_unsafe")
        try:
            descriptor = os.open(
                self.path.parent,
                _safe_open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
            )
        except OSError as exc:
            raise RegimeShadowStoreError("shadow_store_parent_unsafe") from exc
        try:
            after = os.fstat(descriptor)
            if not _same_file(before, after):
                raise RegimeShadowStoreError("shadow_store_parent_unsafe")
            _validate_trusted_directory(after)
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def publish(self, signal: RegimeSignal) -> None:
        """Atomically publish a validated sealed signal with mode ``0600``."""

        if type(signal) is not RegimeSignal:
            raise TypeError("signal must be an exact RegimeSignal instance")
        if signal.may_authorize_execution:
            raise RegimeShadowStoreError("execution_capable_signal_rejected")
        try:
            payload = signal.to_json()
            parsed = RegimeSignal.from_json(payload)
        except (TypeError, RegimeSignalError) as exc:
            raise RegimeShadowStoreError("invalid_signal_rejected") from exc
        if parsed != signal or parsed.may_authorize_execution:
            raise RegimeShadowStoreError("invalid_signal_rejected")

        parent_descriptor = self._open_trusted_parent()
        temporary_name: str | None = None
        temporary_descriptor = -1
        try:
            try:
                existing = os.stat(
                    self.path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(existing.st_mode):
                    raise RegimeShadowStoreError("shadow_store_path_unsafe")
            except FileNotFoundError:
                pass
            for _ in range(16):
                candidate = f".{self.path.name}.{secrets.token_hex(16)}.tmp"
                try:
                    temporary_descriptor = os.open(
                        candidate,
                        _safe_open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                    temporary_name = candidate
                    break
                except FileExistsError:
                    continue
            if temporary_descriptor < 0 or temporary_name is None:
                raise RegimeShadowStoreError("shadow_store_publish_failed")
            os.fchmod(temporary_descriptor, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(temporary_descriptor, "w", encoding="utf-8") as handle:
                temporary_descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                self.path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary_name = None
            os.fsync(parent_descriptor)
        except RegimeShadowStoreError:
            raise
        except OSError as exc:
            raise RegimeShadowStoreError("shadow_store_publish_failed") from exc
        finally:
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)

    def _read_payload_from_verified_fd(self, parent_descriptor: int) -> str:
        try:
            before = os.stat(
                self.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise RegimeShadowStoreError("shadow_signal_missing") from exc
        except OSError as exc:
            raise RegimeShadowStoreError("shadow_signal_unreadable") from exc
        _validate_signal_file(before)
        try:
            descriptor = os.open(
                self.path.name,
                _safe_open_flags(
                    os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                ),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise RegimeShadowStoreError("shadow_signal_missing") from exc
            raise RegimeShadowStoreError("shadow_signal_path_unsafe") from exc
        try:
            after = os.fstat(descriptor)
            if not _same_file(before, after):
                raise RegimeShadowStoreError("shadow_signal_path_unsafe")
            _validate_signal_file(after)
            chunks: list[bytes] = []
            remaining = MAX_ENVELOPE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            final = os.fstat(descriptor)
            try:
                path_after = os.stat(
                    self.path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise RegimeShadowStoreError(
                    "shadow_signal_path_unsafe"
                ) from exc
            _validate_signal_file(final)
            _validate_signal_file(path_after)
            if (
                not _same_snapshot(before, after)
                or not _same_snapshot(after, final)
                or not _same_snapshot(final, path_after)
                or len(raw) != final.st_size
            ):
                raise RegimeShadowStoreError("shadow_signal_path_unsafe")
            if len(raw) > MAX_ENVELOPE_BYTES:
                raise RegimeShadowStoreError("shadow_signal_schema_invalid")
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RegimeShadowStoreError("shadow_signal_schema_invalid") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_session_causal(signal: RegimeSignal, now: datetime) -> None:
        if signal.available_at > now:
            raise RegimeShadowStoreError("shadow_signal_future")
        try:
            schedule = regime_market_schedule(
                signal.effective_session,
                signal.effective_session,
            )
            finalization = pd.Timestamp(
                schedule.iloc[0]["joint_finalization_at"]
            ).to_pydatetime()
        except Exception as exc:
            raise RegimeShadowStoreError("shadow_signal_schema_invalid") from exc
        if now >= finalization:
            raise RegimeShadowStoreError("shadow_signal_stale")

    def load(self, *, now: datetime | None = None) -> RegimeSignal:
        """Load a descriptor-verified signal or raise a safe internal code."""

        current_time = _utc_now(now)
        parent_descriptor = self._open_trusted_parent()
        try:
            payload = self._read_payload_from_verified_fd(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        try:
            signal = RegimeSignal.from_json(payload)
        except (TypeError, RegimeSignalError) as exc:
            raise RegimeShadowStoreError("shadow_signal_schema_invalid") from exc
        if signal.may_authorize_execution:
            raise RegimeShadowStoreError("execution_capable_signal_rejected")
        self._validate_session_causal(signal, current_time)
        return signal

    def dashboard_payload(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return only whitelisted V2 display fields or a redacted failure."""

        try:
            signal = self.load(now=now)
        except RegimeShadowStoreError as exc:
            return unavailable_dashboard_payload(exc.code)
        abstains = _public_codes(signal.abstain_reasons, _PUBLIC_ABSTAINS)
        if "r4_shadow_non_authoritative" not in abstains:
            abstains.append("r4_shadow_non_authoritative")
        return {
            "available": signal.availability.value == "advisory",
            "status": "advisory",
            "source_family": signal.source_family.value,
            "as_of_session": signal.as_of_session.isoformat(),
            "effective_session": signal.effective_session.isoformat(),
            "background_state": signal.background_state.value,
            "shock_state": signal.shock_state.value,
            "composite_label": signal.composite_label,
            "availability": signal.availability.value,
            "reason_codes": _public_codes(signal.reason_codes, _PUBLIC_REASONS),
            "abstain_reasons": abstains,
            "stale": False,
            "may_authorize_execution": False,
        }


class RegimeShadowReader:
    """Read-only capability projection for dashboard composition."""

    __slots__ = ("path",)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name or self.path.name in {".", ".."}:
            raise ValueError("path must identify a signal file")

    def dashboard_payload(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return RegimeShadowStore(self.path).dashboard_payload(now=now)


__all__ = [
    "MAX_ENVELOPE_BYTES",
    "RegimeShadowReader",
    "RegimeShadowStore",
    "RegimeShadowStoreError",
    "unavailable_dashboard_payload",
]
