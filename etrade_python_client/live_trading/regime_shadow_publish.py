"""Bounded, entitlement-gated publication of one advisory V2 shadow signal.

This module deliberately has no import-time configuration, provider I/O, or
order-facing dependency.  A scheduler may invoke :func:`main`, but it must
provide an external entitlement validator before this module reads a provider
credential or creates a network transport.

The default detector needs retained history before it can produce an available
background state.  A first-time historical bootstrap is an explicit,
operator-run collection task; this publisher refuses to replace the current
shadow file with an unavailable result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from live_trading.regime_detector_v2 import (
    BACKGROUND_UNAVAILABLE,
    SHOCK_UNAVAILABLE,
    RegimeDetectorConfig,
    detect_regimes_from_verified_snapshot,
)
from live_trading.regime_evidence_store import RegimeEvidenceStore
from live_trading.regime_market_data_gateway import (
    RegimeMarketDataGateway,
    RequestsProviderTransport,
)
from live_trading.regime_shadow_store import RegimeShadowStore
from live_trading.regime_signal import (
    RegimeSignal,
    RegimeSignalError,
    from_calibrated_v2_trace,
    next_tradable_session,
)


class RegimeShadowPublishError(RuntimeError):
    """A safe, stable reason that a new shadow signal was not published."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _calendar_date(value: date, field_name: str) -> date:
    if isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a calendar date")
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None or value != value.normalize():
            raise TypeError(f"{field_name} must be a calendar date")
        value = value.date()
    if not isinstance(value, date):
        raise TypeError(f"{field_name} must be a calendar date")
    return value


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RegimeShadowPublishError(field_name)
    return value


def _latest_trace_row(trace: pd.DataFrame, session: date) -> pd.DataFrame:
    if not isinstance(trace, pd.DataFrame) or trace.empty:
        raise RegimeShadowPublishError("detector_trace_invalid")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(trace.index, errors="raise"))
    except Exception as exc:
        raise RegimeShadowPublishError("detector_trace_invalid") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RegimeShadowPublishError("detector_trace_invalid")
    if index[-1].date() != session:
        raise RegimeShadowPublishError("detector_tail_session_mismatch")
    tail = trace.tail(1).copy()
    tail.index = index[-1:]
    return tail


def _require_verified_decision_time_row(
    tail: pd.DataFrame, published_sha256: str
) -> None:
    row = tail.iloc[0]
    required = {
        "Input_Provenance_Status": "verified",
        "Evidence_Verification_Kind": "decision_time",
    }
    for column, expected in required.items():
        if column not in row.index or row[column] != expected:
            raise RegimeShadowPublishError("verified_decision_time_required")
    if (
        "Input_Snapshot_SHA256" not in row.index
        or row["Input_Snapshot_SHA256"] != published_sha256
    ):
        raise RegimeShadowPublishError("published_snapshot_mismatch")
    if (
        "Evidence_Decision_Time_Eligible" not in row.index
        or not isinstance(
            row["Evidence_Decision_Time_Eligible"], (bool, np.bool_)
        )
        or not row["Evidence_Decision_Time_Eligible"]
    ):
        raise RegimeShadowPublishError("verified_decision_time_required")
    if (
        "Background_State" not in row.index
        or "Shock_State" not in row.index
        or row["Background_State"] == BACKGROUND_UNAVAILABLE
        or row["Shock_State"] == SHOCK_UNAVAILABLE
    ):
        raise RegimeShadowPublishError("insufficient_history")


def _require_entitlement_capability(
    entitlement_authorized: Callable[[], bool] | None,
) -> None:
    """Fail before provider I/O unless a caller supplies an approval capability."""

    if not callable(entitlement_authorized):
        raise RegimeShadowPublishError("entitlement_invalid")
    try:
        approved = entitlement_authorized()
    except Exception as exc:
        raise RegimeShadowPublishError("entitlement_invalid") from exc
    if approved is not True:
        raise RegimeShadowPublishError("entitlement_invalid")


def publish_shadow_signal(
    *,
    gateway: Any,
    evidence_store: Any,
    shadow_store: RegimeShadowStore,
    refresh_start: date,
    session: date,
    snapshot_start: date,
    entitlement_authorized: Callable[[], bool] | None,
    config: RegimeDetectorConfig | None = None,
) -> RegimeSignal:
    """Refresh and seal exactly one verified, timely advisory signal.

    Every validation occurs before ``shadow_store.publish``.  Thus a failed
    refresh, a mismatched publication head, invalid evidence, or an unusable
    detector result leaves the prior sealed signal untouched.
    """

    refresh_start = _calendar_date(refresh_start, "refresh_start")
    session = _calendar_date(session, "session")
    snapshot_start = _calendar_date(snapshot_start, "snapshot_start")
    if refresh_start > session or snapshot_start > refresh_start:
        raise ValueError("date bounds must satisfy snapshot_start <= refresh_start <= session")
    if not isinstance(shadow_store, RegimeShadowStore):
        raise TypeError("shadow_store must be a RegimeShadowStore")
    if config is not None and not isinstance(config, RegimeDetectorConfig):
        raise TypeError("config must be a RegimeDetectorConfig or None")

    _require_entitlement_capability(entitlement_authorized)

    try:
        refresh = gateway.refresh(
            refresh_start,
            session,
            channel="shadow",
            snapshot_start=snapshot_start,
        )
    except Exception as exc:
        raise RegimeShadowPublishError("refresh_failed") from exc
    snapshot = getattr(refresh, "snapshot", None)
    published_sha256 = getattr(refresh, "published_snapshot_sha256", None)
    if snapshot is None or not bool(getattr(refresh, "published", False)):
        raise RegimeShadowPublishError("verified_refresh_not_published")
    published_sha256 = _required_sha256(
        published_sha256, "verified_refresh_not_published"
    )
    if getattr(snapshot, "snapshot_sha256", None) != published_sha256:
        raise RegimeShadowPublishError("published_snapshot_mismatch")

    try:
        latest = evidence_store.latest_snapshot(
            "shadow", require_verified=True
        )
    except Exception as exc:
        raise RegimeShadowPublishError("latest_verified_snapshot_unavailable") from exc
    if latest is None:
        raise RegimeShadowPublishError("latest_verified_snapshot_unavailable")
    if getattr(latest, "snapshot_sha256", None) != published_sha256:
        raise RegimeShadowPublishError("published_snapshot_mismatch")

    try:
        trace = detect_regimes_from_verified_snapshot(latest, evidence_store, config)
        tail = _latest_trace_row(trace, session)
        _require_verified_decision_time_row(tail, published_sha256)
        annotations = from_calibrated_v2_trace(tail)
        effective_session = next_tradable_session(session).isoformat()
        signal = annotations.get(effective_session)
    except RegimeShadowPublishError:
        raise
    except Exception as exc:
        raise RegimeShadowPublishError("detector_signal_invalid") from exc
    if len(annotations) != 1 or signal is None:
        raise RegimeShadowPublishError("detector_signal_invalid")
    if signal.may_authorize_execution:
        raise RegimeShadowPublishError("execution_capable_signal_rejected")

    try:
        shadow_store.publish(signal)
    except Exception as exc:
        raise RegimeShadowPublishError("shadow_signal_publish_failed") from exc
    return signal


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO YYYY-MM-DD date") from exc


def _load_entitlement_validator(spec: str) -> Callable[[str], bool]:
    """Load an operator-provided validator without bundling entitlement data."""

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise RegimeShadowPublishError("entitlement_validator_unavailable")
    try:
        validator = getattr(importlib.import_module(module_name), attribute)
    except Exception as exc:
        raise RegimeShadowPublishError("entitlement_validator_unavailable") from exc
    if not callable(validator):
        raise RegimeShadowPublishError("entitlement_validator_unavailable")
    return validator


def _require_entitlement(entitlement_id: str, validator_spec: str) -> None:
    if not isinstance(entitlement_id, str) or not entitlement_id.strip():
        raise RegimeShadowPublishError("entitlement_invalid")
    validator = _load_entitlement_validator(validator_spec)
    try:
        approved = validator(entitlement_id.strip())
    except Exception as exc:
        raise RegimeShadowPublishError("entitlement_invalid") from exc
    if approved is not True:
        raise RegimeShadowPublishError("entitlement_invalid")


def _validated_entitlement_capability(
    entitlement_id: str, validator_spec: str
) -> Callable[[], bool]:
    """Resolve external approval once, before credentials or transport setup."""

    _require_entitlement(entitlement_id, validator_spec)
    return lambda: True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one V2 advisory shadow signal from verified provider "
            "evidence. A fresh evidence database needs a separately run, "
            "entitlement-gated historical bootstrap before this command can "
            "produce an available regime."
        )
    )
    parser.add_argument("--evidence-db", required=True, type=Path)
    parser.add_argument("--shadow-signal-file", required=True, type=Path)
    parser.add_argument("--refresh-start", required=True, type=_parse_date)
    parser.add_argument("--session", required=True, type=_parse_date)
    parser.add_argument("--snapshot-start", required=True, type=_parse_date)
    parser.add_argument("--entitlement-id", required=True)
    parser.add_argument(
        "--entitlement-validator",
        required=True,
        help="external module:function; must return True for the opaque entitlement ID",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; entitlement is checked before credentials or network."""

    arguments = _parser().parse_args(argv)
    try:
        entitlement_authorized = _validated_entitlement_capability(
            arguments.entitlement_id, arguments.entitlement_validator
        )
        api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
        if not api_key:
            raise RegimeShadowPublishError("provider_credential_missing")
        with RegimeEvidenceStore(arguments.evidence_db) as evidence_store:
            gateway = RegimeMarketDataGateway(
                evidence_store, RequestsProviderTransport(api_key)
            )
            signal = publish_shadow_signal(
                gateway=gateway,
                evidence_store=evidence_store,
                shadow_store=RegimeShadowStore(arguments.shadow_signal_file),
                refresh_start=arguments.refresh_start,
                session=arguments.session,
                snapshot_start=arguments.snapshot_start,
                entitlement_authorized=entitlement_authorized,
            )
    except RegimeShadowPublishError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError):
        print("shadow_publisher_setup_failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "as_of_session": signal.as_of_session.isoformat(),
                "effective_session": signal.effective_session.isoformat(),
                "signal_sha256": signal.signal_sha256,
                "snapshot_sha256": signal.lineage.snapshot_sha256,
                "evidence_manifest_sha256": signal.lineage.evidence_manifest_sha256,
                "may_authorize_execution": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
