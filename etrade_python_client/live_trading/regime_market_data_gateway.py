"""Evidence-first, shadow-only acquisition of the V2 SPY/VIX input pair.

The gateway deliberately has no scheduler, configuration-file lookup, or
network activity at import time.  A caller injects its transport and clock,
which keeps retries deterministic in tests and makes the boundary between
provider I/O and durable evidence explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Protocol

import pandas as pd

from live_trading.regime_evidence_store import RegimeEvidenceStore
from live_trading.regime_market_data import (
    RegimeMarketDataSnapshot,
    SourceIdentity,
    regime_market_schedule,
)
from live_trading.regime_provider_evidence import (
    CBOE_VIX_DAILY_CSV_PARSER,
    MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
    ProviderParseConfig,
    ProviderParseJob,
    RawProviderResponse,
)


MASSIVE_DAILY_SUMMARY_ENDPOINT = "https://api.massive.com/v1/open-close/SPY"
CBOE_VIX_HISTORY_ENDPOINT = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
DEFAULT_BOOTSTRAP_MAX_SESSIONS = 800
_MASSIVE_SPY_IDENTITY = SourceIdentity(
    provider="massive",
    dataset="aggregates",
    provider_symbol="SPY",
    canonical_instrument="SPY",
    field="close",
    adjustment="unadjusted",
    unit="usd",
)
_CBOE_VIX_IDENTITY = SourceIdentity(
    provider="cboe",
    dataset="vix_daily",
    provider_symbol="VIX",
    canonical_instrument="VIX",
    field="close",
    adjustment="none",
    unit="index_points",
)
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "cookie", "set-cookie", "x-api-key", "api-key"}
)


class RegimeMarketDataGatewayError(RuntimeError):
    """A gateway error whose public message never contains credentials."""


class ProviderTransport(Protocol):
    """Minimal injectable transport; implementations return only safe receipts."""

    def fetch(
        self,
        *,
        provider: str,
        endpoint: str,
        requested_sessions: tuple[date, ...],
        request_parameters: tuple[tuple[str, str], ...],
        request_headers: tuple[tuple[str, str], ...],
        timeout_seconds: float,
        request_started_at: datetime,
        max_body_bytes: int,
    ) -> RawProviderResponse:
        ...


@dataclass(frozen=True)
class GatewayLegResult:
    """Safe, durable outcome of one provider/instrument source attempt."""

    provider: str
    instrument: str
    attempt_id: str
    succeeded: bool
    observation_count: int
    fetch_count: int
    error_code: str | None = None


@dataclass(frozen=True)
class GatewayRefreshResult:
    """Result of one paired refresh; failure never replaces the shadow head."""

    spy: GatewayLegResult
    vix: GatewayLegResult
    snapshot: RegimeMarketDataSnapshot | None
    published_snapshot_sha256: str | None
    publication_error_code: str | None = None

    @property
    def published(self) -> bool:
        return self.published_snapshot_sha256 is not None


class RequestsProviderTransport:
    """Small ``requests`` adapter that does not retain an API key in receipts."""

    def __init__(
        self, massive_api_key: str, *, now: Callable[[], datetime] | None = None
    ):
        key = str(massive_api_key).strip()
        if not key:
            raise ValueError("massive_api_key must be non-empty")
        self._massive_api_key = key
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _safe_response_headers(headers) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(name), str(value))
            for name, value in headers.items()
            if str(name).lower() not in _SENSITIVE_HEADER_NAMES
        )

    def fetch(
        self,
        *,
        provider: str,
        endpoint: str,
        requested_sessions: tuple[date, ...],
        request_parameters: tuple[tuple[str, str], ...],
        request_headers: tuple[tuple[str, str], ...],
        timeout_seconds: float,
        request_started_at: datetime,
        max_body_bytes: int,
    ) -> RawProviderResponse:
        """Fetch one response with a bounded body and safe retained metadata."""

        if provider not in {"massive", "cboe"}:
            raise ValueError("provider must be massive or cboe")
        if provider == "massive":
            if (
                len(requested_sessions) != 1
                or endpoint
                != f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/"
                f"{requested_sessions[0].isoformat()}"
            ):
                raise ValueError(
                    "Massive transport requires the fixed daily-summary endpoint"
                )
        elif endpoint != CBOE_VIX_HISTORY_ENDPOINT:
            raise ValueError(
                "Cboe transport requires the fixed VIX-history endpoint"
            )
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        # Importing requests is harmless, but delaying it avoids requiring it
        # for deterministic parser/store-only test runs.
        import requests

        outbound = {str(name): str(value) for name, value in request_headers}
        if provider == "massive":
            outbound["Authorization"] = f"Bearer {self._massive_api_key}"
        response = None
        try:
            response = requests.get(
                endpoint,
                headers=outbound,
                params=dict(request_parameters),
                timeout=timeout_seconds,
                stream=True,
                allow_redirects=False,
            )
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) > max_body_bytes:
                body = b""
                body_complete = False
            else:
                chunks: list[bytes] = []
                received = 0
                body_complete = True
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_body_bytes:
                        body_complete = False
                        break
                    chunks.append(bytes(chunk))
                body = b"".join(chunks)
            completed_at = self._now()
            return RawProviderResponse(
                provider=provider,
                endpoint=endpoint,
                requested_sessions=requested_sessions,
                request_parameters=request_parameters,
                request_started_at=request_started_at,
                completed_at=completed_at,
                status_code=int(response.status_code),
                headers=self._safe_response_headers(response.headers),
                body=body,
                body_complete=body_complete,
            )
        except (requests.RequestException, ValueError, OverflowError) as exc:
            # Never preserve an upstream exception: URLs and diagnostic text can
            # contain an outbound bearer header or query credential.
            raise RegimeMarketDataGatewayError("TRANSPORT_FAILURE") from None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass


class RegimeMarketDataGateway:
    """Acquire, retain, parse, and atomically publish one verified SPY/VIX pair."""

    def __init__(
        self,
        store: RegimeEvidenceStore,
        transport: ProviderTransport,
        *,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        max_sessions: int = DEFAULT_BOOTSTRAP_MAX_SESSIONS,
        max_body_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not isinstance(store, RegimeEvidenceStore):
            raise TypeError("store must be a RegimeEvidenceStore")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least one")
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self._store = store
        self._transport = transport
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or (lambda seconds: __import__("time").sleep(seconds))
        self._timeout_seconds = float(timeout_seconds)
        self._max_attempts = int(max_attempts)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        self._max_sessions = int(max_sessions)
        self._max_body_bytes = int(max_body_bytes)

    @staticmethod
    def _utc_now(clock: Callable[[], datetime]) -> datetime:
        value = pd.Timestamp(clock())
        if value.tzinfo is None:
            raise ValueError("gateway clock must return a timezone-aware timestamp")
        return value.tz_convert("UTC").to_pydatetime()

    @staticmethod
    def _sessions(start: date, end: date) -> tuple[date, ...]:
        schedule = regime_market_schedule(start, end)
        return tuple(pd.Timestamp(session).date() for session in schedule.index)

    @staticmethod
    def _retryable(response: RawProviderResponse) -> bool:
        return (
            not response.body_complete
            or int(response.status_code) in _RETRYABLE_STATUSES
        )

    @staticmethod
    def _request_endpoint(provider: str, session: date | None = None) -> str:
        if provider == "massive":
            if session is None:
                raise ValueError("Massive requests require one session")
            return f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/{session.isoformat()}"
        if provider == "cboe":
            return CBOE_VIX_HISTORY_ENDPOINT
        raise ValueError("provider must be massive or cboe")

    def _fetch_with_retries(
        self,
        *,
        attempt_id: str,
        provider: str,
        sessions: tuple[date, ...],
        endpoint: str,
        request_parameters: tuple[tuple[str, str], ...],
    ) -> tuple[str | None, int, str | None, pd.Timestamp | None]:
        """Capture every received response before deciding whether to retry."""

        fetch_count = 0
        last_completion: pd.Timestamp | None = None
        for ordinal in range(self._max_attempts):
            started_at = self._utc_now(self._now)
            try:
                response = self._transport.fetch(
                    provider=provider,
                    endpoint=endpoint,
                    requested_sessions=sessions,
                    request_parameters=request_parameters,
                    request_headers=(("Accept", "application/json" if provider == "massive" else "text/csv"),),
                    timeout_seconds=self._timeout_seconds,
                    request_started_at=started_at,
                    max_body_bytes=self._max_body_bytes,
                )
            except Exception:
                if ordinal + 1 < self._max_attempts:
                    try:
                        self._sleep(self._retry_backoff_seconds * (2**ordinal))
                    except Exception:
                        return (
                            None,
                            fetch_count,
                            "RETRY_BACKOFF_FAILURE",
                            last_completion,
                        )
                    continue
                return None, fetch_count, "TRANSPORT_FAILURE", last_completion

            if not isinstance(response, RawProviderResponse):
                return None, fetch_count, "INVALID_TRANSPORT_RESPONSE", last_completion
            fetch_count += 1
            observed_at = pd.Timestamp(self._utc_now(self._now))
            response_started_at = pd.Timestamp(
                response.request_started_at
            ).tz_convert("UTC")
            response_completed_at = pd.Timestamp(
                response.completed_at
            ).tz_convert("UTC")
            if (
                response_started_at != pd.Timestamp(started_at)
                or response_completed_at > observed_at
            ):
                return (
                    None,
                    fetch_count,
                    "EVIDENCE_CLOCK_FAILURE",
                    observed_at,
                )
            # A custom transport is equally subject to the production body
            # cap.  Do not persist a response exceeding the configured bound.
            if len(response.body) > self._max_body_bytes:
                return None, fetch_count, "RESPONSE_BODY_TOO_LARGE", last_completion
            # capture_response rejects a response whose clock precedes its
            # source attempt; that is an evidence failure, not a retryable HTTP
            # condition, so it safely ends this leg.
            try:
                fetch_sha256 = self._store.capture_response(attempt_id, response)
            except Exception:
                return None, fetch_count, "EVIDENCE_CAPTURE_FAILURE", last_completion
            completed_at = response_completed_at
            last_completion = (
                completed_at
                if last_completion is None
                else max(last_completion, completed_at)
            )
            if 200 <= int(response.status_code) < 300 and response.body_complete:
                return fetch_sha256, fetch_count, None, last_completion
            if not self._retryable(response) or ordinal + 1 >= self._max_attempts:
                return None, fetch_count, "UPSTREAM_RESPONSE_REJECTED", last_completion
            try:
                self._sleep(self._retry_backoff_seconds * (2**ordinal))
            except Exception:
                return (
                    None,
                    fetch_count,
                    "RETRY_BACKOFF_FAILURE",
                    last_completion,
                )
        return None, fetch_count, "TRANSPORT_FAILURE", last_completion

    def _complete_at(self, minimum: pd.Timestamp | None = None) -> datetime:
        current = pd.Timestamp(self._utc_now(self._now))
        if minimum is not None:
            current = max(current, minimum)
        return current.to_pydatetime()

    def _record_failure(
        self,
        attempt_id: str,
        code: str,
        minimum: pd.Timestamp | None,
    ) -> None:
        self._store.record_failure(
            attempt_id,
            self._complete_at(minimum),
            code,
            "provider refresh did not produce a complete verified source leg",
        )

    def _refresh_spy(
        self,
        sessions: tuple[date, ...],
        start: date,
        end: date,
    ) -> tuple[GatewayLegResult, tuple]:
        attempt_id = self._store.begin_attempt(
            "massive", "SPY", start, end, self._utc_now(self._now)
        )
        jobs: list[ProviderParseJob] = []
        fetch_count = 0
        last_completion: pd.Timestamp | None = None
        for session in sessions:
            fetch_sha256, count, code, completed = self._fetch_with_retries(
                attempt_id=attempt_id,
                provider="massive",
                sessions=(session,),
                endpoint=self._request_endpoint("massive", session),
                request_parameters=(("adjusted", "false"),),
            )
            fetch_count += count
            if completed is not None:
                last_completion = completed if last_completion is None else max(last_completion, completed)
            if fetch_sha256 is None:
                self._record_failure(attempt_id, code or "SPY_FETCH_FAILURE", last_completion)
                return GatewayLegResult("massive", "SPY", attempt_id, False, 0, fetch_count, code), ()
            config = ProviderParseConfig(
                parser_id=MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parser_id,
                requested_sessions=(session,),
                source_identity=_MASSIVE_SPY_IDENTITY,
                request_parameters=(("adjusted", "false"),),
            )
            jobs.append(ProviderParseJob(fetch_sha256=fetch_sha256, parser=MASSIVE_DAILY_TICKER_SUMMARY_PARSER, config=config))
        try:
            observations = self._store.record_success_from_captures(
                attempt_id, self._complete_at(last_completion), tuple(jobs)
            )
        except Exception:
            self._record_failure(attempt_id, "SPY_PARSE_OR_EVIDENCE_FAILURE", last_completion)
            return GatewayLegResult("massive", "SPY", attempt_id, False, 0, fetch_count, "SPY_PARSE_OR_EVIDENCE_FAILURE"), ()
        return GatewayLegResult("massive", "SPY", attempt_id, True, len(observations), fetch_count), observations

    def _refresh_vix(
        self,
        sessions: tuple[date, ...],
        start: date,
        end: date,
    ) -> tuple[GatewayLegResult, tuple]:
        attempt_id = self._store.begin_attempt(
            "cboe", "VIX", start, end, self._utc_now(self._now)
        )
        fetch_sha256, fetch_count, code, completed = self._fetch_with_retries(
            attempt_id=attempt_id,
            provider="cboe",
            sessions=sessions,
            endpoint=self._request_endpoint("cboe"),
            request_parameters=(),
        )
        if fetch_sha256 is None:
            self._record_failure(attempt_id, code or "VIX_FETCH_FAILURE", completed)
            return GatewayLegResult("cboe", "VIX", attempt_id, False, 0, fetch_count, code), ()
        config = ProviderParseConfig(
            parser_id=CBOE_VIX_DAILY_CSV_PARSER.parser_id,
            requested_sessions=sessions,
            source_identity=_CBOE_VIX_IDENTITY,
        )
        try:
            observations = self._store.record_success_from_captures(
                attempt_id,
                self._complete_at(completed),
                (ProviderParseJob(fetch_sha256=fetch_sha256, parser=CBOE_VIX_DAILY_CSV_PARSER, config=config),),
            )
        except Exception:
            self._record_failure(attempt_id, "VIX_PARSE_OR_EVIDENCE_FAILURE", completed)
            return GatewayLegResult("cboe", "VIX", attempt_id, False, 0, fetch_count, "VIX_PARSE_OR_EVIDENCE_FAILURE"), ()
        return GatewayLegResult("cboe", "VIX", attempt_id, True, len(observations), fetch_count), observations

    def refresh(
        self,
        start: date,
        end: date,
        *,
        channel: str = "shadow",
        snapshot_start: date | None = None,
    ) -> GatewayRefreshResult:
        """Refresh a bounded range; only a complete pair can move ``channel``.

        ``start`` and ``end`` bound provider I/O. For an incremental daily
        refresh, ``snapshot_start`` assembles the detector's rolling history
        from prior verified revisions without refetching every Massive session.
        """

        sessions = self._sessions(start, end)
        if not sessions:
            raise ValueError("requested range contains no NYSE sessions")
        if len(sessions) > self._max_sessions:
            raise ValueError("requested range exceeds gateway max_sessions")
        if snapshot_start is not None:
            if (
                isinstance(snapshot_start, datetime)
                or not isinstance(snapshot_start, date)
            ):
                raise TypeError("snapshot_start must be a calendar date")
            if snapshot_start > start:
                raise ValueError("snapshot_start cannot be after refresh start")
        spy, spy_observations = self._refresh_spy(sessions, start, end)
        vix, vix_observations = self._refresh_vix(sessions, start, end)
        if not spy.succeeded or not vix.succeeded:
            return GatewayRefreshResult(spy, vix, None, None)
        as_of = max(
            [pd.Timestamp(item.ingested_at) for item in (*spy_observations, *vix_observations)]
            + [pd.Timestamp(self._utc_now(self._now))]
        )
        snapshot: RegimeMarketDataSnapshot | None = None
        try:
            snapshot = self._store.assemble_verified_snapshot(
                snapshot_start or start,
                end,
                as_of=as_of,
            )
            published = self._store.publish_verified_snapshot(snapshot, channel=channel)
        except Exception:
            # Both provider attempts have already reached success; this is a
            # publication-only failure and cannot erase the prior channel head.
            return GatewayRefreshResult(
                spy,
                vix,
                snapshot,
                None,
                publication_error_code="VERIFIED_PUBLICATION_FAILURE",
            )
        return GatewayRefreshResult(spy, vix, snapshot, published)
