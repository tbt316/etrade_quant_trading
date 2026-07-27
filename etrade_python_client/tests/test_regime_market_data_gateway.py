"""Deterministic provider-evidence and gateway tests.

The fixtures mirror the documented Massive Daily Ticker Summary JSON and
Cboe VIX_History.csv schemas.  No test in this module may use live network
access or wall-clock sleeps.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from live_trading.regime_detector_v2 import (
    detect_regimes_from_snapshot,
    detect_regimes_from_verified_snapshot,
)
from live_trading.regime_evidence_store import RegimeEvidenceStore
from live_trading.regime_market_data import regime_market_schedule
from live_trading.regime_market_data_gateway import (
    CBOE_VIX_HISTORY_ENDPOINT,
    MASSIVE_DAILY_SUMMARY_ENDPOINT,
    RegimeMarketDataGateway,
    RegimeMarketDataGatewayError,
    RequestsProviderTransport,
    _CBOE_VIX_IDENTITY,
    _MASSIVE_SPY_IDENTITY,
)
from live_trading.regime_provider_evidence import (
    CBOE_VIX_DAILY_CSV_PARSER,
    MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
    ProviderEvidenceError,
    ProviderParseConfig,
    RawProviderResponse,
)


REGULAR_SESSION = date(2025, 3, 10)
HISTORY_START = date(2025, 3, 6)
NEXT_SESSION = date(2025, 3, 11)
EARLY_CLOSE_SESSION = date(2025, 11, 28)
API_KEY = "must-never-appear-in-evidence"
AUTHORIZATION = f"Bearer {API_KEY}"
GATEWAY_NOW = datetime(2025, 3, 10, 21, 0, tzinfo=timezone.utc)


def _massive_body(
    session: date,
    *,
    close: float = 560.25,
    after_hours: float = 999.0,
    symbol: str = "SPY",
    response_session: date | None = None,
    status: str = "OK",
) -> bytes:
    return json.dumps(
        {
            "afterHours": after_hours,
            "close": close,
            "from": (response_session or session).isoformat(),
            "high": close + 1.0,
            "low": close - 1.0,
            "open": close - 0.5,
            "preMarket": close - 0.75,
            "status": status,
            "symbol": symbol,
            "volume": 42_000_000,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _vix_body(*rows: tuple[date, float]) -> bytes:
    lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    for session, close in rows:
        lines.append(
            f"{session:%m/%d/%Y},"
            f"{close - 0.25:.6f},"
            f"{close + 0.75:.6f},"
            f"{close - 0.75:.6f},"
            f"{close:.6f}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _event_at(session: date, instrument: str) -> pd.Timestamp:
    schedule = regime_market_schedule(session, session)
    column = "spy_event_at" if instrument == "SPY" else "vix_event_at"
    return pd.Timestamp(schedule.loc[pd.Timestamp(session), column])


def _raw_response(
    provider: str,
    sessions: tuple[date, ...],
    body: bytes,
    *,
    completed_at: pd.Timestamp | None = None,
    request_started_at: datetime | None = None,
    status_code: int = 200,
    request_parameters: tuple[tuple[str, str], ...] = (),
):
    if completed_at is None:
        completed_at = max(
            _event_at(session, "SPY" if provider == "massive" else "VIX")
            for session in sessions
        ) + pd.Timedelta(minutes=5)
        if request_started_at is not None:
            completed_at = max(
                completed_at,
                pd.Timestamp(request_started_at).tz_convert("UTC"),
            )
    endpoint = (
        f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/{sessions[0].isoformat()}"
        if provider == "massive"
        else CBOE_VIX_HISTORY_ENDPOINT
    )
    media_type = "application/json" if provider == "massive" else "text/csv"
    return RawProviderResponse(
        provider=provider,
        endpoint=endpoint,
        requested_sessions=sessions,
        request_started_at=request_started_at or (
            completed_at - pd.Timedelta(seconds=1)
        ).to_pydatetime(),
        completed_at=completed_at.to_pydatetime(),
        status_code=status_code,
        headers=(("Content-Type", media_type),),
        body=body,
        request_parameters=request_parameters,
    )


def _massive_config(session: date) -> ProviderParseConfig:
    return ProviderParseConfig(
        parser_id=MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parser_id,
        requested_sessions=(session,),
        source_identity=_MASSIVE_SPY_IDENTITY,
        request_parameters=(("adjusted", "false"),),
    )


def _vix_config(*sessions: date) -> ProviderParseConfig:
    return ProviderParseConfig(
        parser_id=CBOE_VIX_DAILY_CSV_PARSER.parser_id,
        requested_sessions=sessions,
        source_identity=_CBOE_VIX_IDENTITY,
    )


class ScriptedTransport:
    """A no-network transport whose scripts are deterministic callables."""

    def __init__(self, scripted):
        self._scripted = deque(scripted)
        self.calls = []

    def fetch(self, **kwargs):
        self.calls.append(kwargs)
        action = self._scripted.popleft()
        if isinstance(action, BaseException):
            raise action
        return action(kwargs) if callable(action) else action

    def assert_drained(self, testcase: unittest.TestCase) -> None:
        testcase.assertFalse(self._scripted, "unconsumed transport fixtures")


class _FakeRequestsResponse:
    def __init__(self, body: bytes, headers=None, status_code=200):
        self.headers = headers or {"Content-Type": "application/json"}
        self.status_code = status_code
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size):
        del chunk_size
        yield self._body

    def close(self):
        self.closed = True


class ProviderParserTests(unittest.TestCase):
    def test_massive_uses_exact_path_unadjusted_regular_close_and_clock(self):
        response = _raw_response(
            "massive",
            (REGULAR_SESSION,),
            _massive_body(REGULAR_SESSION, close=560.25, after_hours=999.0),
            request_parameters=(("adjusted", "false"),),
        )

        parsed = MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parse(
            response,
            _massive_config(REGULAR_SESSION),
            ingested_at=response.completed_at,
        )

        observation = parsed.outputs[0].observation
        self.assertEqual(observation.close, 560.25)
        self.assertNotEqual(observation.close, 999.0)
        self.assertEqual(
            observation.event_at, pd.Timestamp("2025-03-10T20:00:00Z")
        )
        self.assertEqual(
            response.endpoint,
            "https://api.massive.com/v1/open-close/SPY/2025-03-10",
        )
        self.assertEqual(response.request_parameters, (("adjusted", "false"),))

    def test_cboe_exact_rows_and_regular_and_early_close_clocks(self):
        sessions = (REGULAR_SESSION, EARLY_CLOSE_SESSION)
        completed = _event_at(EARLY_CLOSE_SESSION, "VIX") + pd.Timedelta(minutes=5)
        response = _raw_response(
            "cboe",
            sessions,
            _vix_body(
                (date(2025, 3, 7), 19.0),
                (REGULAR_SESSION, 20.5),
                (EARLY_CLOSE_SESSION, 15.25),
            ),
            completed_at=completed,
        )

        parsed = CBOE_VIX_DAILY_CSV_PARSER.parse(
            response, _vix_config(*sessions), ingested_at=response.completed_at
        )

        self.assertEqual([item.observation.close for item in parsed.outputs], [20.5, 15.25])
        self.assertEqual(
            parsed.outputs[0].observation.event_at,
            pd.Timestamp("2025-03-10T20:15:00Z"),
        )
        self.assertEqual(
            parsed.outputs[1].observation.event_at,
            pd.Timestamp("2025-11-28T18:15:00Z"),
        )

    def test_parser_rejects_missing_duplicate_wrong_nonfinite_partial_and_preclose_evidence(self):
        massive = _raw_response(
            "massive",
            (REGULAR_SESSION,),
            _massive_body(REGULAR_SESSION),
            request_parameters=(("adjusted", "false"),),
        )
        bad_massive_bodies = (
            _massive_body(REGULAR_SESSION, symbol="QQQ"),
            _massive_body(REGULAR_SESSION, response_session=date(2025, 3, 7)),
            _massive_body(REGULAR_SESSION, close=float("nan")),
            _massive_body(REGULAR_SESSION, close=True),
            (
                b'{"close":560.0,"close":561.0,"from":"2025-03-10",'
                b'"status":"OK","symbol":"SPY"}'
            ),
            json.dumps({"status": "OK", "symbol": "SPY", "from": "2025-03-10"}).encode(),
        )
        for body in bad_massive_bodies:
            with self.subTest(massive_body=body):
                with self.assertRaises(ProviderEvidenceError):
                    MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parse(
                        dataclasses.replace(massive, body=body),
                        _massive_config(REGULAR_SESSION),
                        ingested_at=massive.completed_at,
                    )

        duplicate_vix = _raw_response(
            "cboe",
            (REGULAR_SESSION,),
            _vix_body((REGULAR_SESSION, 20.0), (REGULAR_SESSION, 21.0)),
        )
        missing_vix = _raw_response("cboe", (REGULAR_SESSION,), _vix_body())
        extra_field_vix = _raw_response(
            "cboe",
            (REGULAR_SESSION,),
            _vix_body((REGULAR_SESSION, 20.0)).replace(
                b"20.000000\n",
                b"20.000000,unexpected\n",
            ),
        )
        for response in (duplicate_vix, missing_vix, extra_field_vix):
            with self.subTest(vix_body=response.body):
                with self.assertRaises(ProviderEvidenceError):
                    CBOE_VIX_DAILY_CSV_PARSER.parse(
                        response,
                        _vix_config(REGULAR_SESSION),
                        ingested_at=response.completed_at,
                    )

        pre_close = dataclasses.replace(
            massive,
            request_started_at=_event_at(REGULAR_SESSION, "SPY") - pd.Timedelta(seconds=2),
            completed_at=_event_at(REGULAR_SESSION, "SPY") - pd.Timedelta(seconds=1),
        )
        with self.assertRaises(ProviderEvidenceError):
            MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parse(
                pre_close,
                _massive_config(REGULAR_SESSION),
                ingested_at=_event_at(REGULAR_SESSION, "SPY"),
            )
        with self.assertRaises(ProviderEvidenceError):
            RawProviderResponse(
                provider="massive",
                endpoint=massive.endpoint,
                requested_sessions=(REGULAR_SESSION,),
                request_started_at=massive.request_started_at,
                completed_at=massive.completed_at,
                status_code=200,
                headers=(("Content-Type", "application/json"),),
                body=massive.body,
                request_parameters=(("adjusted", "false"),),
                body_complete=False,
            )


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = RegimeEvidenceStore(Path(self.temp_dir.name) / "evidence.sqlite3")
        self.sleeps = []

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _gateway(self, transport, **overrides):
        now = overrides.pop("now", lambda: GATEWAY_NOW)
        return RegimeMarketDataGateway(
            self.store,
            transport,
            now=now,
            sleep=self.sleeps.append,
            max_sessions=10,
            **overrides,
        )

    @staticmethod
    def _massive_ok(kwargs, *, close=560.25, status_code=200, body=None):
        session = kwargs["requested_sessions"][0]
        return _raw_response(
            "massive",
            (session,),
            _massive_body(session, close=close) if body is None else body,
            request_started_at=kwargs["request_started_at"],
            request_parameters=kwargs["request_parameters"],
            status_code=status_code,
        )

    @staticmethod
    def _vix_ok(kwargs, *, close=20.5, body=None):
        sessions = kwargs["requested_sessions"]
        return _raw_response(
            "cboe",
            sessions,
            _vix_body(*((session, close) for session in sessions)) if body is None else body,
            request_started_at=kwargs["request_started_at"],
        )

    def _success_transport(self, *, spy_close=560.25, vix_close=20.5):
        return ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, close=spy_close),
                lambda kwargs: self._vix_ok(kwargs, close=vix_close),
            ]
        )

    def test_gateway_publishes_verified_pair_and_reparses_retained_bytes(self):
        transport = self._success_transport()
        result = self._gateway(transport).refresh(REGULAR_SESSION, REGULAR_SESSION)
        transport.assert_drained(self)

        self.assertTrue(result.published)
        self.assertTrue(result.spy.succeeded)
        self.assertTrue(result.vix.succeeded)
        self.assertEqual(result.spy.observation_count, 1)
        self.assertEqual(result.vix.observation_count, 1)
        self.assertEqual(result.published_snapshot_sha256, result.snapshot.snapshot_sha256)
        report = self.store.verify_snapshot(result.snapshot)
        self.assertTrue(report.verified)
        self.assertEqual(
            self.store.latest_snapshot("shadow", require_verified=True).snapshot_sha256,
            result.published_snapshot_sha256,
        )
        self.assertEqual(self.store._connection.execute("SELECT COUNT(*) FROM raw_payload_blobs").fetchone()[0], 2)
        self.assertEqual(self.store._connection.execute("SELECT COUNT(*) FROM fetch_receipts").fetchone()[0], 2)
        self.assertEqual(self.store._connection.execute("SELECT COUNT(*) FROM parser_receipts").fetchone()[0], 2)
        self.assertEqual(transport.calls[0]["endpoint"], f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/2025-03-10")
        self.assertEqual(transport.calls[0]["request_parameters"], (("adjusted", "false"),))
        self.assertNotIn("Authorization", dict(transport.calls[0]["request_headers"]))

        unverified = detect_regimes_from_snapshot(result.snapshot)
        with mock.patch.object(
            self.store,
            "verify_snapshot",
            wraps=self.store.verify_snapshot,
        ) as verify_snapshot:
            verified = detect_regimes_from_verified_snapshot(
                result.snapshot,
                self.store,
            )
        verify_snapshot.assert_called_once_with(result.snapshot)
        self.assertFalse(unverified.attrs["source_provenance_verified"])
        self.assertTrue(verified.attrs["source_provenance_verified"])
        self.assertTrue(
            verified["Input_Provenance_Status"].eq("verified").all()
        )
        self.assertTrue(
            verified["Input_Provenance_Evidence"]
            .eq("durable_raw_payload_and_parser_receipts_verified")
            .all()
        )
        self.assertTrue(
            verified["Evidence_Manifest_SHA256"]
            .eq(verified.attrs["evidence_manifest_sha256"])
            .all()
        )
        self.assertTrue(
            verified["Evidence_Verification_Kind"]
            .eq("decision_time")
            .all()
        )
        self.assertTrue(verified["Evidence_Decision_Time_Eligible"].all())
        self.assertTrue(verified.attrs["evidence_decision_time_eligible"])
        self.assertTrue(verified.attrs["freshness_assessed"])
        self.assertFalse(verified["Execution_Eligible"].any())
        self.assertFalse(verified.attrs["execution_eligible"])

        replay_report = dataclasses.replace(
            report,
            verification_kind="verified_replay",
        )
        with mock.patch.object(
            self.store,
            "verify_snapshot",
            return_value=replay_report,
        ):
            replay = detect_regimes_from_verified_snapshot(
                result.snapshot,
                self.store,
            )
        self.assertFalse(replay["Evidence_Decision_Time_Eligible"].any())
        self.assertFalse(replay.attrs["evidence_decision_time_eligible"])
        self.assertFalse(replay.attrs["freshness_assessed"])
        self.assertTrue(
            replay["Data_Quality"]
            .str.contains("evidence_verified_replay_not_decision_time")
            .all()
        )
        self.assertFalse(replay["Execution_Eligible"].any())

        with RegimeEvidenceStore(
            Path(self.temp_dir.name) / "empty.sqlite3"
        ) as empty_store:
            with self.assertRaisesRegex(
                ValueError,
                "complete durable provider evidence",
            ):
                detect_regimes_from_verified_snapshot(
                    result.snapshot,
                    empty_store,
                )

    def test_daily_refresh_assembles_prior_verified_history_without_refetch(self):
        bootstrap_transport = ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, close=558.0),
                lambda kwargs: self._massive_ok(kwargs, close=559.0),
                lambda kwargs: self._massive_ok(kwargs, close=560.0),
                lambda kwargs: self._vix_ok(kwargs, close=20.0),
            ]
        )
        bootstrap = self._gateway(bootstrap_transport).refresh(
            HISTORY_START,
            REGULAR_SESSION,
        )
        bootstrap_transport.assert_drained(self)
        self.assertTrue(bootstrap.published)

        next_now = datetime(
            2025,
            3,
            11,
            21,
            0,
            tzinfo=timezone.utc,
        )
        daily_transport = ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, close=561.0),
                lambda kwargs: self._vix_ok(kwargs, close=19.5),
            ]
        )
        daily = self._gateway(
            daily_transport,
            now=lambda: next_now,
        ).refresh(
            NEXT_SESSION,
            NEXT_SESSION,
            snapshot_start=HISTORY_START,
        )
        daily_transport.assert_drained(self)

        self.assertTrue(daily.published)
        inputs = daily.snapshot.detector_inputs()["prices"]
        self.assertEqual(
            [item.date() for item in inputs.index],
            [
                date(2025, 3, 6),
                date(2025, 3, 7),
                REGULAR_SESSION,
                NEXT_SESSION,
            ],
        )
        self.assertEqual(
            self.store._connection.execute(
                "SELECT COUNT(*) FROM fetch_receipts "
                "WHERE provider = 'massive'"
            ).fetchone()[0],
            4,
        )
        receipt_count = self.store._connection.execute(
            "SELECT COUNT(*) FROM parser_receipts"
        ).fetchone()[0]
        with mock.patch.object(
            self.store,
            "_verify_parser_receipt",
            wraps=self.store._verify_parser_receipt,
        ) as verify_receipt:
            report = self.store.verify_snapshot(daily.snapshot)
        self.assertTrue(report.verified)
        self.assertEqual(receipt_count, 6)
        self.assertEqual(verify_receipt.call_count, receipt_count)

    def test_retry_is_bounded_captured_and_uses_injected_sleep_only(self):
        transport = ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, status_code=503),
                lambda kwargs: self._massive_ok(kwargs),
                lambda kwargs: self._vix_ok(kwargs),
            ]
        )
        result = self._gateway(transport, max_attempts=2, retry_backoff_seconds=0.25).refresh(
            REGULAR_SESSION, REGULAR_SESSION
        )
        transport.assert_drained(self)

        self.assertTrue(result.published)
        self.assertEqual(result.spy.fetch_count, 2)
        self.assertEqual(self.sleeps, [0.25])
        self.assertEqual(
            self.store._connection.execute(
                "SELECT COUNT(*) FROM fetch_receipts WHERE provider = 'massive'"
            ).fetchone()[0],
            2,
        )

    def test_body_cap_is_enforced_before_oversize_response_can_be_captured(self):
        transport = ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, body=b"x" * 129),
                lambda kwargs: self._vix_ok(kwargs),
            ]
        )
        result = self._gateway(transport, max_body_bytes=128).refresh(
            REGULAR_SESSION, REGULAR_SESSION
        )
        transport.assert_drained(self)

        self.assertFalse(result.published)
        self.assertEqual(result.spy.error_code, "RESPONSE_BODY_TOO_LARGE")
        self.assertTrue(result.vix.succeeded)
        self.assertEqual(
            self.store._connection.execute(
                "SELECT COUNT(*) FROM fetch_receipts WHERE provider = 'massive'"
            ).fetchone()[0],
            0,
        )

    def test_transport_cannot_backdate_or_future_date_gateway_receipt(self):
        def future_massive(kwargs):
            session = kwargs["requested_sessions"][0]
            return _raw_response(
                "massive",
                (session,),
                _massive_body(session),
                request_started_at=kwargs["request_started_at"],
                completed_at=pd.Timestamp(GATEWAY_NOW)
                + pd.Timedelta(minutes=1),
                request_parameters=kwargs["request_parameters"],
            )

        transport = ScriptedTransport(
            [
                future_massive,
                lambda kwargs: self._vix_ok(kwargs),
            ]
        )
        result = self._gateway(transport).refresh(
            REGULAR_SESSION,
            REGULAR_SESSION,
        )
        transport.assert_drained(self)

        self.assertFalse(result.published)
        self.assertEqual(result.spy.error_code, "EVIDENCE_CLOCK_FAILURE")
        self.assertTrue(result.vix.succeeded)
        self.assertEqual(
            self.store._connection.execute(
                "SELECT COUNT(*) FROM fetch_receipts WHERE provider = 'massive'"
            ).fetchone()[0],
            0,
        )

    def test_independent_legs_and_one_leg_failure_retains_prior_verified_head(self):
        first = self._gateway(self._success_transport()).refresh(REGULAR_SESSION, REGULAR_SESSION)
        previous = first.published_snapshot_sha256
        transport = ScriptedTransport(
            [
                lambda kwargs: self._massive_ok(kwargs, close=561.0),
                lambda kwargs: self._vix_ok(kwargs, body=_vix_body()),
            ]
        )
        result = self._gateway(transport).refresh(REGULAR_SESSION, REGULAR_SESSION)
        transport.assert_drained(self)

        self.assertTrue(result.spy.succeeded)
        self.assertFalse(result.vix.succeeded)
        self.assertFalse(result.published)
        self.assertEqual(result.vix.error_code, "VIX_PARSE_OR_EVIDENCE_FAILURE")
        self.assertEqual(
            self.store.latest_snapshot("shadow", require_verified=True).snapshot_sha256,
            previous,
        )
        self.assertEqual(self.store.source_health("massive", "SPY")["last_attempt_status"], "success")
        self.assertEqual(self.store.source_health("cboe", "VIX")["last_attempt_status"], "failure")

    def test_credentials_are_never_retained_or_exposed_by_transport_errors(self):
        response = _FakeRequestsResponse(
            _massive_body(REGULAR_SESSION),
            headers={
                "Content-Type": "application/json",
                "Authorization": AUTHORIZATION,
                "X-Api-Key": API_KEY,
            },
        )
        transport = RequestsProviderTransport(API_KEY, now=lambda: GATEWAY_NOW)
        with mock.patch("requests.get", return_value=response) as request:
            captured = transport.fetch(
                provider="massive",
                endpoint=f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/2025-03-10",
                requested_sessions=(REGULAR_SESSION,),
                request_parameters=(("adjusted", "false"),),
                request_headers=(("Accept", "application/json"),),
                timeout_seconds=1.0,
                request_started_at=GATEWAY_NOW,
                max_body_bytes=4096,
            )
        self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], AUTHORIZATION)
        self.assertFalse(request.call_args.kwargs["allow_redirects"])
        self.assertNotIn(API_KEY, json.dumps(captured.to_metadata_dict()))
        self.assertNotIn("Authorization", dict(captured.headers))
        self.assertTrue(response.closed)

        attempt_id = self.store.begin_attempt(
            "massive", "SPY", REGULAR_SESSION, REGULAR_SESSION, GATEWAY_NOW
        )
        self.store.capture_response(attempt_id, captured)
        stored = self.store._connection.execute(
            "SELECT headers_json, response_metadata_json, receipt_json FROM fetch_receipts"
        ).fetchone()
        self.assertNotIn(API_KEY, " ".join(stored))
        self.assertNotIn("Authorization", " ".join(stored))

        bad_response = _FakeRequestsResponse(
            b"body", headers={"Content-Length": "not-a-number"}
        )
        with mock.patch("requests.get", return_value=bad_response):
            with self.assertRaises(RegimeMarketDataGatewayError) as error:
                transport.fetch(
                    provider="massive",
                    endpoint=f"{MASSIVE_DAILY_SUMMARY_ENDPOINT}/2025-03-10",
                    requested_sessions=(REGULAR_SESSION,),
                    request_parameters=(("adjusted", "false"),),
                    request_headers=(("Accept", "application/json"),),
                    timeout_seconds=1.0,
                    request_started_at=GATEWAY_NOW,
                    max_body_bytes=4096,
                )
        self.assertNotIn(API_KEY, str(error.exception))

        with mock.patch("requests.get") as request:
            with self.assertRaisesRegex(
                ValueError,
                "fixed daily-summary endpoint",
            ):
                transport.fetch(
                    provider="massive",
                    endpoint="https://example.com/v1/open-close/SPY/2025-03-10",
                    requested_sessions=(REGULAR_SESSION,),
                    request_parameters=(("adjusted", "false"),),
                    request_headers=(("Accept", "application/json"),),
                    timeout_seconds=1.0,
                    request_started_at=GATEWAY_NOW,
                    max_body_bytes=4096,
                )
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
