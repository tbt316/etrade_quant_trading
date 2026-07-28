import ast
import base64
import hashlib
import importlib
import io
import json
import os
import queue
import re
import tempfile
import threading
import unittest
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch


class _ExplodingBody:
    def __init__(self):
        self.read_calls = 0

    def read(self, _length=-1):
        self.read_calls += 1
        raise AssertionError("disabled routes must not read the request body")


class _CountingBody(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.read_lengths = []

    def read(self, length=-1):
        self.read_lengths.append(length)
        return super().read(length)


class _ManualSubmission:
    def __init__(self, request_id, state="SUBMITTED"):
        self.request_id = request_id
        self.intent_id = "intent-1"
        self.state = state
        self.broker_order_id = (
            "broker-1" if state == "SUBMITTED" else None
        )
        self.reason_code = (
            None if state == "SUBMITTED" else "TEST_REASON"
        )
        self.created = True

    def dashboard_payload(self):
        return {
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "state": self.state,
            "broker_order_id": self.broker_order_id,
            "reason_code": self.reason_code,
            "created": self.created,
        }


class _ManualExecutor:
    execution_enabled = True
    account_id = "12345678"
    environment = "sandbox"
    runtime_config_sha256 = "a" * 64

    def __init__(self, state="SUBMITTED", recent=None):
        self.state = state
        self.calls = []
        self.recent = list(recent or [])
        self.recent_calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return _ManualSubmission(
            kwargs["request_id"],
            state=self.state,
        )

    def recent_submissions(self, *, limit):
        self.recent_calls.append(limit)
        return self.recent[:limit]


class _DurableManualOpenStatus:
    def __init__(
        self,
        *,
        proposal_id="proposal-1",
        status="BROKER_ACKNOWLEDGED",
        durable_state="SUBMITTED",
    ):
        self.proposal_id = proposal_id
        self.status = status
        self.durable_state = durable_state

    def dashboard_payload(self):
        return {
            "proposal_id": self.proposal_id,
            "intent_id": "intent-1",
            "status": self.status,
            "durable_state": self.durable_state,
            "broker_order_id": "broker-1",
            "reason_code": None,
            "created_at": "2026-07-27T18:00:00+00:00",
            "updated_at": "2026-07-27T18:00:01+00:00",
            "ticker": "SPY",
            "side": "PUT",
            "expiration": "2026-09-18",
            "sell_strike": "600",
            "buy_strike": "595",
            "limit_credit": "1.25",
            "quantity": 1,
        }


class _DashboardIdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, _tag, attrs):
        element_id = dict(attrs).get("id")
        if element_id:
            self.ids.add(element_id)


class DashboardReadOnlyContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(
            "live_trading.etrade_cover_call_new"
        )

    def setUp(self):
        self.module.DASHBOARD_LOGIN_FAILURE_LIMITER.record_success()
        self.module.DASHBOARD_PIN_FAILURE_LIMITER.record_success()

    def _post(self, path, *, authenticated=True):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = path
        handler.headers = {
            "Authorization": "Basic ignored",
            "Content-Length": "1048576",
        }
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (authenticated, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None

        forbidden = AssertionError(
            "disabled route reached a legacy collaborator"
        )
        with (
            patch("builtins.open", side_effect=forbidden),
            patch.object(
                self.module,
                "log_dashboard_request",
                side_effect=forbidden,
            ),
            patch.object(
                self.module,
                "enqueue_manual_trade_request",
                side_effect=forbidden,
            ),
            patch.object(
                self.module,
                "submit_manual_open_fast_async",
                side_effect=forbidden,
            ),
            patch.object(
                self.module,
                "update_manual_trade_status",
                side_effect=forbidden,
            ),
        ):
            handler._do_POST_logic()

        return (
            response_status[-1],
            json.loads(handler.wfile.getvalue()),
            handler.rfile.read_calls,
        )

    def _manual_post(
        self,
        payload,
        *,
        executor,
        content_type="application/json",
        authenticated=True,
        request_header_id=None,
    ):
        encoded = json.dumps(payload).encode("utf-8")
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = self.module.MANUAL_OPEN_DASHBOARD_PATH
        handler.headers = {
            "Authorization": "Basic ignored",
            "Content-Type": content_type,
            "Content-Length": str(len(encoded)),
            "X-Manual-Open-Request-Id": (
                request_header_id
                if request_header_id is not None
                else str(payload.get("request_id") or "")
            ),
        }
        handler.rfile = io.BytesIO(encoded)
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (authenticated, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None
        with (
            patch.object(
                self.module,
                "MANUAL_OPEN_EXECUTOR",
                executor,
            ),
            patch.object(
                self.module,
                "load_live_settings",
                return_value={"pin": "Strong-Pin-42"},
            ),
            patch.object(self.module, "log_dashboard_request"),
        ):
            handler._do_POST_logic()
        return (
            response_status[-1],
            json.loads(handler.wfile.getvalue()),
        )

    def _dashboard_json_post(
        self,
        path,
        *,
        body,
        content_length,
        content_type="application/json",
        authenticated=True,
    ):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = path
        headers = {
            "Authorization": "Basic ignored",
            "Content-Type": content_type,
        }
        if content_length is not None:
            headers["Content-Length"] = content_length
        handler.headers = headers
        handler.rfile = body
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (authenticated, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None
        handler._do_POST_logic()
        return (
            response_status[-1],
            json.loads(handler.wfile.getvalue()),
            response_headers,
        )

    def test_exact_disabled_routes_return_fixed_503_before_body_or_side_effects(self):
        expected = {
            "code": "LEGACY_EXECUTION_DISABLED",
            "error": "Trading actions are disabled.",
            "read_only": True,
            "execution_enabled": False,
        }
        queue_size = self.module.MANUAL_TRADE_QUEUE.qsize()
        manual_event = self.module.MANUAL_TRADE_REQUESTED.is_set()

        for path in sorted(self.module.DISABLED_DASHBOARD_EXECUTION_PATHS):
            with self.subTest(path=path):
                status, payload, read_calls = self._post(path)
                self.assertEqual(status, 503)
                self.assertEqual(payload, expected)
                self.assertEqual(read_calls, 0)

        self.assertEqual(
            self.module.MANUAL_TRADE_QUEUE.qsize(),
            queue_size,
        )
        self.assertEqual(
            self.module.MANUAL_TRADE_REQUESTED.is_set(),
            manual_event,
        )

    def test_query_is_stripped_but_nonexact_suffix_is_not_an_execution_route(self):
        status, payload, read_calls = self._post(
            "/api/execute_manual_order?cache_bust=1"
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["code"], "LEGACY_EXECUTION_DISABLED")
        self.assertEqual(read_calls, 0)

        status, payload, read_calls = self._post(
            "/api/execute_manual_order/extra"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "Not found"})
        self.assertEqual(read_calls, 0)

    def test_authentication_precedes_disabled_route_disclosure(self):
        status, payload, read_calls = self._post(
            "/api/execute_manual_order",
            authenticated=False,
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload, {"error": "Authentication required"})
        self.assertEqual(read_calls, 0)

    def test_get_api_routes_require_an_exact_path(self):
        for path in (
            "/api/settings-extra",
            "/api/status/extra",
            "/api/preview_spread-extra?ticker=SPY",
            "/api/positions-extra",
            "/api/gex-extra",
        ):
            with self.subTest(path=path):
                handler = object.__new__(self.module.RefreshHandler)
                handler.path = path
                handler.headers = {"Authorization": "Basic ignored"}
                handler.rfile = _ExplodingBody()
                handler.wfile = io.BytesIO()
                handler.check_auth = lambda _header: (True, 0)
                response_status = []
                handler.send_response = response_status.append
                handler.send_header = lambda _key, _value: None
                handler.end_headers = lambda: None

                with patch.object(
                    self.module,
                    "load_live_settings",
                    side_effect=AssertionError(
                        "suffix route reached a collaborator"
                    ),
                ):
                    handler._do_GET_logic()

                self.assertEqual(response_status, [404])
                self.assertEqual(
                    json.loads(handler.wfile.getvalue()),
                    {"error": "Not found"},
                )
                self.assertEqual(handler.rfile.read_calls, 0)

    def test_executable_preview_overwrites_scanner_economics_with_signed_quote(
        self,
    ):
        observed_at = datetime(
            2026, 7, 27, 18, 0, tzinfo=timezone.utc
        )
        sell_leg = SimpleNamespace(
            symbol="SPY",
            call_put="PUT",
            strike_price=700,
            expiration_date=date(2026, 8, 21),
            delta=-0.15,
            osi_key="SPY---260821P00700000",
        )
        buy_leg = SimpleNamespace(
            symbol="SPY",
            call_put="PUT",
            strike_price=695,
            expiration_date=date(2026, 8, 21),
            delta=-0.10,
            osi_key="SPY---260821P00695000",
        )
        scanner_spread = {
            "sell_option": sell_leg,
            "buy_option": buy_leg,
            "profit": Decimal("4.99"),
        }

        def option_spread(*_args, **_kwargs):
            return scanner_spread

        fake_accounts = SimpleNamespace(
            account={"accountId": "12345678"},
            get_option_spread_by_price=option_spread,
            get_stock_price=lambda _ticker: 720,
        )

        class ExactQuoteExecutor:
            def __init__(self):
                self.preview = None

            def issue_proposal(self, preview):
                self.preview = preview
                return SimpleNamespace(
                    dashboard_payload=lambda: {
                        "proposal_token": "signed-exact-quote",
                        "proposal_id": "proposal-exact-quote",
                        "proposal_expires_at":
                            "2099-08-21T18:00:00+00:00",
                        "execution_account_id": "12345678",
                        "execution_environment": "production",
                        "runtime_config_sha256": "a" * 64,
                        "ticker": "SPY",
                        "broker_symbol": "SPY",
                        "side": "PUT",
                        "expiration": "2026-08-21",
                        "sell_strike": "700",
                        "buy_strike": "695",
                        "sell_osi_key":
                            "SPY---260821P00700000",
                        "buy_osi_key":
                            "SPY---260821P00695000",
                        "premium": "1.25",
                        "limit_credit": "1.25",
                        "quote_receipt_sha256": "3" * 64,
                        "quote_snapshot_sha256": "4" * 64,
                        "sell_quote_observed_at":
                            observed_at.isoformat(),
                        "buy_quote_observed_at":
                            observed_at.isoformat(),
                        "quote_observed_at": observed_at.isoformat(),
                        "max_quantity": 5,
                        "execution_enabled": True,
                    }
                )

        executor = ExactQuoteExecutor()
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = (
            "/api/preview_spread?ticker=SPY&delta=0.15&width=5"
            "&weeks=6&expiration=2026-08-21&side=PUT"
        )
        handler.headers = {"Authorization": "Basic ignored"}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (True, 0)
        response_status = []
        handler.send_response = response_status.append
        handler.send_header = lambda _key, _value: None
        handler.end_headers = lambda: None
        capability = {
            "manual_open_enabled": True,
            "manual_open_reason": None,
        }

        with (
            patch.object(
                self.module,
                "accounts",
                fake_accounts,
                create=True,
            ),
            patch.object(
                self.module,
                "MANUAL_OPEN_EXECUTOR",
                executor,
            ),
            patch.object(
                self.module,
                "_manual_open_status_payload",
                return_value=capability,
            ),
            patch.object(
                self.module,
                "load_live_settings",
                return_value={},
            ),
        ):
            handler._do_GET_logic()

        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(response_status, [200])
        self.assertEqual(
            executor.preview.limit_credit,
            Decimal("4.99"),
        )
        self.assertEqual(payload["premium"], "1.25")
        self.assertEqual(payload["limit_credit"], "1.25")
        self.assertEqual(payload["sell_strike"], "700")
        self.assertEqual(payload["buy_strike"], "695")
        self.assertEqual(
            payload["execution_environment"],
            "production",
        )
        self.assertEqual(
            payload["runtime_config_sha256"],
            "a" * 64,
        )
        self.assertEqual(payload["quote_receipt_sha256"], "3" * 64)
        self.assertNotEqual(
            Decimal(payload["limit_credit"]),
            executor.preview.limit_credit,
        )
        self.assertEqual(handler.rfile.read_calls, 0)

    def test_supervised_manual_open_accepts_only_the_signed_envelope(self):
        request_id = str(uuid.uuid4())
        executor = _ManualExecutor()
        payload = {
            "pin": "Strong-Pin-42",
            "proposal_token": "server-signed-proposal",
            "quantity": 2,
            "request_id": request_id,
        }

        status, response = self._manual_post(
            payload,
            executor=executor,
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["state"], "SUBMITTED")
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(response["intent_id"], "intent-1")
        self.assertTrue(response["manual_open_enabled"])
        self.assertEqual(
            response["manual_open_account_id"],
            "12345678",
        )
        self.assertFalse(response["automatic_execution_enabled"])
        self.assertEqual(
            executor.calls,
            [
                {
                    "proposal_token": "server-signed-proposal",
                    "quantity": 2,
                    "request_id": request_id,
                }
            ],
        )

        browser_terms = dict(payload)
        browser_terms["sell_strike"] = 500
        status, response = self._manual_post(
            browser_terms,
            executor=executor,
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["code"], "MANUAL_OPEN_INVALID")
        self.assertEqual(
            response["submission_disposition"],
            "NOT_ATTEMPTED",
        )
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(len(executor.calls), 1)

        mismatched_header = str(uuid.uuid4())
        status, response = self._manual_post(
            payload,
            executor=executor,
            request_header_id=mismatched_header,
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            response["code"],
            "MANUAL_OPEN_REQUEST_ID_MISMATCH",
        )
        self.assertEqual(response["request_id"], mismatched_header)
        self.assertEqual(
            response["submission_disposition"],
            "NOT_ATTEMPTED",
        )
        self.assertEqual(len(executor.calls), 1)

    def test_manual_open_pin_and_uncertain_submission_fail_closed(self):
        request_id = str(uuid.uuid4())
        payload = {
            "pin": "wrong-pin",
            "proposal_token": "server-signed-proposal",
            "quantity": 1,
            "request_id": request_id,
        }
        executor = _ManualExecutor()

        status, response = self._manual_post(
            payload,
            executor=executor,
        )
        self.assertEqual(status, 403)
        self.assertEqual(response["code"], "MANUAL_OPEN_PIN_REJECTED")
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(
            response["submission_disposition"],
            "NOT_ATTEMPTED",
        )
        self.assertEqual(executor.calls, [])

        payload["pin"] = "Strong-Pin-42"
        uncertain = _ManualExecutor(state="SUBMISSION_UNKNOWN")
        status, response = self._manual_post(
            payload,
            executor=uncertain,
        )
        self.assertEqual(status, 202)
        self.assertEqual(response["state"], "SUBMISSION_UNKNOWN")
        self.assertIn("Do not create another request", response["message"])
        self.assertEqual(len(uncertain.calls), 1)

        failed = _ManualExecutor(state="FAILED")
        status, response = self._manual_post(
            payload,
            executor=failed,
        )
        self.assertEqual(status, 422)
        self.assertEqual(response["state"], "FAILED")
        self.assertIn(
            "not automatic proof that E*TRADE rejected",
            response["message"],
        )
        self.assertEqual(len(failed.calls), 1)

    def test_manual_open_rejects_unbounded_body_before_reading(self):
        for content_length in (
            str(self.module.MAX_MANUAL_OPEN_REQUEST_BYTES + 1),
            "9" * 10_000,
        ):
            with self.subTest(content_length_size=len(content_length)):
                handler = object.__new__(self.module.RefreshHandler)
                handler.path = self.module.MANUAL_OPEN_DASHBOARD_PATH
                handler.headers = {
                    "Authorization": "Basic ignored",
                    "Content-Type": "application/json",
                    "Content-Length": content_length,
                }
                handler.rfile = _ExplodingBody()
                handler.wfile = io.BytesIO()
                handler.check_auth = lambda _header: (True, 0)
                response_status = []
                handler.send_response = response_status.append
                handler.send_header = lambda _key, _value: None
                handler.end_headers = lambda: None

                handler._do_POST_logic()

                self.assertEqual(response_status, [413])
                self.assertEqual(handler.rfile.read_calls, 0)

    def test_status_projection_uses_only_durable_local_manual_open_records(self):
        executor = _ManualExecutor(
            recent=[_DurableManualOpenStatus()]
        )

        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        self.assertTrue(payload["manual_open_enabled"])
        self.assertEqual(payload["manual_open_account_id"], "12345678")
        self.assertEqual(executor.recent_calls, [25])
        self.assertEqual(
            payload["manual_open_recent"],
            [{
                "proposal_id": "proposal-1",
                "intent_id": "intent-1",
                "status": "BROKER_ACKNOWLEDGED",
                "durable_state": "SUBMITTED",
                "broker_order_id": "broker-1",
                "reason_code": None,
                "created_at": "2026-07-27T18:00:00+00:00",
                "updated_at": "2026-07-27T18:00:01+00:00",
                "ticker": "SPY",
                "side": "PUT",
                "expiration": "2026-09-18",
                "sell_strike": "600",
                "buy_strike": "595",
                "limit_credit": "1.25",
                "quantity": 1,
            }],
        )
        self.assertEqual(
            payload["manual_open_history_account_id"],
            "12345678",
        )
        self.assertEqual(
            payload["manual_open_history_environment"],
            "sandbox",
        )
        self.assertEqual(
            payload["manual_open_history_config_sha256"],
            "a" * 64,
        )

    def test_production_manual_open_binding_remains_enabled(self):
        executor = _ManualExecutor(
            recent=[_DurableManualOpenStatus()]
        )
        executor.environment = "production"

        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        self.assertTrue(payload["manual_open_enabled"])
        self.assertEqual(
            payload["manual_open_history_environment"],
            "production",
        )
        self.assertEqual(
            payload["manual_open_history_account_id"],
            "12345678",
        )

    def test_status_projection_preserves_history_when_mutation_is_disabled(self):
        executor = _ManualExecutor(
            recent=[_DurableManualOpenStatus()]
        )
        executor.execution_enabled = False

        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        self.assertFalse(payload["manual_open_enabled"])
        self.assertEqual(
            payload["manual_open_recent"][0]["proposal_id"],
            "proposal-1",
        )
        self.assertEqual(executor.recent_calls, [25])

    def test_status_projection_preserves_do_not_retry_unknown_submission(self):
        executor = _ManualExecutor(
            recent=[
                _DurableManualOpenStatus(
                    status="DO_NOT_RETRY",
                    durable_state="SUBMISSION_UNKNOWN",
                )
            ]
        )

        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        recovered = payload["manual_open_recent"][0]
        self.assertEqual(recovered["proposal_id"], "proposal-1")
        self.assertEqual(recovered["status"], "DO_NOT_RETRY")
        self.assertEqual(
            recovered["durable_state"],
            "SUBMISSION_UNKNOWN",
        )

    def test_status_projection_disables_opening_when_durable_read_fails(self):
        executor = _ManualExecutor()
        executor.recent_submissions = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("ledger integrity failure")
        )

        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        self.assertFalse(payload["manual_open_enabled"])
        self.assertIsNone(payload["manual_open_account_id"])
        self.assertEqual(payload["manual_open_recent"], [])
        self.assertEqual(
            payload["manual_open_history_account_id"],
            "12345678",
        )
        self.assertEqual(
            payload["manual_open_history_environment"],
            "sandbox",
        )
        self.assertEqual(
            payload["manual_open_history_config_sha256"],
            "a" * 64,
        )
        self.assertIn(
            "status integrity could not be verified",
            payload["manual_open_reason"],
        )

    def test_manual_open_post_is_blocked_when_durable_status_is_unreadable(self):
        request_id = str(uuid.uuid4())
        executor = _ManualExecutor()
        executor.recent_submissions = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("ledger integrity failure")
        )

        status, response = self._manual_post(
            {
                "pin": "Strong-Pin-42",
                "proposal_token": "server-signed-proposal",
                "quantity": 1,
                "request_id": request_id,
            },
            executor=executor,
        )

        self.assertEqual(status, 503)
        self.assertFalse(response["manual_open_enabled"])
        self.assertEqual(response["request_id"], request_id)
        self.assertEqual(
            response["submission_disposition"],
            "NOT_ATTEMPTED",
        )
        self.assertIn(
            "status integrity could not be verified",
            response["manual_open_reason"],
        )
        self.assertEqual(executor.calls, [])

    def test_authenticated_status_embeds_no_store_durable_manual_open_read_model(self):
        executor = _ManualExecutor(
            recent=[_DurableManualOpenStatus()]
        )
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/status"
        handler.headers = {"Authorization": "Basic ignored"}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (True, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None

        with (
            patch.object(
                self.module,
                "MANUAL_OPEN_EXECUTOR",
                executor,
            ),
            patch.object(self.module, "accounts", None, create=True),
            patch.object(
                self.module,
                "is_market_open",
                return_value=(False, "CLOSED", None, None),
            ),
            patch.object(
                self.module,
                "load_trade_status",
                return_value={},
            ),
            patch.object(
                self.module,
                "load_live_settings",
                return_value={"target_weeks": 6},
            ),
            patch.object(
                self.module,
                "get_manual_trade_status_snapshot",
                return_value={"latest": {}, "requests": []},
            ),
        ):
            handler._do_GET_logic()

        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(response_status, [200])
        self.assertEqual(
            response_headers["Cache-Control"],
            "no-store, max-age=0",
        )
        self.assertEqual(
            response_headers["Cross-Origin-Resource-Policy"],
            "same-origin",
        )
        self.assertEqual(
            payload["manual_open_recent"][0]["proposal_id"],
            "proposal-1",
        )
        self.assertEqual(executor.recent_calls, [25])
        self.assertEqual(handler.rfile.read_calls, 0)

    def test_scheduled_oauth_renewal_uses_the_full_refresh_path(self):
        now = datetime(2026, 7, 27, 12, 0)
        renewed = (object(), "https://apisb.etrade.test")
        with patch.object(
            self.module,
            "_refresh_etrade_session",
            return_value=renewed,
        ) as refresh:
            self.assertIsNone(
                self.module._renew_etrade_session_if_due(
                    now,
                    now - timedelta(minutes=59, seconds=59),
                )
            )
            self.assertEqual(
                self.module._renew_etrade_session_if_due(
                    now,
                    now - timedelta(minutes=60),
                ),
                renewed,
            )

        refresh.assert_called_once_with("scheduled renewal")

    def test_full_oauth_refresh_rebuilds_manual_open_executor(self):
        new_session = object()
        new_accounts = object()
        new_market = object()
        with (
            patch.object(
                self.module,
                "oauth",
                return_value=(
                    new_session,
                    "https://apisb.etrade.test",
                ),
            ),
            patch.object(
                self.module,
                "Accounts",
                return_value=new_accounts,
            ),
            patch.object(
                self.module,
                "Market",
                return_value=new_market,
            ),
            patch.object(
                self.module,
                "_select_runtime_account",
            ) as select_account,
            patch.object(
                self.module,
                "_attach_etrade_auth_refresh_callbacks",
            ),
            patch.object(
                self.module,
                "_configure_manual_open_executor",
            ) as configure_executor,
            patch.object(
                self.module,
                "use_sandbox",
                True,
                create=True,
            ),
            patch.object(
                self.module,
                "accounts",
                object(),
                create=True,
            ),
            patch.object(
                self.module,
                "market",
                object(),
                create=True,
            ),
            patch.object(
                self.module,
                "etrade_instance",
                None,
                create=True,
            ),
            patch.object(
                self.module,
                "session",
                object(),
                create=True,
            ),
            patch.object(
                self.module,
                "base_url",
                "https://old.etrade.test",
                create=True,
            ),
            patch.object(
                self.module,
                "last_renewal_time",
                datetime(2026, 7, 27, 10, 0),
                create=True,
            ),
        ):
            refreshed = self.module._refresh_etrade_session(
                "scheduled renewal"
            )

        self.assertEqual(
            refreshed,
            (new_session, "https://apisb.etrade.test"),
        )
        select_account.assert_called_once_with(new_accounts)
        configure_executor.assert_called_once_with(new_session)

    def test_mutation_entrypoints_are_single_reject_call_tombstones(self):
        expected_reasons = {
            "enqueue_manual_trade_request":
                "legacy dashboard manual-trade queue",
            "_execute_neutralize_leg":
                "legacy dashboard neutralize worker",
            "release_margin":
                "legacy automatic margin release",
            "submit_manual_open_fast_async":
                "legacy dashboard manual-open scheduler",
            "_manual_open_fast_worker":
                "legacy dashboard manual-open worker",
            "submit_order_async":
                "legacy dashboard order scheduler",
            "_order_worker":
                "legacy dashboard order worker",
            "monitor_and_nudge_stale_orders":
                "legacy stale-order nudge worker",
        }
        module_source = Path(self.module.__file__).read_text(
            encoding="utf-8"
        )
        module_ast = ast.parse(module_source)
        functions = {
            node.name: node
            for node in module_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        for name, reason in expected_reasons.items():
            with self.subTest(name=name):
                body = functions[name].body
                self.assertEqual(len(body), 1)
                call = body[0]
                self.assertIsInstance(call, ast.Expr)
                self.assertIsInstance(call.value, ast.Call)
                self.assertEqual(
                    getattr(call.value.func, "id", None),
                    "reject_legacy_execution",
                )
                self.assertEqual(
                    ast.literal_eval(call.value.args[0]),
                    reason,
                )

    def test_dequeue_never_returns_or_processes_legacy_work(self):
        pending = queue.Queue()
        pending.put({"request_id": "legacy"})
        event = self.module.threading.Event()
        event.set()
        with (
            patch.object(self.module, "MANUAL_TRADE_QUEUE", pending),
            patch.object(self.module, "MANUAL_TRADE_REQUESTED", event),
            patch.object(
                self.module,
                "update_manual_trade_status",
                side_effect=AssertionError("status must not change"),
            ),
        ):
            self.assertIsNone(self.module.dequeue_manual_trade_request())
            self.assertFalse(event.is_set())
            self.assertEqual(pending.qsize(), 1)

    def test_settings_migrate_and_persist_auto_open_as_false(self):
        settings = {
            "target_delta": 0.13,
            "hedge_spread": 20.0,
            "spy_hedge_spread": 20.0,
            "spx_hedge_spread": 200.0,
            "pair_quantity": 1,
            "spy_pair_quantity": 1,
            "spx_pair_quantity": 1,
            "target_expiration": None,
            "spy_target_expiration": None,
            "spx_target_expiration": None,
            "auto_open_enabled": True,
            "pin": "87654321",
            "dashboard_user": "read-only-operator",
            "dashboard_pass": "a-strong-dashboard-password",
            "dashboard_auth_secret": hashlib.sha256(
                b"dashboard-auth-secret"
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            os.chmod(path, 0o600)

            with patch.object(
                self.module,
                "LIVE_SETTINGS_FILE",
                str(path),
            ):
                loaded = self.module.load_live_settings()
                self.assertIs(loaded["auto_open_enabled"], False)
                self.assertIs(
                    json.loads(path.read_text(encoding="utf-8"))[
                        "auto_open_enabled"
                    ],
                    False,
                )

                loaded["auto_open_enabled"] = True
                self.assertTrue(self.module.save_live_settings(loaded))
                self.assertIs(
                    json.loads(path.read_text(encoding="utf-8"))[
                        "auto_open_enabled"
                    ],
                    False,
                )

    def test_dashboard_exposes_only_supervised_manual_open(self):
        template = Path(self.module.__file__).with_name(
            "dashboard_template.html"
        ).read_text(encoding="utf-8")
        accounts_source = (
            Path(self.module.__file__).parents[1]
            / "accounts"
            / "accounts_bo.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "Manual opening unavailable",
            template,
        )
        self.assertIn(
            "Read only — all E*TRADE order actions are disabled",
            accounts_source,
        )
        for endpoint in self.module.DISABLED_DASHBOARD_EXECUTION_PATHS:
            self.assertNotIn(endpoint, template)
            self.assertNotIn(endpoint, accounts_source)
        for fragment in (
            "data-close-position",
            "openConfirmForTicker",
            "confirmClose",
            "confirmNeutralize",
            "executeNow",
            "executeClose",
            "executeNeutralize",
        ):
            self.assertNotIn(fragment, template)
            self.assertNotIn(fragment, accounts_source)
        self.assertNotIn("order.view_orders()", accounts_source)

        self.assertIn("/api/preview_spread", template)
        self.assertIn("/api/manual_open", template)
        self.assertIn("crypto.randomUUID()", template)
        self.assertIn("Submit Once", template)
        self.assertIn("manual_open_recent", template)
        self.assertIn("manual_open_attempt.v1", template)
        self.assertIn(
            "item => String(item?.proposal_id || '') === stored.proposal_id",
            template,
        )
        self.assertIn(
            "SUBMITTED / BROKER ACKNOWLEDGED",
            template,
        )
        self.assertIn("This is not a fill confirmation", template)
        self.assertIn("DO NOT RETRY this proposal", template)
        self.assertIn(
            "A durable manual-open submission is unresolved. DO NOT RETRY",
            template,
        )
        self.assertIn(
            "manualOpenRecoveryBlocked = manualOpenRequiresReconciliation",
            template,
        )
        self.assertIn(
            "This does not automatically mean E*TRADE rejected the order",
            template,
        )
        submit_start = template.index(
            "async function submitConfirmedManualOpen()"
        )
        submit_end = template.index(
            "const PORTFOLIO_REFRESH_INTERVAL_MS",
            submit_start,
        )
        submit_source = template[submit_start:submit_end]
        self.assertLess(
            submit_source.index(
                "persistManualOpenAttempt(pendingManualOpen)"
            ),
            submit_source.index("fetch('/api/manual_open'"),
        )
        self.assertEqual(
            template.count("fetch('/api/manual_open'"),
            1,
        )
        self.assertNotIn("/api/manual_open", accounts_source)
        self.assertIn("fetch('/refresh'", template)
        self.assertIn('src="/api/positions"', template)
        self.assertIn("minlength=\"8\"", template)
        self.assertIn("slice(0, 64)", template)
        self.assertNotIn("slice(0, 4)", template)
        self.assertNotIn("replace(/\\D/g", template)
        self.assertNotIn('inputmode="numeric"', template)
        self.assertNotIn('pattern="[0-9]*"', template)
        self.assertIn("manual_open_attempt.v2", template)
        self.assertIn("manual_open_history_account_id", template)
        self.assertIn("manual_open_history_environment", template)
        self.assertIn("manual_open_history_config_sha256", template)
        self.assertIn("data.execution_environment", template)
        self.assertIn("data.runtime_config_sha256", template)
        self.assertIn(
            "currentManualOpenEnvironment.toUpperCase()",
            template,
        )
        self.assertIn("X-Manual-Open-Request-Id", template)
        self.assertIn("submission_disposition", template)
        self.assertIn(
            "!response.ok\n"
            "                    && payload?.submission_disposition "
            "=== 'NOT_ATTEMPTED'",
            template,
        )
        self.assertIn("isValidatedManualOpenResponse(", template)
        self.assertIn("state: 'SUBMISSION_UNKNOWN'", template)
        self.assertIn("fetch('/api/logout'", template)
        self.assertIn("window.location.replace('/login')", template)
        self.assertIn("manual-open-confirm-pin", template)
        self.assertNotIn("currentDashboardPin", template)
        self.assertNotIn("let currentPin", template)
        self.assertNotIn("sessionStorage.setItem('dashboard_pin'", template)
        self.assertNotIn("innerHTML", template)
        self.assertIsNone(re.search(r"\son[a-z]+=", template))
        self.assertNotIn("fonts.googleapis.com", template)
        self.assertIn(
            "chart.js@4.4.7/dist/chart.umd.min.js",
            template,
        )
        self.assertIn(
            "sha384-vsrfeLOOY6KuIYKDlmVH5UiBmgIdB1oEf7p01YgWHuqmOHfZr374+odEv96n9tNC",
            template,
        )
        self.assertEqual(
            template.count(self.module.DASHBOARD_CSP_NONCE_PLACEHOLDER),
            2,
        )
        self.assertIn(
            (
                "if (!frameDocument || !frameDocument.documentElement "
                "|| !frameDocument.body) return;"
            ),
            template,
        )
        self.assertIn(
            "frame.contentWindow && frame.contentWindow.ResizeObserver",
            template,
        )
        self.assertIn(
            "data-positions-artifact-state",
            self.module._read_only_positions_fallback(),
        )
        self.assertIn(
            "compactFallback ? 120 : 320",
            template,
        )

        parser = _DashboardIdParser()
        parser.feed(template)
        self.assertTrue({
            "manual-order-status",
            "manual-order-status-body",
            "manual-open-confirm-overlay",
            "manual-open-submit-btn",
            "spy-manual-open-btn",
            "spx-manual-open-btn",
            "manual-open-confirm-pin",
            "manual-open-confirm-environment",
        }.issubset(parser.ids))

    def test_dashboard_and_login_are_nonce_bound_under_csp(self):
        for path in ("/dashboard", "/login"):
            with self.subTest(path=path):
                handler = object.__new__(self.module.RefreshHandler)
                handler.path = path
                handler.headers = {"Authorization": "Basic ignored"}
                handler.rfile = _ExplodingBody()
                handler.wfile = io.BytesIO()
                handler.check_auth = lambda _header: (True, 0)
                response_status = []
                response_headers = {}
                handler.send_response = response_status.append
                handler.send_header = response_headers.__setitem__
                handler.end_headers = lambda: None

                with patch.object(
                    Path,
                    "read_bytes",
                    side_effect=AssertionError(
                        "dashboard template must stay pinned"
                    ),
                ):
                    handler._do_GET_logic()

                rendered = handler.wfile.getvalue().decode("utf-8")
                csp = response_headers["Content-Security-Policy"]
                nonce_match = re.search(
                    r"script-src 'nonce-([^']+)' 'strict-dynamic'",
                    csp,
                )
                self.assertIsNotNone(nonce_match)
                nonce = nonce_match.group(1)
                self.assertIn(f'nonce="{nonce}"', rendered)
                self.assertNotIn(
                    self.module.DASHBOARD_CSP_NONCE_PLACEHOLDER,
                    rendered,
                )
                self.assertIn("object-src 'none'", csp)
                self.assertIn("frame-ancestors 'none'", csp)
                self.assertNotIn(
                    "script-src 'self' 'unsafe-inline'",
                    csp,
                )
                self.assertEqual(response_status, [200])
                if path == "/dashboard":
                    self.assertEqual(
                        response_headers[
                            "X-Dashboard-Build-SHA256"
                        ],
                        self.module.PINNED_DASHBOARD_TEMPLATE_SHA256,
                    )

    def test_dashboard_failure_limiter_is_atomic_and_expires(self):
        now = [100.0]
        limiter = self.module._DashboardFailureLimiter(
            max_failures=3,
            window_seconds=30,
            lockout_seconds=20,
            clock=lambda: now[0],
        )
        barrier = threading.Barrier(3)
        results = []

        def fail():
            barrier.wait()
            results.append(limiter.record_failure())

        threads = [threading.Thread(target=fail) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sum(value > 0 for value in results), 1)
        self.assertEqual(limiter.retry_after(), 20)
        now[0] += 21
        self.assertEqual(limiter.retry_after(), 0)
        limiter.record_failure()
        limiter.record_success()
        self.assertEqual(limiter.retry_after(), 0)

    def test_dashboard_json_endpoints_reject_unsafe_bodies_before_use(self):
        endpoints = (
            "/api/login",
            "/api/verify_pin",
            "/api/settings",
        )
        cases = (
            (
                "missing_length",
                None,
                lambda: _ExplodingBody(),
                411,
                "DASHBOARD_JSON_LENGTH_REQUIRED",
            ),
            (
                "invalid_length",
                "not-a-number",
                lambda: _ExplodingBody(),
                400,
                "DASHBOARD_JSON_INVALID_LENGTH",
            ),
            (
                "negative_length",
                "-1",
                lambda: _ExplodingBody(),
                400,
                "DASHBOARD_JSON_INVALID_LENGTH",
            ),
            (
                "empty_body",
                "0",
                lambda: _ExplodingBody(),
                400,
                "DASHBOARD_JSON_EMPTY_BODY",
            ),
            (
                "oversized",
                str(
                    self.module.MAX_DASHBOARD_JSON_REQUEST_BYTES
                    + 1
                ),
                lambda: _ExplodingBody(),
                413,
                "DASHBOARD_JSON_BODY_TOO_LARGE",
            ),
            (
                "pathologically_large_length",
                "9" * 10000,
                lambda: _ExplodingBody(),
                413,
                "DASHBOARD_JSON_BODY_TOO_LARGE",
            ),
            (
                "incomplete",
                "3",
                lambda: _CountingBody(b"{}"),
                400,
                "DASHBOARD_JSON_INCOMPLETE_BODY",
            ),
            (
                "malformed",
                "1",
                lambda: _CountingBody(b"{"),
                400,
                "DASHBOARD_JSON_INVALID_BODY",
            ),
            (
                "non_object",
                "2",
                lambda: _CountingBody(b"[]"),
                400,
                "DASHBOARD_JSON_INVALID_BODY",
            ),
        )

        for endpoint in endpoints:
            for (
                case,
                content_length,
                body_factory,
                expected_status,
                expected_code,
            ) in cases:
                with self.subTest(endpoint=endpoint, case=case):
                    self.module.DASHBOARD_LOGIN_FAILURE_LIMITER.record_success()
                    self.module.DASHBOARD_PIN_FAILURE_LIMITER.record_success()
                    body = body_factory()
                    with patch.object(
                        self.module,
                        "load_live_settings",
                        side_effect=AssertionError(
                            "invalid JSON must not reach settings"
                        ),
                    ):
                        status, payload, _headers = (
                            self._dashboard_json_post(
                                endpoint,
                                body=body,
                                content_length=content_length,
                            )
                        )
                    self.assertEqual(status, expected_status)
                    self.assertEqual(payload["code"], expected_code)
                    if isinstance(body, _ExplodingBody):
                        self.assertEqual(body.read_calls, 0)

    def test_dashboard_json_endpoints_require_json_content_type(self):
        for endpoint in (
            "/api/login",
            "/api/verify_pin",
            "/api/settings",
        ):
            with self.subTest(endpoint=endpoint):
                body = _ExplodingBody()
                status, payload, _headers = self._dashboard_json_post(
                    endpoint,
                    body=body,
                    content_length="2",
                    content_type="text/plain",
                )
                self.assertEqual(status, 415)
                self.assertEqual(
                    payload["code"],
                    "DASHBOARD_JSON_CONTENT_TYPE",
                )
                self.assertEqual(body.read_calls, 0)

    def test_successful_login_and_basic_auth_clear_prior_failures(self):
        settings = {
            "dashboard_user": "operator",
            "dashboard_pass": "correct horse battery staple",
            "dashboard_auth_secret": hashlib.sha256(
                b"dashboard-auth-secret"
            ).hexdigest(),
        }

        form_limiter = self.module._DashboardFailureLimiter(
            max_failures=2,
            window_seconds=60,
            lockout_seconds=30,
        )
        self.assertEqual(form_limiter.record_failure(), 0)
        login_body = json.dumps(
            {
                "username": settings["dashboard_user"],
                "password": settings["dashboard_pass"],
            }
        ).encode("utf-8")
        with (
            patch.object(
                self.module,
                "DASHBOARD_LOGIN_FAILURE_LIMITER",
                form_limiter,
            ),
            patch.object(
                self.module,
                "load_live_settings",
                return_value=settings,
            ),
        ):
            status, payload, headers = self._dashboard_json_post(
                "/api/login",
                body=_CountingBody(login_body),
                content_length=str(len(login_body)),
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("Set-Cookie", headers)
        self.assertEqual(form_limiter.record_failure(), 0)

        basic_limiter = self.module._DashboardFailureLimiter(
            max_failures=2,
            window_seconds=60,
            lockout_seconds=30,
        )
        self.assertEqual(basic_limiter.record_failure(), 0)
        authorization = "Basic " + base64.b64encode(
            (
                f"{settings['dashboard_user']}:"
                f"{settings['dashboard_pass']}"
            ).encode("utf-8")
        ).decode("ascii")
        handler = object.__new__(self.module.RefreshHandler)
        handler.headers = {"Authorization": authorization}
        with (
            patch.object(
                self.module,
                "DASHBOARD_LOGIN_FAILURE_LIMITER",
                basic_limiter,
            ),
            patch.object(
                self.module,
                "load_live_settings",
                return_value=settings,
            ),
        ):
            self.assertEqual(
                handler.check_auth(authorization),
                (True, 0),
            )
        self.assertEqual(basic_limiter.record_failure(), 0)

    def test_blank_dashboard_password_preserves_existing_secret(self):
        current_settings = {
            "pin": "Strong-Pin-42",
            "dashboard_user": "operator",
            "dashboard_pass": "existing dashboard password",
            "dashboard_auth_secret": hashlib.sha256(
                b"dashboard-auth-secret"
            ).hexdigest(),
            "auto_open_enabled": False,
        }
        request = {
            "pin": current_settings["pin"],
            "dashboard_user": "renamed-operator",
            "dashboard_pass": "",
            "target_delta": 0.17,
        }
        body = json.dumps(request).encode("utf-8")
        saved = []

        def save(settings):
            saved.append(dict(settings))
            return True

        with (
            patch.object(
                self.module,
                "load_live_settings",
                return_value=dict(current_settings),
            ),
            patch.object(
                self.module,
                "save_live_settings",
                side_effect=save,
            ),
            patch.object(self.module, "log_dashboard_request"),
        ):
            status, payload, _headers = self._dashboard_json_post(
                "/api/settings",
                body=_CountingBody(body),
                content_length=str(len(body)),
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            saved[0]["dashboard_pass"],
            current_settings["dashboard_pass"],
        )
        self.assertEqual(
            saved[0]["dashboard_user"],
            "renamed-operator",
        )
        self.assertNotEqual(
            saved[0]["dashboard_auth_secret"],
            current_settings["dashboard_auth_secret"],
        )
        self.assertRegex(
            saved[0]["dashboard_auth_secret"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(saved[0]["target_delta"], 0.17)

    def test_action_pin_rate_limit_is_shared_across_endpoints(self):
        verify_body = json.dumps({"pin": "wrong-pin"}).encode("utf-8")
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/verify_pin"
        handler.headers = {
            "Authorization": "Basic ignored",
            "Content-Type": "application/json",
            "Content-Length": str(len(verify_body)),
            "X-Forwarded-For": "203.0.113.99",
        }
        handler.rfile = io.BytesIO(verify_body)
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (True, 0)
        response_status = []
        handler.send_response = response_status.append
        handler.send_header = lambda _key, _value: None
        handler.end_headers = lambda: None
        with patch.object(
            self.module,
            "load_live_settings",
            return_value={"pin": "Strong-Pin-42"},
        ):
            handler._do_POST_logic()
        self.assertEqual(response_status, [403])

        request_id = str(uuid.uuid4())
        request = {
            "pin": "wrong-pin",
            "proposal_token": "server-signed-proposal",
            "quantity": 1,
            "request_id": request_id,
        }
        final_status = None
        final_response = None
        for _ in range(self.module.DASHBOARD_PIN_MAX_FAILURES - 1):
            final_status, final_response = self._manual_post(
                request,
                executor=_ManualExecutor(),
            )

        self.assertEqual(final_status, 429)
        self.assertEqual(
            final_response["code"],
            "DASHBOARD_AUTH_RATE_LIMITED",
        )
        self.assertEqual(final_response["request_id"], request_id)
        self.assertEqual(
            final_response["submission_disposition"],
            "NOT_ATTEMPTED",
        )

        blocked_handler = object.__new__(self.module.RefreshHandler)
        blocked_handler.path = self.module.MANUAL_OPEN_DASHBOARD_PATH
        blocked_handler.headers = {
            "Authorization": "Basic ignored",
            "Content-Type": "application/json",
            "Content-Length": "100",
            "X-Manual-Open-Request-Id": request_id,
            "X-Forwarded-For": "198.51.100.22",
        }
        blocked_handler.rfile = _ExplodingBody()
        blocked_handler.wfile = io.BytesIO()
        blocked_handler.check_auth = lambda _header: (True, 0)
        blocked_status = []
        blocked_handler.send_response = blocked_status.append
        blocked_handler.send_header = lambda _key, _value: None
        blocked_handler.end_headers = lambda: None
        blocked_handler._do_POST_logic()
        self.assertEqual(blocked_status, [429])
        self.assertEqual(blocked_handler.rfile.read_calls, 0)

    def test_manual_status_rejects_future_projection_fields(self):
        class StatusWithSecret(_DurableManualOpenStatus):
            def dashboard_payload(self):
                payload = super().dashboard_payload()
                payload["proposal_token"] = "must-not-leak"
                return payload

        executor = _ManualExecutor(recent=[StatusWithSecret()])
        with patch.object(
            self.module,
            "MANUAL_OPEN_EXECUTOR",
            executor,
        ):
            payload = self.module._manual_open_status_payload()

        self.assertFalse(payload["manual_open_enabled"])
        self.assertEqual(payload["manual_open_recent"], [])
        self.assertNotIn("proposal_token", json.dumps(payload))

    def test_logout_is_exact_post_and_clears_cookie(self):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/logout"
        handler.headers = {}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (False, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None

        handler._do_POST_logic()

        self.assertEqual(response_status, [200])
        self.assertIn("Max-Age=0", response_headers["Set-Cookie"])
        self.assertEqual(handler.rfile.read_calls, 0)

        handler.path = "/api/logout-extra"
        handler.wfile = io.BytesIO()
        response_status.clear()
        handler._do_POST_logic()
        self.assertEqual(response_status, [401])

    def test_positions_artifact_can_render_only_inside_same_origin_dashboard(self):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/positions"
        handler.headers = {"Authorization": "Basic ignored"}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (True, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None

        safe_html = (
            "<p>Read only — all E*TRADE order actions are disabled</p>"
        )
        with (
            patch.object(self.module.os.path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data=safe_html)),
            patch.object(
                self.module,
                "_repair_benchmark_option_value_gaps",
                side_effect=lambda content: content,
            ),
        ):
            handler._do_GET_logic()

        self.assertEqual(response_status, [200])
        self.assertEqual(response_headers["X-Frame-Options"], "SAMEORIGIN")
        self.assertEqual(
            response_headers["Content-Security-Policy"],
            self.module.POSITIONS_FRAME_CSP,
        )
        self.assertIn(
            "script-src 'none'",
            response_headers["Content-Security-Policy"],
        )
        self.assertEqual(handler.wfile.getvalue(), safe_html.encode())
        self.assertEqual(handler.rfile.read_calls, 0)

    def test_stale_positions_artifact_is_replaced_before_serving(self):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/positions"
        handler.headers = {"Authorization": "Basic ignored"}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: (True, 0)
        response_status = []
        response_headers = {}
        handler.send_response = response_status.append
        handler.send_header = response_headers.__setitem__
        handler.end_headers = lambda: None
        stale = (
            '<button data-close-position="1">Close</button>'
            "<script>fetch('/api/close_position')</script>"
        )

        with (
            patch.object(self.module.os.path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data=stale)),
            patch.object(
                self.module,
                "_repair_benchmark_option_value_gaps",
                side_effect=lambda content: content,
            ),
        ):
            handler._do_GET_logic()

        rendered = handler.wfile.getvalue().decode("utf-8")
        self.assertEqual(response_status, [503])
        self.assertIn(self.module.POSITIONS_READ_ONLY_MARKER, rendered)
        self.assertNotIn("data-close-position", rendered)
        self.assertNotIn("/api/close_position", rendered)
        self.assertNotIn("<script", rendered)
        self.assertEqual(
            response_headers["Content-Security-Policy"],
            self.module.POSITIONS_FRAME_CSP,
        )
        self.assertEqual(handler.rfile.read_calls, 0)

    def test_current_read_only_generator_shape_is_accepted(self):
        generated = f"""
        <!doctype html>
        <style>.read-only-cell {{ color: #fecaca; }}</style>
        <div>{self.module.POSITIONS_READ_ONLY_MARKER}</div>
        <table><tr><td class="read-only-cell">Read only</td></tr></table>
        <script>
          window.setInterval(() => window.location.reload(), 60000);
        </script>
        """

        rendered, valid = self.module._validated_positions_artifact(
            generated
        )

        self.assertTrue(valid)
        self.assertEqual(rendered, generated)
        self.assertNotIn("data-close-position", rendered)
        for endpoint in self.module.DISABLED_DASHBOARD_EXECUTION_PATHS:
            self.assertNotIn(endpoint, rendered)


if __name__ == "__main__":
    unittest.main()
