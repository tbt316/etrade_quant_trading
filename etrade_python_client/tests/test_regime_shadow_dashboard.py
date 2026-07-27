import base64
import importlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch


class _ShadowStoreStub:
    def __init__(self):
        self.calls = 0
        self.payload = {
            "available": False,
            "status": "unavailable",
            "source_family": "v2_shadow",
            "as_of_session": None,
            "effective_session": None,
            "background_state": "unavailable",
            "shock_state": "unavailable",
            "composite_label": "unavailable+unavailable",
            "availability": "unavailable",
            "reason_codes": ["missing"],
            "abstain_reasons": [
                "missing",
                "r4_shadow_non_authoritative",
            ],
            "stale": False,
            "may_authorize_execution": False,
        }

    def dashboard_payload(self):
        self.calls += 1
        return dict(self.payload)


class RegimeShadowDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(
            "live_trading.etrade_cover_call_new"
        )
        cls.original_load_settings = cls.module.load_live_settings
        cls.original_store = cls.module.REGIME_V2_SHADOW_STORE
        cls.store = _ShadowStoreStub()
        cls.module.REGIME_V2_SHADOW_STORE = cls.store
        cls.module.load_live_settings = lambda: {
            "dashboard_user": "shadow-test",
            "dashboard_pass": "test-password",
            "dashboard_auth_secret": "test-secret",
        }

    @classmethod
    def tearDownClass(cls):
        cls.module.load_live_settings = cls.original_load_settings
        cls.module.REGIME_V2_SHADOW_STORE = cls.original_store

    def setUp(self):
        self.store.calls = 0
        self.store.payload = {
            "available": False,
            "status": "unavailable",
            "source_family": "v2_shadow",
            "as_of_session": None,
            "effective_session": None,
            "background_state": "unavailable",
            "shock_state": "unavailable",
            "composite_label": "unavailable+unavailable",
            "availability": "unavailable",
            "reason_codes": ["missing"],
            "abstain_reasons": [
                "missing",
                "r4_shadow_non_authoritative",
            ],
            "stale": False,
            "may_authorize_execution": False,
        }

    @property
    def auth_header(self):
        token = base64.b64encode(
            b"shadow-test:test-password"
        ).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    def _request(self, path, *, authenticated):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = path
        handler.headers = self.auth_header if authenticated else {}
        handler.wfile = io.BytesIO()
        response_headers = {}
        response_status = []
        handler.send_response = lambda code: response_status.append(code)
        handler.send_header = (
            lambda name, value: response_headers.__setitem__(
                name.lower(),
                value,
            )
        )
        handler.end_headers = lambda: None

        handler._do_GET_logic()

        return response_status[-1], response_headers, handler.wfile.getvalue()

    def test_unauthenticated_request_is_generic_and_does_not_read_store(self):
        status, headers, body = self._request(
            "/api/regime_v2_shadow",
            authenticated=False,
        )

        self.assertEqual(status, 401)
        self.assertEqual(
            json.loads(body),
            {"error": "Authentication required"},
        )
        self.assertEqual(self.store.calls, 0)
        self.assertNotIn("access-control-allow-origin", headers)
        self.assertEqual(
            headers["cross-origin-resource-policy"],
            "same-origin",
        )

    def test_valid_route_is_redacted_same_origin_and_execution_isolated(self):
        self.store.payload = {
            "available": True,
            "status": "advisory",
            "source_family": "v2_shadow",
            "as_of_session": "2026-07-24",
            "effective_session": "2026-07-27",
            "background_state": "elevated",
            "shock_state": "active",
            "composite_label": "elevated+active",
            "availability": "advisory",
            "reason_codes": ["vix_daily_change_extreme"],
            "abstain_reasons": [
                "calibration_lineage_missing",
                "execution_ineligible",
                "r4_shadow_non_authoritative",
            ],
            "stale": False,
            "may_authorize_execution": False,
        }
        close_before = list(self.module.CURRENT_CLOSE_PROPOSALS)
        neutralize_before = list(self.module.CURRENT_NEUTRALIZE_PROPOSALS)
        refresh_before = self.module.REFRESH_REQUESTED.is_set()
        manual_before = self.module.MANUAL_TRADE_REQUESTED.is_set()

        with (
            patch.object(
                self.module,
                "calculate_spy_gex",
                side_effect=AssertionError("GEX path must not run"),
            ),
            patch.object(
                self.module,
                "calculate_spy_regime_status",
                side_effect=AssertionError("legacy HMM path must not run"),
            ),
            patch.object(
                self.module,
                "get_probability_engine",
                side_effect=AssertionError("EV path must not run"),
            ),
            patch.object(
                self.module,
                "build_regime_return_arrays",
                side_effect=AssertionError("return buckets must not run"),
            ),
        ):
            status, headers, body = self._request(
                "/api/regime_v2_shadow",
                authenticated=True,
            )

        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(self.store.calls, 1)
        self.assertEqual(
            headers["content-type"],
            "application/json; charset=utf-8",
        )
        self.assertEqual(headers["cache-control"], "no-store, max-age=0")
        self.assertNotIn("access-control-allow-origin", headers)
        self.assertFalse(payload["may_authorize_execution"])
        self.assertNotIn("lineage", payload)
        self.assertNotIn("signal_sha256", payload)
        self.assertNotIn("available_at", payload)
        self.assertEqual(
            self.module.CURRENT_CLOSE_PROPOSALS,
            close_before,
        )
        self.assertEqual(
            self.module.CURRENT_NEUTRALIZE_PROPOSALS,
            neutralize_before,
        )
        self.assertEqual(
            self.module.REFRESH_REQUESTED.is_set(),
            refresh_before,
        )
        self.assertEqual(
            self.module.MANUAL_TRADE_REQUESTED.is_set(),
            manual_before,
        )

    def test_unavailable_and_nonexact_routes_fail_closed(self):
        status, _, body = self._request(
            "/api/regime_v2_shadow?cache_bust=1",
            authenticated=True,
        )
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["reason_codes"], ["missing"])
        self.assertEqual(self.store.calls, 1)

        self.store.payload["lineage"] = {"snapshot_sha256": "secret"}
        status, _, body = self._request(
            "/api/regime_v2_shadow",
            authenticated=True,
        )
        payload = json.loads(body)
        self.assertEqual(status, 503)
        self.assertEqual(payload["reason_codes"], ["invalid"])
        self.assertNotIn("lineage", payload)
        self.assertEqual(self.store.calls, 2)

        status, _, _ = self._request(
            "/api/regime_v2_shadow/extra",
            authenticated=True,
        )
        self.assertEqual(status, 404)
        self.assertEqual(self.store.calls, 2)

    def test_template_uses_separate_endpoint_and_text_only_v2_rendering(self):
        template_path = Path(self.module.__file__).with_name(
            "dashboard_template.html"
        )
        html = template_path.read_text(encoding="utf-8")
        render_section = html.split(
            "function renderRegimeV2ShadowStatus(payload)",
            1,
        )[1].split("async function fetchRegimeV2Shadow()", 1)[0]
        fetch_section = html.split(
            "async function fetchRegimeV2Shadow()",
            1,
        )[1].split("async function fetchGexData()", 1)[0]

        self.assertIn("Regime V2 Shadow — Advisory Only", html)
        self.assertIn("CANNOT AUTHORIZE EXECUTION", html)
        self.assertNotIn("innerHTML", render_section)
        self.assertGreaterEqual(render_section.count("textContent"), 10)
        self.assertIn("fetch('/api/regime_v2_shadow'", fetch_section)
        self.assertNotIn("/api/gex", fetch_section)


if __name__ == "__main__":
    unittest.main()
