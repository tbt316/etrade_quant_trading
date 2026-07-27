import ast
import importlib
import io
import json
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch


class _ExplodingBody:
    def __init__(self):
        self.read_calls = 0

    def read(self, _length=-1):
        self.read_calls += 1
        raise AssertionError("disabled routes must not read the request body")


class DashboardReadOnlyContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(
            "live_trading.etrade_cover_call_new"
        )

    def _post(self, path, *, authenticated=True):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = path
        handler.headers = {
            "Authorization": "Basic ignored",
            "Content-Length": "1048576",
        }
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: authenticated
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
            "dashboard_auth_secret": "a" * 64,
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

    def test_source_dashboards_are_permanently_read_only(self):
        template = Path(self.module.__file__).with_name(
            "dashboard_template.html"
        ).read_text(encoding="utf-8")
        accounts_source = (
            Path(self.module.__file__).parents[1]
            / "accounts"
            / "accounts_bo.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "Read only — all E*TRADE order actions are disabled",
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
        self.assertIn("fetch('/refresh'", template)
        self.assertIn('src="/api/positions"', template)
        self.assertIn("minlength=\"8\"", template)
        self.assertIn("slice(0, 64)", template)
        self.assertNotIn("slice(0, 4)", template)
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

    def test_positions_artifact_can_render_only_inside_same_origin_dashboard(self):
        handler = object.__new__(self.module.RefreshHandler)
        handler.path = "/api/positions"
        handler.headers = {"Authorization": "Basic ignored"}
        handler.rfile = _ExplodingBody()
        handler.wfile = io.BytesIO()
        handler.check_auth = lambda _header: True
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
        handler.check_auth = lambda _header: True
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
