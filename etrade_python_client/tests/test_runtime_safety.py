"""Focused fail-closed coverage for the live runtime boundary."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from accounts.accounts_bo import Accounts
from live_trading.runtime_safety import (
    ARM_SCHEMA_VERSION,
    LegacyExecutionDisabled,
    MAX_PRODUCTION_ARM_LIFETIME,
    OwnerOnlyRotatingFileHandler,
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
    build_runtime_safety_boundary,
    configure_owner_only_logger,
    issue_production_arm,
    main as runtime_safety_main,
    production_arm_signature,
    read_owner_only_json,
    secure_lock_file,
    secure_append_text,
    validate_dashboard_credentials,
    write_owner_only_json,
)


class _Response:
    status_code = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)
        self.request = SimpleNamespace(headers={})

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *args, **kwargs):
        return _Response(self.payload)


def _account(key="account-key", display="display-id", institution="BROKERAGE"):
    return {
        "accountIdKey": key,
        "accountId": display,
        "institutionType": institution,
        "accountStatus": "OPEN",
    }


def _arm_document(*, issued_at=None, expires_at=None, extra=None):
    issued_at = issued_at or datetime.now(timezone.utc)
    document = {
        "version": ARM_SCHEMA_VERSION,
        "environment": "production",
        "expected_account_id": "display-id",
        "expected_account_id_key": "account-key",
        "expected_institution_type": "BROKERAGE",
        "issued_at": issued_at.isoformat(),
        "expires_at": (expires_at or issued_at + timedelta(minutes=5)).isoformat(),
    }
    if extra:
        document.update(extra)
    return document


class RuntimeSafetyTests(unittest.TestCase):
    def test_missing_environment_never_defaults_to_production(self):
        with self.assertRaisesRegex(RuntimeSafetyError, "explicit"):
            build_runtime_safety_boundary(
                environment=None,
                legacy_sandbox=None,
                expected_account_id=None,
                expected_account_id_key=None,
                expected_institution_type=None,
            )

    def test_explicit_legacy_sandbox_is_the_only_identity_free_compatibility_path(self):
        boundary = build_runtime_safety_boundary(
            environment=None,
            legacy_sandbox=True,
            expected_account_id=None,
            expected_account_id_key=None,
            expected_institution_type=None,
        )
        self.assertTrue(boundary.use_sandbox)
        self.assertEqual(boundary.account_selection_kwargs, {"selected_account_id": 1})

    def test_production_requires_matching_unexpired_independent_arm_document(self):
        with tempfile.TemporaryDirectory() as directory:
            arm_path = Path(directory) / "production-arm.json"
            secret = "s" * 32
            document = _arm_document()
            document["signature"] = production_arm_signature(document, secret)
            arm_path.write_text(json.dumps(document), encoding="utf-8")
            os.chmod(arm_path, 0o600)

            boundary = build_runtime_safety_boundary(
                environment="production",
                legacy_sandbox=None,
                expected_account_id="display-id",
                expected_account_id_key="account-key",
                expected_institution_type="BROKERAGE",
                production_arm_file=arm_path,
                production_arm_secret=secret,
            )

        self.assertFalse(boundary.use_sandbox)
        self.assertEqual(boundary.account_selection_kwargs["selected_account_id"], None)
        boundary.verify_account(_account())
        with self.assertRaisesRegex(RuntimeSafetyError, "armed identity"):
            boundary.verify_account(_account(display="wrong"))

    def test_production_rejects_a_missing_or_weak_arm_proof(self):
        with self.assertRaisesRegex(RuntimeSafetyError, "arm"):
            build_runtime_safety_boundary(
                environment="production",
                legacy_sandbox=None,
                expected_account_id="display-id",
                expected_account_id_key="account-key",
                expected_institution_type="BROKERAGE",
                production_arm_file=None,
                production_arm_secret="s" * 32,
            )

    def test_production_arm_rejects_expired_future_overlong_and_extra_documents(self):
        secret = "s" * 32
        now = datetime.now(timezone.utc)
        cases = (
            (_arm_document(issued_at=now - timedelta(minutes=6), expires_at=now - timedelta(seconds=1)), "expired"),
            (_arm_document(issued_at=now + timedelta(seconds=61)), "future"),
            (
                _arm_document(
                    issued_at=now,
                    expires_at=now + MAX_PRODUCTION_ARM_LIFETIME + timedelta(seconds=1),
                ),
                "lifetime",
            ),
            (_arm_document(extra={"unexpected": "field"}), "schema"),
        )
        with tempfile.TemporaryDirectory() as directory:
            arm_path = Path(directory) / "production-arm.json"
            for document, expected_error in cases:
                if expected_error == "schema":
                    document.pop("unexpected")
                    document["signature"] = production_arm_signature(document, secret)
                    document["unexpected"] = "field"
                else:
                    document["signature"] = production_arm_signature(document, secret)
                arm_path.write_text(json.dumps(document), encoding="utf-8")
                os.chmod(arm_path, 0o600)
                with self.assertRaisesRegex(RuntimeSafetyError, expected_error):
                    build_runtime_safety_boundary(
                        environment="production",
                        legacy_sandbox=None,
                        expected_account_id="display-id",
                        expected_account_id_key="account-key",
                        expected_institution_type="BROKERAGE",
                        production_arm_file=arm_path,
                        production_arm_secret=secret,
                        now=now,
                    )

    def test_production_arm_expiry_is_rechecked_before_account_use(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            arm_path = Path(directory) / "production-arm.json"
            issue_production_arm(
                output=arm_path,
                expected_account_id="display-id",
                expected_account_id_key="account-key",
                expected_institution_type="BROKERAGE",
                ttl_seconds=1,
                secret="s" * 32,
                now=now,
            )
            boundary = build_runtime_safety_boundary(
                environment="production",
                legacy_sandbox=None,
                expected_account_id="display-id",
                expected_account_id_key="account-key",
                expected_institution_type="BROKERAGE",
                production_arm_file=arm_path,
                production_arm_secret="s" * 32,
                now=now,
            )
        with self.assertRaisesRegex(RuntimeSafetyError, "expired"):
            boundary.verify_account(_account(), now=now + timedelta(seconds=1))

    def test_arm_operator_command_creates_an_owner_only_exact_document(self):
        with tempfile.TemporaryDirectory() as directory:
            arm_path = Path(directory) / "production-arm.json"
            with mock.patch.dict(os.environ, {"ETRADE_PRODUCTION_ARMING_SECRET": "s" * 32}, clear=False):
                result = runtime_safety_main(
                    [
                        "issue-production-arm",
                        "--output", str(arm_path),
                        "--expected-account-id", "display-id",
                        "--expected-account-id-key", "account-key",
                        "--expected-institution-type", "BROKERAGE",
                        "--ttl-seconds", "300",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(os.stat(arm_path).st_mode & 0o777, 0o600)
            self.assertEqual(
                set(json.loads(arm_path.read_text(encoding="utf-8"))),
                {
                    "version", "environment", "expected_account_id", "expected_account_id_key",
                    "expected_institution_type", "issued_at", "expires_at", "signature",
                },
            )

    def test_arm_writer_rejects_an_unsafe_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            unsafe_parent = Path(directory) / "unsafe"
            unsafe_parent.mkdir()
            os.chmod(unsafe_parent, 0o777)
            with self.assertRaisesRegex(RuntimeSafetyError, "parent"):
                issue_production_arm(
                    output=unsafe_parent / "production-arm.json",
                    expected_account_id="display-id",
                    expected_account_id_key="account-key",
                    expected_institution_type="BROKERAGE",
                    ttl_seconds=300,
                    secret="s" * 32,
                )

    def test_exact_account_selection_rejects_index_and_identity_ambiguity(self):
        payload = {
            "AccountListResponse": {
                "Accounts": {"Account": [_account(), _account()]}
            }
        }
        accounts = Accounts(_Session(payload), "https://example.invalid", use_sandbox=False, consumer_key="key")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            accounts.account_list(
                None,
                expected_account_id_key="account-key",
                expected_account_id="display-id",
                expected_institution_type="BROKERAGE",
            )

        accounts = Accounts(
            _Session({"AccountListResponse": {"Accounts": {"Account": [_account()]}}}),
            "https://example.invalid",
            use_sandbox=False,
            consumer_key="key",
        )
        accounts.account_list(
            None,
            expected_account_id_key="account-key",
            expected_account_id="display-id",
            expected_institution_type="BROKERAGE",
        )
        self.assertEqual(accounts.account["accountIdKey"], "account-key")

    def test_live_trade_agent_preserves_exact_binding_through_refresh(self):
        from core_api import stock_trade_class as trade_module

        created_accounts = []

        class FakeAccounts:
            def __init__(self, *args, **kwargs):
                self.account = _account()
                self.calls = []
                created_accounts.append(self)

            def account_list(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return []

        order_factory = mock.Mock()
        with mock.patch.object(trade_module, "Accounts", FakeAccounts), mock.patch.object(
            trade_module, "Market", mock.Mock()
        ), mock.patch.object(trade_module, "Order", order_factory):
            boundary = build_runtime_safety_boundary(
                environment="sandbox",
                legacy_sandbox=None,
                expected_account_id="display-id",
                expected_account_id_key="account-key",
                expected_institution_type="BROKERAGE",
            )
            agent = trade_module.LiveTradeAgent(
                authenticated_session=object(),
                base_url="https://example.invalid",
                selected_account=None,
                use_sandbox=True,
                expected_account_id_key="account-key",
                expected_account_id="display-id",
                expected_institution_type="BROKERAGE",
                runtime_safety=boundary,
            )
            agent.refresh_session(object(), "https://example.invalid")

        self.assertEqual(len(created_accounts), 2)
        self.assertEqual(order_factory.call_count, 2)
        for call in order_factory.call_args_list:
            self.assertIs(call.kwargs["runtime_safety"], boundary)
        for account_client in created_accounts:
            self.assertEqual(
                account_client.calls,
                [
                    (
                        (None,),
                        {
                            "expected_account_id_key": "account-key",
                            "expected_account_id": "display-id",
                            "expected_institution_type": "BROKERAGE",
                        },
                    )
                ],
            )

    def test_authenticated_live_agent_requires_a_runtime_boundary(self):
        from core_api import stock_trade_class as trade_module

        with self.assertRaisesRegex(RuntimeError, "RuntimeSafetyBoundary"):
            trade_module.LiveTradeAgent(
                authenticated_session=object(),
                base_url="https://example.invalid",
            )

    def test_live_agent_rejects_arguments_conflicting_with_its_boundary(self):
        from core_api import stock_trade_class as trade_module

        boundary = build_runtime_safety_boundary(
            environment="sandbox",
            legacy_sandbox=None,
            expected_account_id=None,
            expected_account_id_key=None,
            expected_institution_type=None,
        )
        with self.assertRaisesRegex(RuntimeError, "conflict"):
            trade_module.LiveTradeAgent(
                authenticated_session=object(),
                base_url="https://example.invalid",
                use_sandbox=False,
                runtime_safety=boundary,
            )

    def test_live_clients_resolve_configless_environment_credentials(self):
        from market.market_bo import Market
        from order.order_bo import Order

        env = {"ETRADE_LIVE_CONSUMER_KEY": "env-only-consumer-key"}
        with mock.patch.dict(os.environ, env, clear=False):
            account_client = Accounts(object(), "https://example.invalid", use_sandbox=False)
            market_client = Market(object(), "https://example.invalid", use_sandbox=False)
            order_client = Order(object(), _account(), "https://example.invalid", use_sandbox=False)
        self.assertEqual(account_client.consumer_key, env["ETRADE_LIVE_CONSUMER_KEY"])
        self.assertEqual(market_client.consumer_key, env["ETRADE_LIVE_CONSUMER_KEY"])
        self.assertEqual(order_client.consumer_key, env["ETRADE_LIVE_CONSUMER_KEY"])

    def test_order_api_rejects_missing_boundary_before_broker_io(self):
        from order.order_bo import Order

        session = mock.Mock()
        order_client = Order(
            session,
            _account(),
            "https://example.invalid",
            use_sandbox=False,
            consumer_key="key",
        )
        with self.assertRaisesRegex(LegacyExecutionDisabled, "quarantined"):
            order_client.preview_order(
                {
                    "securityType": "EQ",
                    "client_order_id": "client-id",
                    "priceType": "LIMIT",
                    "orderTerm": "GOOD_FOR_DAY",
                    "limitPrice": 1.0,
                    "symbol": "SPY",
                    "orderAction": "BUY",
                    "quantity": 1,
                }
            )
        session.post.assert_not_called()

    def test_legacy_order_api_cannot_be_enabled_by_a_valid_runtime_boundary(self):
        from order.order_bo import Order

        boundary = mock.Mock(spec=RuntimeSafetyBoundary)
        boundary.use_sandbox = False
        session = mock.Mock()
        order_client = Order(
            session,
            _account(),
            "https://example.invalid",
            use_sandbox=False,
            consumer_key="key",
            runtime_safety=boundary,
        )
        boundary.verify_account.reset_mock()
        with self.assertRaisesRegex(LegacyExecutionDisabled, "quarantined"):
            order_client.preview_order(
                {
                    "securityType": "EQ",
                    "client_order_id": "client-id",
                    "priceType": "LIMIT",
                    "orderTerm": "GOOD_FOR_DAY",
                    "limitPrice": 1.0,
                    "symbol": "SPY",
                    "orderAction": "BUY",
                    "quantity": 1,
                }
            )
        boundary.verify_account.assert_not_called()
        session.post.assert_not_called()

    def test_legacy_interactive_order_client_is_quarantined(self):
        from order.order import Order

        with self.assertRaisesRegex(LegacyExecutionDisabled, "quarantined"):
            Order(object(), _account(), "https://example.invalid")

    def test_live_agent_refresh_resolves_configless_environment_credentials(self):
        from core_api.stock_trade_class import LiveTradeAgent

        boundary = build_runtime_safety_boundary(
            environment="sandbox",
            legacy_sandbox=None,
            expected_account_id="display-id",
            expected_account_id_key="account-key",
            expected_institution_type="BROKERAGE",
        )

        def select_account(client, *args, **kwargs):
            client.account = _account()
            return [client.account]

        with mock.patch.dict(os.environ, {"ETRADE_SANDBOX_CONSUMER_KEY": "sandbox-env-key"}, clear=False), mock.patch.object(
            Accounts, "account_list", select_account
        ):
            agent = LiveTradeAgent(
                authenticated_session=object(),
                base_url="https://example.invalid",
                runtime_safety=boundary,
            )
            agent.refresh_session(object(), "https://example.invalid")
        self.assertEqual(agent.market.consumer_key, "sandbox-env-key")
        self.assertEqual(agent.order.consumer_key, "sandbox-env-key")

    def test_dashboard_credentials_reject_defaults_and_accept_strong_values(self):
        with self.assertRaisesRegex(RuntimeSafetyError, "weak"):
            validate_dashboard_credentials(
                {"dashboard_user": "operator", "dashboard_pass": "short", "pin": "1234"}
            )
        validate_dashboard_credentials(
            {
                "dashboard_user": "operator",
                "dashboard_pass": "long-random-dashboard-password",
                "pin": "84927163",
            }
        )
        with self.assertRaisesRegex(RuntimeSafetyError, "weak"):
            validate_dashboard_credentials(
                {
                    "dashboard_user": "operator",
                    "dashboard_pass": "long-random-dashboard-password",
                    "pin": "9" * 65,
                }
            )

    def test_dashboard_settings_and_logs_do_not_leave_plaintext_credentials(self):
        from live_trading import etrade_cover_call_new as dashboard

        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            log_path = Path(directory) / "dashboard.log"
            settings = {
                "dashboard_user": "operator",
                "dashboard_pass": "long-random-dashboard-password",
                "pin": "84927163",
                "dashboard_auth_secret": "session-secret-value",
            }
            with mock.patch.object(dashboard, "LIVE_SETTINGS_FILE", str(settings_path)), mock.patch.object(
                dashboard, "DASHBOARD_LOG_FILE", str(log_path)
            ):
                self.assertTrue(dashboard.save_live_settings(settings))
                self.assertEqual(os.stat(settings_path).st_mode & 0o777, 0o600)
                dashboard.log_dashboard_request("/api/settings", settings)
            recorded = log_path.read_text(encoding="utf-8")
            self.assertEqual(os.stat(log_path).st_mode & 0o777, 0o600)
            self.assertNotIn(settings["dashboard_pass"], recorded)
            self.assertNotIn(settings["pin"], recorded)
            self.assertNotIn(settings["dashboard_auth_secret"], recorded)

    def test_order_audit_log_is_owner_only_and_csv_encoded(self):
        from live_trading import etrade_cover_call_new as dashboard

        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "orders.csv"
            with mock.patch.object(dashboard, "AUDIT_LOG_FILE", str(audit_path)):
                dashboard.log_order_execution(
                    {
                        "ticker": "SPY",
                        "sell_strike": 500,
                        "long_strike": 495,
                        "qty": 1,
                        "order_id": "broker-id",
                    },
                    'reason containing, comma',
                )
            self.assertEqual(os.stat(audit_path).st_mode & 0o777, 0o600)
            recorded = audit_path.read_text(encoding="utf-8")
            self.assertIn('"reason containing, comma"', recorded)

    def test_legacy_oauth_helper_uses_owner_only_environment_scoped_cache(self):
        from live_trading import etrade_check_option as legacy

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / ".etrade_oauth"
            token = {"access_token": "token", "access_token_secret": "secret"}
            with mock.patch.object(legacy, "ETRADE_OAUTH_FILE", str(cache_path)):
                self.assertTrue(legacy.save_etrade_oauth(token, use_sandbox=False))
                self.assertEqual(legacy.get_etrade_oauth(use_sandbox=False), token)
            self.assertEqual(os.stat(cache_path).st_mode & 0o777, 0o600)

    def test_quarantined_spread_import_uses_owner_only_log_and_cannot_authenticate(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os, stat; "
                    "from live_trading import etrade_put_credit_spread as legacy; "
                    "from live_trading.runtime_safety import RuntimeSafetyError; "
                    "assert stat.S_IMODE(os.stat('etrade_trader.log').st_mode) == 0o600; "
                    "\ntry:\n legacy.get_etrade_session(False)\n"
                    "except RuntimeSafetyError:\n pass\n"
                    "else:\n raise AssertionError('legacy authentication was not quarantined')",
                ],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_log_writer_and_rotating_handler_fail_closed_for_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "dashboard.log"
            secure_append_text(log_path, "safe\n")
            self.assertEqual(os.stat(log_path).st_mode & 0o777, 0o600)
            rotating_path = Path(directory) / "rotating.log"
            rotating_handler = OwnerOnlyRotatingFileHandler(rotating_path, maxBytes=100, backupCount=1)
            rotating_handler.close()
            self.assertEqual(os.stat(rotating_path).st_mode & 0o777, 0o600)
            os.chmod(log_path, 0o644)
            with self.assertRaisesRegex(RuntimeSafetyError, "unsafe"):
                secure_append_text(log_path, "blocked\n")

            target = Path(directory) / "target.log"
            target.write_text("target", encoding="utf-8")
            link_path = Path(directory) / "linked.log"
            link_path.symlink_to(target)
            with self.assertRaisesRegex(RuntimeSafetyError, "unsafe"):
                secure_append_text(link_path, "blocked\n")
            with self.assertRaisesRegex(RuntimeSafetyError, "unsafe"):
                OwnerOnlyRotatingFileHandler(link_path, maxBytes=100, backupCount=1)

            unsafe_parent = Path(directory) / "unsafe-parent"
            unsafe_parent.mkdir()
            os.chmod(unsafe_parent, 0o777)
            with self.assertRaisesRegex(RuntimeSafetyError, "parent"):
                secure_append_text(unsafe_parent / "dashboard.log", "blocked\n")

    def test_client_logger_redacts_auth_headers_and_order_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "client.log"
            logger = configure_owner_only_logger("runtime-safety-redaction-test", str(log_path))
            self.assertFalse(logger.propagate)
            logger.debug(
                "Request Header: %s",
                {
                    "Authorization": "OAuth secret",
                    "Cookie": "sid=secret",
                    "User-Agent": "private-client-version",
                },
            )
            logger.debug("Request payload: %s", "<Order><token>secret</token></Order>")
            logger.debug(
                "Response Body: %s",
                {"accountId": "private-account", "position": {"quantity": 42}},
            )
            logger.debug(json.dumps({"accountIdKey": "private-key", "orderId": 1234}))
            logger.debug("Request url: %s", "https://example.invalid/accounts/private-key/orders")
            logger.error("Failed broker response: %s", "plain private broker body")
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            recorded = log_path.read_text(encoding="utf-8")
            self.assertNotIn("OAuth secret", recorded)
            self.assertNotIn("sid=secret", recorded)
            self.assertNotIn("private-client-version", recorded)
            self.assertNotIn("<Order>", recorded)
            self.assertNotIn("private-account", recorded)
            self.assertNotIn("private-key", recorded)
            self.assertNotIn("quantity", recorded)
            self.assertNotIn("plain private broker body", recorded)
            self.assertIn("sha256", recorded)

    def test_sensitive_json_rewrite_rejects_links_and_unsafe_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / ".etrade_oauth"
            write_owner_only_json(cache_path, {"access_token": "long-token", "padding": "x" * 1000})
            write_owner_only_json(cache_path, {"access_token": "short-token"})
            self.assertEqual(os.stat(cache_path).st_mode & 0o777, 0o600)
            self.assertEqual(read_owner_only_json(cache_path, label="OAuth cache"), {"access_token": "short-token"})
            self.assertNotIn("padding", cache_path.read_text(encoding="utf-8"))
            os.chmod(cache_path, 0o644)
            with self.assertRaisesRegex(RuntimeSafetyError, "owner-only"):
                read_owner_only_json(cache_path, label="OAuth cache")
            link_path = Path(directory) / "linked-cache"
            link_path.symlink_to(cache_path)
            with self.assertRaisesRegex(RuntimeSafetyError, "invalid|unavailable"):
                read_owner_only_json(link_path, label="OAuth cache")

    def test_runtime_lock_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock"
            handle = secure_lock_file(lock_path)
            handle.close()
            self.assertEqual(os.stat(lock_path).st_mode & 0o777, 0o600)
            os.chmod(lock_path, 0o644)
            with self.assertRaisesRegex(RuntimeSafetyError, "unsafe"):
                secure_lock_file(lock_path)
            link_path = Path(directory) / "linked-lock"
            link_path.symlink_to(lock_path)
            with self.assertRaisesRegex(RuntimeSafetyError, "unsafe"):
                secure_lock_file(link_path)

    def test_runtime_lock_excludes_a_second_process(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock"
            handle = secure_lock_file(lock_path)
            try:
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(repository)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from live_trading.runtime_safety import secure_lock_file; "
                        f"secure_lock_file({str(lock_path)!r})",
                    ],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                )
            finally:
                handle.close()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already running", result.stderr)

    def test_clean_cwd_import_creates_a_protected_client_log(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                "-c",
                "import live_trading.etrade_cover_call_new; import os, stat; "
                "assert stat.S_IMODE(os.stat('python_client.log').st_mode) == 0o600",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository)
            result = subprocess.run(command, cwd=directory, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_settings_reject_an_unsafe_parent_directory(self):
        from live_trading import etrade_cover_call_new as dashboard

        with tempfile.TemporaryDirectory() as directory:
            unsafe_parent = Path(directory) / "unsafe"
            unsafe_parent.mkdir()
            os.chmod(unsafe_parent, 0o777)
            settings = {
                "dashboard_user": "operator",
                "dashboard_pass": "long-random-dashboard-password",
                "pin": "84927163",
            }
            with mock.patch.object(dashboard, "LIVE_SETTINGS_FILE", str(unsafe_parent / "settings.json")):
                self.assertFalse(dashboard.save_live_settings(settings))

    def test_safe_dashboard_responses_include_browser_security_headers(self):
        from live_trading import etrade_cover_call_new as dashboard

        handler = object.__new__(dashboard.RefreshHandler)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = io.BytesIO()
        handler._send_safe_response(200, {"ok": True})
        headers = {call.args[0]: call.args[1] for call in handler.send_header.call_args_list}
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")

    def test_dashboard_server_is_loopback_only(self):
        from live_trading import etrade_cover_call_new as dashboard

        fake_server = mock.Mock()
        with mock.patch.object(dashboard, "ThreadingHTTPServer", return_value=fake_server) as server:
            dashboard.start_refresh_server(port=8765)
        server.assert_called_once_with(("127.0.0.1", 8765), dashboard.RefreshHandler)
        with self.assertRaisesRegex(RuntimeSafetyError, "loopback"):
            dashboard.start_refresh_server(host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
