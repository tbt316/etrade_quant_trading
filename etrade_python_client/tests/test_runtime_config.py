"""Contract tests for strict, side-effect-free live runtime configuration."""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path

from live_trading.runtime_config import (
    RuntimeConfigError,
    load_runtime_config,
    main as runtime_config_main,
    resolve_dashboard_secrets,
    resolve_runtime_secrets,
    validate_runtime_directories,
)


MAX_SIGNED_SQLITE_INTEGER = (1 << 63) - 1
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "mode",
    "runtime_root",
    "strategy",
    "data",
    "model",
    "execution",
    "risk",
}


def _account(
    *,
    account_id: str = "display-id",
    account_id_key: str = "opaque-account-key",
    institution_type: str = "BROKERAGE",
) -> dict[str, str]:
    return {
        "account_id": account_id,
        "account_id_key": account_id_key,
        "institution_type": institution_type,
    }


def _document(
    *,
    mode: str = "paper",
    runtime_root: str = "runtime",
) -> dict[str, object]:
    broker_backed = mode in {"sandbox", "shadow", "live"}
    return {
        "schema_version": 1,
        "mode": mode,
        "runtime_root": runtime_root,
        "strategy": {
            "enabled": False,
            "strategy_id": "disabled",
            "symbols": [],
        },
        "data": {
            "require_complete_snapshots": True,
            "max_snapshot_age_seconds": 300,
        },
        "model": {
            "enabled": False,
            "required_for_entry": False,
            "max_signal_age_seconds": 86_400,
        },
        "execution": {
            "selected_account_id_key": (
                "opaque-account-key" if broker_backed else None
            ),
            "account_allowlist": [_account()] if broker_backed else [],
            "broker_mutations_enabled": False,
        },
        "risk": {
            "max_order_contracts": 0,
            "max_order_loss_cents": 0,
            "max_account_open_risk_cents": 0,
            "max_daily_loss_cents": 0,
            "max_quote_age_seconds": 30,
        },
    }


def _enabled_document(*, runtime_root: str = "runtime") -> dict[str, object]:
    document = _document(runtime_root=runtime_root)
    document["strategy"] = {
        "enabled": True,
        "strategy_id": "covered-call.v1",
        "symbols": ["SPY", "QQQ"],
    }
    document["risk"] = {
        "max_order_contracts": 10,
        "max_order_loss_cents": 50_000,
        "max_account_open_risk_cents": 100_000,
        "max_daily_loss_cents": 75_000,
        "max_quote_age_seconds": 30,
    }
    return document


def _write_payload(
    directory: Path,
    payload: bytes,
    *,
    name: str = "runtime-config.json",
    mode: int = 0o600,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    os.chmod(path, mode)
    return path


def _write_config(
    directory: Path,
    document: dict[str, object] | None = None,
    *,
    name: str = "runtime-config.json",
    mode: int = 0o600,
) -> Path:
    payload = json.dumps(
        _document() if document is None else document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _write_payload(directory, payload, name=name, mode=mode)


def _set_nested(
    document: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    target: object = document
    for part in path[:-1]:
        assert isinstance(target, dict)
        target = target[part]
    assert isinstance(target, dict)
    target[path[-1]] = value


def _create_runtime_directories(config) -> None:
    for directory in config.paths.directories:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)


def _secret_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "etrade_consumer_key": "consumer-key-value",
        "etrade_consumer_secret": "consumer-secret-value",
        "etrade_username": "broker-operator",
        "etrade_password": "broker-password-value",
        "dashboard_username": "dashboard-operator",
        "dashboard_password": "correct-horse-battery-staple",
        "dashboard_pin": "A9~strong",
        "dashboard_session_secret": "session-secret-value-with-at-least-32-characters",
    }
    document.update(overrides)
    return document


def _environment_secrets(
    *,
    broker_prefix: str | None = None,
) -> dict[str, str]:
    environment = {
        "ETRADE_DASHBOARD_USER": "environment-operator",
        "ETRADE_DASHBOARD_PASSWORD": "environment-password-value",
        "ETRADE_DASHBOARD_PIN": "B7!runtime",
        "ETRADE_DASHBOARD_SESSION_SECRET": (
            "environment-session-secret-with-at-least-32-characters"
        ),
    }
    if broker_prefix is not None:
        environment.update(
            {
                f"ETRADE_{broker_prefix}_CONSUMER_KEY": (
                    "environment-consumer-key"
                ),
                f"ETRADE_{broker_prefix}_CONSUMER_SECRET": (
                    "environment-consumer-secret"
                ),
                "ETRADE_USER": "environment-broker-user",
                "ETRADE_PASS": "environment-broker-password",
            }
        )
    return environment


def _write_secret_file(config, document=None, *, mode: int = 0o600) -> Path:
    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.paths.root, 0o700)
    os.chmod(config.paths.state_dir, 0o700)
    return _write_payload(
        config.paths.state_dir,
        json.dumps(
            _secret_document() if document is None else document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        name=config.paths.secrets_file.name,
        mode=mode,
    )


class RuntimeConfigSchemaTests(unittest.TestCase):
    def test_all_modes_have_explicit_broker_and_account_semantics(self):
        expected_environments = {
            "paper": None,
            "sandbox": "sandbox",
            "shadow": "production",
            "live": "production",
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for mode, expected_environment in expected_environments.items():
                with self.subTest(mode=mode):
                    path = _write_config(
                        base,
                        _document(mode=mode, runtime_root=f"runtime-{mode}"),
                        name=f"{mode}.json",
                    )
                    config = load_runtime_config(path)
                    self.assertEqual(config.mode, mode)
                    self.assertEqual(
                        config.broker_environment,
                        expected_environment,
                    )
                    self.assertFalse(config.broker_mutations_authorized)
                    self.assertFalse(config.starts_armed)
                    self.assertFalse(
                        config.execution.broker_mutations_enabled
                    )
                    if mode == "paper":
                        self.assertIsNone(config.selected_account)
                        self.assertEqual(
                            config.execution.account_allowlist,
                            (),
                        )
                    else:
                        self.assertEqual(
                            config.selected_account,
                            config.execution.account_allowlist[0],
                        )
                        self.assertEqual(
                            config.selected_account.account_id_key,
                            "opaque-account-key",
                        )

    def test_schema_is_exact_at_every_object_boundary(self):
        invalid_documents: list[tuple[str, dict[str, object]]] = []

        missing_top = _document()
        missing_top.pop("risk")
        invalid_documents.append(("missing top-level field", missing_top))

        unknown_top = _document()
        unknown_top["armed"] = True
        invalid_documents.append(("unknown top-level field", unknown_top))

        section_fields = {
            "strategy": "strategy_id",
            "data": "max_snapshot_age_seconds",
            "model": "max_signal_age_seconds",
            "execution": "broker_mutations_enabled",
            "risk": "max_quote_age_seconds",
        }
        for section, field in section_fields.items():
            missing = _document()
            section_value = missing[section]
            assert isinstance(section_value, dict)
            section_value.pop(field)
            invalid_documents.append((f"{section} missing field", missing))

            unknown = _document()
            section_value = unknown[section]
            assert isinstance(section_value, dict)
            section_value["unexpected"] = "value"
            invalid_documents.append((f"{section} unknown field", unknown))

        for schema_version in (0, 2, True, "1"):
            document = _document()
            document["schema_version"] = schema_version
            invalid_documents.append(
                (f"schema version {schema_version!r}", document)
            )

        for mode in ("", "Paper", "production", "sandbox "):
            document = _document()
            document["mode"] = mode
            invalid_documents.append((f"mode {mode!r}", document))

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, (label, document) in enumerate(invalid_documents):
                with self.subTest(case=label):
                    path = _write_config(
                        base,
                        document,
                        name=f"invalid-{index}.json",
                    )
                    with self.assertRaises(RuntimeConfigError):
                        load_runtime_config(path)

            primitive = _write_payload(
                base,
                b"[]",
                name="primitive.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "object"):
                load_runtime_config(primitive)

    def test_broker_accounts_are_exact_unique_and_fail_closed(self):
        invalid_documents: list[tuple[str, dict[str, object]]] = []

        paper_account = _document()
        paper_execution = paper_account["execution"]
        assert isinstance(paper_execution, dict)
        paper_execution["selected_account_id_key"] = "opaque-account-key"
        paper_execution["account_allowlist"] = [_account()]
        invalid_documents.append(("paper account", paper_account))

        for label, selected, accounts in (
            ("missing selected key", None, [_account()]),
            ("empty allowlist", "opaque-account-key", []),
            ("selected key mismatch", "different-key", [_account()]),
            (
                "duplicate key",
                "opaque-account-key",
                [_account(), _account(account_id="second-display-id")],
            ),
        ):
            document = _document(mode="sandbox")
            execution = document["execution"]
            assert isinstance(execution, dict)
            execution["selected_account_id_key"] = selected
            execution["account_allowlist"] = accounts
            invalid_documents.append((label, document))

        malformed_account = _document(mode="shadow")
        malformed_execution = malformed_account["execution"]
        assert isinstance(malformed_execution, dict)
        malformed_execution["account_allowlist"] = [
            _account(institution_type="brokerage")
        ]
        invalid_documents.append(("malformed identity", malformed_account))

        extra_account_field = _document(mode="live")
        extra_execution = extra_account_field["execution"]
        assert isinstance(extra_execution, dict)
        account = _account()
        account["nickname"] = "primary"
        extra_execution["account_allowlist"] = [account]
        invalid_documents.append(("extra account identity field", extra_account_field))

        mutations = _document(mode="sandbox")
        mutations_execution = mutations["execution"]
        assert isinstance(mutations_execution, dict)
        mutations_execution["broker_mutations_enabled"] = True
        invalid_documents.append(("configuration mutation authority", mutations))

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, (label, document) in enumerate(invalid_documents):
                with self.subTest(case=label):
                    path = _write_config(
                        base,
                        document,
                        name=f"invalid-account-{index}.json",
                    )
                    with self.assertRaises(RuntimeConfigError):
                        load_runtime_config(path)

            valid = load_runtime_config(
                _write_config(
                    base,
                    _document(mode="shadow"),
                    name="valid-shadow.json",
                )
            )
            selected = valid.verify_broker_account(
                {
                    "accountId": "display-id",
                    "accountIdKey": "opaque-account-key",
                    "institutionType": "BROKERAGE",
                }
            )
            self.assertEqual(selected.account_id, "display-id")
            for field in ("accountId", "accountIdKey", "institutionType"):
                actual = {
                    "accountId": "display-id",
                    "accountIdKey": "opaque-account-key",
                    "institutionType": "BROKERAGE",
                }
                actual[field] = "WRONG"
                with self.subTest(mismatched_field=field):
                    with self.assertRaisesRegex(
                        RuntimeConfigError,
                        "does not match",
                    ):
                        valid.verify_broker_account(actual)
            self.assertNotIn("display-id", repr(valid))
            self.assertNotIn("opaque-account-key", repr(valid))

    def test_numeric_ranges_accept_boundaries_and_reject_out_of_range_values(self):
        boundaries = _enabled_document()
        data = boundaries["data"]
        model = boundaries["model"]
        risk = boundaries["risk"]
        assert isinstance(data, dict)
        assert isinstance(model, dict)
        assert isinstance(risk, dict)
        data["max_snapshot_age_seconds"] = 86_400
        model["max_signal_age_seconds"] = 86_400
        risk.update(
            {
                "max_order_contracts": 1_000_000,
                "max_order_loss_cents": MAX_SIGNED_SQLITE_INTEGER,
                "max_account_open_risk_cents": MAX_SIGNED_SQLITE_INTEGER,
                "max_daily_loss_cents": MAX_SIGNED_SQLITE_INTEGER,
                "max_quote_age_seconds": 300,
            }
        )

        invalid_values = (
            (("data", "max_snapshot_age_seconds"), 0),
            (("data", "max_snapshot_age_seconds"), 86_401),
            (("model", "max_signal_age_seconds"), 0),
            (("model", "max_signal_age_seconds"), 86_401),
            (("risk", "max_order_contracts"), -1),
            (("risk", "max_order_contracts"), 1_000_001),
            (("risk", "max_order_loss_cents"), -1),
            (
                ("risk", "max_order_loss_cents"),
                MAX_SIGNED_SQLITE_INTEGER + 1,
            ),
            (("risk", "max_account_open_risk_cents"), -1),
            (
                ("risk", "max_account_open_risk_cents"),
                MAX_SIGNED_SQLITE_INTEGER + 1,
            ),
            (("risk", "max_daily_loss_cents"), -1),
            (
                ("risk", "max_daily_loss_cents"),
                MAX_SIGNED_SQLITE_INTEGER + 1,
            ),
            (("risk", "max_quote_age_seconds"), 0),
            (("risk", "max_quote_age_seconds"), 301),
            (("risk", "max_quote_age_seconds"), True),
            (("risk", "max_order_loss_cents"), 1.5),
        )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            valid_path = _write_config(
                base,
                boundaries,
                name="boundaries.json",
            )
            config = load_runtime_config(valid_path)
            self.assertEqual(
                config.risk.max_daily_loss_cents,
                MAX_SIGNED_SQLITE_INTEGER,
            )
            self.assertEqual(config.risk.max_quote_age_seconds, 300)

            for index, (field_path, value) in enumerate(invalid_values):
                with self.subTest(field=".".join(field_path), value=value):
                    document = _document()
                    _set_nested(document, field_path, value)
                    path = _write_config(
                        base,
                        document,
                        name=f"invalid-range-{index}.json",
                    )
                    with self.assertRaises(RuntimeConfigError):
                        load_runtime_config(path)

    def test_cross_field_strategy_data_model_and_risk_rules_are_strict(self):
        invalid_documents: list[tuple[str, dict[str, object]]] = []

        incomplete = _document()
        data = incomplete["data"]
        assert isinstance(data, dict)
        data["require_complete_snapshots"] = False
        invalid_documents.append(("incomplete snapshots", incomplete))

        model_required = _document()
        model = model_required["model"]
        assert isinstance(model, dict)
        model["required_for_entry"] = True
        invalid_documents.append(("disabled required model", model_required))

        duplicate_symbols = _enabled_document()
        strategy = duplicate_symbols["strategy"]
        assert isinstance(strategy, dict)
        strategy["symbols"] = ["SPY", "SPY"]
        invalid_documents.append(("duplicate symbols", duplicate_symbols))

        malformed_symbol = _enabled_document()
        strategy = malformed_symbol["strategy"]
        assert isinstance(strategy, dict)
        strategy["symbols"] = ["spy"]
        invalid_documents.append(("malformed symbol", malformed_symbol))

        inconsistent_disabled = _document()
        strategy = inconsistent_disabled["strategy"]
        assert isinstance(strategy, dict)
        strategy["symbols"] = ["SPY"]
        invalid_documents.append(("disabled strategy with symbols", inconsistent_disabled))

        order_exceeds_account = _enabled_document()
        risk = order_exceeds_account["risk"]
        assert isinstance(risk, dict)
        risk["max_order_loss_cents"] = 100_001
        risk["max_account_open_risk_cents"] = 100_000
        invalid_documents.append(("order exceeds account capacity", order_exceeds_account))

        for field in (
            "max_order_contracts",
            "max_order_loss_cents",
            "max_account_open_risk_cents",
            "max_daily_loss_cents",
        ):
            zero_risk = _enabled_document()
            risk = zero_risk["risk"]
            assert isinstance(risk, dict)
            risk[field] = 0
            invalid_documents.append((f"enabled strategy zero {field}", zero_risk))

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, (label, document) in enumerate(invalid_documents):
                with self.subTest(case=label):
                    path = _write_config(
                        base,
                        document,
                        name=f"cross-field-{index}.json",
                    )
                    with self.assertRaises(RuntimeConfigError):
                        load_runtime_config(path)

    def test_duplicate_keys_and_non_finite_json_numbers_are_rejected(self):
        canonical = json.dumps(_document(), separators=(",", ":"))
        duplicate = canonical.replace(
            '"mode":"paper"',
            '"mode":"paper","mode":"live"',
            1,
        )
        non_finite = {
            "NaN": canonical.replace(
                '"max_quote_age_seconds":30',
                '"max_quote_age_seconds":NaN',
                1,
            ),
            "Infinity": canonical.replace(
                '"max_quote_age_seconds":30',
                '"max_quote_age_seconds":Infinity',
                1,
            ),
            "-Infinity": canonical.replace(
                '"max_quote_age_seconds":30',
                '"max_quote_age_seconds":-Infinity',
                1,
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            duplicate_path = _write_payload(
                base,
                duplicate.encode("utf-8"),
                name="duplicate.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "duplicate"):
                load_runtime_config(duplicate_path)

            for label, payload in non_finite.items():
                with self.subTest(number=label):
                    path = _write_payload(
                        base,
                        payload.encode("utf-8"),
                        name=f"{label}.json",
                    )
                    with self.assertRaisesRegex(RuntimeConfigError, "non-finite"):
                        load_runtime_config(path)

            deeply_nested = _write_payload(
                base,
                ("[" * 2_000 + "]" * 2_000).encode("ascii"),
                name="deeply-nested.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "valid UTF-8 JSON"):
                load_runtime_config(deeply_nested)


class RuntimePathAndFilesystemTests(unittest.TestCase):
    def test_relative_root_is_config_relative_and_loading_creates_no_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config_directory = base / "configuration"
            working_directory = base / "unrelated-working-directory"
            working_directory.mkdir()
            path = _write_config(
                config_directory,
                _document(runtime_root="private-runtime"),
            )
            payload_before = path.read_bytes()
            metadata_before = path.stat()
            expected_hash = hashlib.sha256(payload_before).hexdigest()
            canonical_config_directory = Path(
                os.path.realpath(config_directory)
            )
            runtime_root = canonical_config_directory / "private-runtime"
            original_working_directory = Path.cwd()
            try:
                os.chdir(working_directory)
                config = load_runtime_config(path)
            finally:
                os.chdir(original_working_directory)

            self.assertEqual(config.paths.root, runtime_root)
            self.assertEqual(
                config.source_path,
                canonical_config_directory / path.name,
            )
            self.assertEqual(config.source_sha256, expected_hash)
            self.assertFalse(runtime_root.exists())
            self.assertEqual(list(working_directory.iterdir()), [])
            self.assertEqual(path.read_bytes(), payload_before)
            metadata_after = path.stat()
            self.assertEqual(metadata_after.st_mode, metadata_before.st_mode)
            self.assertEqual(metadata_after.st_mtime_ns, metadata_before.st_mtime_ns)

            expected_paths = {
                "state_dir": runtime_root / "state",
                "cache_dir": runtime_root / "cache",
                "logs_dir": runtime_root / "logs",
                "artifacts_dir": runtime_root / "artifacts",
                "execution_dir": runtime_root / "execution",
                "data_dir": runtime_root / "data",
                "model_dir": runtime_root / "model",
                "secrets_file": runtime_root / "state" / "secrets.json",
                "oauth_file": runtime_root / "state" / "oauth.json",
                "production_arm_file": (
                    runtime_root / "state" / "production-arm.json"
                ),
                "runtime_lock_file": runtime_root / "state" / "runtime.lock",
                "live_settings_file": (
                    runtime_root / "state" / "legacy-settings.json"
                ),
                "trade_status_file": (
                    runtime_root / "state" / "trade-status.json"
                ),
                "manual_trade_status_file": (
                    runtime_root / "state" / "manual-trade-status.json"
                ),
                "spy_tracker_file": (
                    runtime_root / "state" / "spy-tracking.json"
                ),
                "spy_gains_cache_file": (
                    runtime_root / "cache" / "spy-gains.json"
                ),
                "order_intent_ledger_file": (
                    runtime_root / "execution" / "order-intents.sqlite3"
                ),
                "client_log_file": runtime_root / "logs" / "client.log",
                "dashboard_log_file": runtime_root / "logs" / "dashboard.log",
                "order_audit_log_file": (
                    runtime_root / "logs" / "order-audit.csv"
                ),
                "positions_artifact_file": (
                    runtime_root / "artifacts" / "positions.html"
                ),
                "regime_shadow_file": (
                    runtime_root / "model" / "regime-v2-shadow.json"
                ),
                "spy_vix_price_cache_file": (
                    runtime_root / "cache" / "spy-vix-prices.json"
                ),
            }
            for attribute, expected in expected_paths.items():
                with self.subTest(path=attribute):
                    self.assertEqual(getattr(config.paths, attribute), expected)

            with self.assertRaises(FrozenInstanceError):
                config.mode = "live"
            with self.assertRaises(FrozenInstanceError):
                config.paths.root = base / "different"
            self.assertIsInstance(config.strategy.symbols, tuple)
            self.assertIsInstance(config.execution.account_allowlist, tuple)

    def test_absolute_root_is_preserved_and_unsafe_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            absolute_root = base / "absolute-runtime"
            absolute_path = _write_config(
                base,
                _document(runtime_root=str(absolute_root)),
                name="absolute.json",
            )
            self.assertEqual(
                load_runtime_config(absolute_path).paths.root,
                Path(os.path.realpath(absolute_root)),
            )
            self.assertFalse(absolute_root.exists())

            unsafe_roots = (
                "",
                "runtime/../elsewhere",
                "/",
                "runtime\\state",
                "runtime\x00state",
            )
            for index, root in enumerate(unsafe_roots):
                with self.subTest(runtime_root=root):
                    path = _write_config(
                        base,
                        _document(runtime_root=root),
                        name=f"unsafe-root-{index}.json",
                    )
                    with self.assertRaises(RuntimeConfigError):
                        load_runtime_config(path)

            shared_temp = Path("/tmp").resolve()
            if shared_temp.is_dir():
                shared_mode = stat.S_IMODE(shared_temp.stat().st_mode)
                if shared_mode & stat.S_ISVTX and shared_mode & 0o022:
                    shared_path = _write_config(
                        base,
                        _document(runtime_root=str(shared_temp)),
                        name="shared-temp-root.json",
                    )
                    with self.assertRaisesRegex(
                        RuntimeConfigError,
                        "group- or world-writable",
                    ):
                        load_runtime_config(shared_path)
                    unprotected = (
                        shared_temp
                        / f"etrade-runtime-unprotected-{os.getpid()}"
                    )
                    self.assertFalse(unprotected.exists())
                    unprotected_path = _write_config(
                        base,
                        _document(runtime_root=str(unprotected)),
                        name="shared-temp-child.json",
                    )
                    with self.assertRaisesRegex(
                        RuntimeConfigError,
                        "private directory",
                    ):
                        load_runtime_config(unprotected_path)

            surrogate = _document()
            surrogate["runtime_root"] = "\ud800"
            surrogate_path = _write_config(
                base,
                surrogate,
                name="surrogate-root.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "Unicode"):
                load_runtime_config(surrogate_path)

            with self.assertRaisesRegex(RuntimeConfigError, "path is invalid"):
                load_runtime_config(
                    "~codex-user-that-must-not-exist/runtime.json"
                )

    def test_runtime_root_rejects_a_single_dot_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = _write_config(
                base,
                _document(runtime_root="."),
            )
            with self.assertRaises(RuntimeConfigError):
                load_runtime_config(path)

    def test_runtime_root_rejects_an_embedded_dot_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = _write_config(
                base,
                _document(runtime_root="runtime/./nested"),
            )
            with self.assertRaises(RuntimeConfigError):
                load_runtime_config(path)

    def test_existing_runtime_components_reject_unsafe_directories_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)

            world_writable = base / "world-writable"
            world_writable.mkdir()
            os.chmod(world_writable, 0o777)
            world_writable_config = _write_config(
                base,
                _document(runtime_root="world-writable/runtime"),
                name="world-writable.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "world-writable"):
                load_runtime_config(world_writable_config)

            group_writable = base / "group-writable"
            group_writable.mkdir()
            os.chmod(group_writable, 0o770)
            group_writable_config = _write_config(
                base,
                _document(runtime_root="group-writable/runtime"),
                name="group-writable.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "group-"):
                load_runtime_config(group_writable_config)

            target = base / "symlink-target"
            target.mkdir()
            os.chmod(target, 0o700)
            link = base / "runtime-link"
            link.symlink_to(target, target_is_directory=True)
            symlink_config = _write_config(
                base,
                _document(runtime_root="runtime-link/runtime"),
                name="symlink-root.json",
            )
            with self.assertRaisesRegex(
                RuntimeConfigError,
                "symlink|symbolic link",
            ):
                load_runtime_config(symlink_config)

            component = base / "regular-component"
            component.write_text("not a directory", encoding="utf-8")
            non_directory_config = _write_config(
                base,
                _document(runtime_root="regular-component/runtime"),
                name="non-directory-root.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "non-directory"):
                load_runtime_config(non_directory_config)

    def test_configuration_file_must_be_safe_regular_and_non_symbolic(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)

            readable = _write_config(
                base,
                name="readable.json",
                mode=0o644,
            )
            self.assertEqual(load_runtime_config(readable).schema_version, 1)

            for index, mode in enumerate((0o620, 0o602, 0o666)):
                with self.subTest(mode=oct(mode)):
                    path = _write_config(
                        base,
                        name=f"writable-{index}.json",
                        mode=mode,
                    )
                    with self.assertRaisesRegex(RuntimeConfigError, "safe"):
                        load_runtime_config(path)

            target = _write_config(base, name="target.json")
            link = base / "linked.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeConfigError, "regular file"):
                load_runtime_config(link)

            directory = base / "directory.json"
            directory.mkdir()
            with self.assertRaisesRegex(RuntimeConfigError, "regular file"):
                load_runtime_config(directory)

            fifo = base / "fifo.json"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(RuntimeConfigError, "regular file"):
                load_runtime_config(fifo)

            oversized = _write_payload(
                base,
                b" " * (64 * 1024 + 1),
                name="oversized.json",
            )
            with self.assertRaisesRegex(RuntimeConfigError, "regular file|large"):
                load_runtime_config(oversized)

    def test_runtime_directory_validation_is_exact_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = _write_config(base)
            config = load_runtime_config(path)
            self.assertFalse(config.paths.root.exists())
            with self.assertRaisesRegex(RuntimeConfigError, "unavailable"):
                validate_runtime_directories(config.paths)
            self.assertFalse(config.paths.root.exists())

            _create_runtime_directories(config)
            before = {
                directory: (
                    directory.stat().st_mode,
                    directory.stat().st_mtime_ns,
                )
                for directory in config.paths.directories
            }
            validate_runtime_directories(config.paths)
            after = {
                directory: (
                    directory.stat().st_mode,
                    directory.stat().st_mtime_ns,
                )
                for directory in config.paths.directories
            }
            self.assertEqual(after, before)

        invalid_kinds = ("missing", "mode", "file", "symlink")
        for kind in invalid_kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                config = load_runtime_config(_write_config(base))
                _create_runtime_directories(config)
                target = config.paths.model_dir
                if kind == "missing":
                    target.rmdir()
                elif kind == "mode":
                    os.chmod(target, 0o750)
                elif kind == "file":
                    target.rmdir()
                    target.write_text("not a directory", encoding="utf-8")
                else:
                    target.rmdir()
                    target.symlink_to(
                        config.paths.data_dir,
                        target_is_directory=True,
                    )
                with self.assertRaisesRegex(
                    RuntimeConfigError,
                    "unavailable|owner-only",
                ):
                    validate_runtime_directories(config.paths)


class RuntimeSecretTests(unittest.TestCase):
    def test_dashboard_secret_resolver_is_environment_only_and_least_privilege(
        self,
    ):
        accessed: list[str] = []

        class RecordingEnvironment(dict[str, str]):
            def get(self, key, default=None):
                accessed.append(key)
                if key.startswith("ETRADE_LIVE_") or key in {
                    "ETRADE_USER",
                    "ETRADE_PASS",
                    "ETRADE_DASHBOARD_PIN",
                }:
                    raise AssertionError(f"unexpected secret read: {key}")
                return super().get(key, default)

        environment = RecordingEnvironment(_environment_secrets())
        secrets = resolve_dashboard_secrets(environ=environment)

        self.assertEqual(secrets.username, "environment-operator")
        self.assertEqual(
            secrets.password,
            "environment-password-value",
        )
        self.assertEqual(secrets.source, "environment")
        self.assertEqual(
            accessed,
            [
                "ETRADE_DASHBOARD_USER",
                "ETRADE_DASHBOARD_PASSWORD",
                "ETRADE_DASHBOARD_SESSION_SECRET",
            ],
        )
        for value in (
            secrets.username,
            secrets.password,
            secrets.session_secret,
        ):
            self.assertNotIn(value, repr(secrets))
        with self.assertRaises(FrozenInstanceError):
            secrets.password = "replacement-password"

    def test_dashboard_secret_resolver_rejects_missing_or_weak_values(self):
        invalid = (
            {},
            {
                "ETRADE_DASHBOARD_USER": "admin",
                "ETRADE_DASHBOARD_PASSWORD": (
                    "environment-password-value"
                ),
                "ETRADE_DASHBOARD_SESSION_SECRET": (
                    "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
                ),
            },
            {
                "ETRADE_DASHBOARD_USER": "environment-operator",
                "ETRADE_DASHBOARD_PASSWORD": "short",
                "ETRADE_DASHBOARD_SESSION_SECRET": (
                    "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
                ),
            },
            {
                "ETRADE_DASHBOARD_USER": "environment-operator",
                "ETRADE_DASHBOARD_PASSWORD": (
                    "environment-password-value"
                ),
                "ETRADE_DASHBOARD_SESSION_SECRET": "s" * 64,
            },
        )
        for environment in invalid:
            with self.subTest(environment=environment):
                with self.assertRaises(RuntimeConfigError):
                    resolve_dashboard_secrets(environ=environment)

    def test_complete_environment_is_authoritative_and_repr_is_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_runtime_config(
                _write_config(base, _document(mode="sandbox"))
            )
            environment = _environment_secrets(broker_prefix="SANDBOX")
            environment["ETRADE_PRODUCTION_ARMING_SECRET"] = (
                "independent-production-arm-secret"
            )
            self.assertFalse(config.paths.secrets_file.exists())

            secrets = resolve_runtime_secrets(
                config,
                environ=environment,
            )

            self.assertEqual(secrets.source, "environment")
            self.assertEqual(
                secrets.etrade_consumer_key,
                environment["ETRADE_SANDBOX_CONSUMER_KEY"],
            )
            self.assertEqual(
                secrets.dashboard_username,
                environment["ETRADE_DASHBOARD_USER"],
            )
            rendered = repr(secrets)
            self.assertIn("configured_fields", rendered)
            self.assertIn("source='environment'", rendered)
            for value in environment.values():
                self.assertNotIn(value, rendered)
            self.assertFalse(hasattr(secrets, "production_arming_secret"))
            with self.assertRaises(FrozenInstanceError):
                secrets.dashboard_password = "replacement-password"
            self.assertFalse(config.paths.secrets_file.exists())

    def test_partial_environment_overrides_owner_only_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_runtime_config(_write_config(base))
            fallback = _secret_document()
            _write_secret_file(config, fallback)
            environment = {
                "ETRADE_DASHBOARD_USER": "environment-operator",
                "ETRADE_DASHBOARD_PASSWORD": "environment-password-value",
            }

            secrets = resolve_runtime_secrets(
                config,
                environ=environment,
            )

            self.assertEqual(
                secrets.source,
                "environment+owner-only-file",
            )
            self.assertEqual(
                secrets.dashboard_username,
                "environment-operator",
            )
            self.assertEqual(
                secrets.dashboard_password,
                "environment-password-value",
            )
            self.assertEqual(
                secrets.dashboard_pin,
                fallback["dashboard_pin"],
            )
            self.assertEqual(
                secrets.dashboard_session_secret,
                fallback["dashboard_session_secret"],
            )
            self.assertIsNone(secrets.etrade_consumer_key)
            self.assertIsNone(secrets.etrade_consumer_secret)

    def test_owner_only_secret_fallback_rejects_modes_symlinks_and_schema_drift(self):
        cases = ("group-readable", "symlink", "unknown-field")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                config = load_runtime_config(_write_config(base))
                if case == "group-readable":
                    _write_secret_file(config, mode=0o640)
                elif case == "symlink":
                    target = base / "secret-target.json"
                    _write_payload(
                        base,
                        json.dumps(_secret_document()).encode("utf-8"),
                        name=target.name,
                        mode=0o600,
                    )
                    config.paths.state_dir.mkdir(parents=True, exist_ok=True)
                    os.chmod(config.paths.root, 0o700)
                    os.chmod(config.paths.state_dir, 0o700)
                    config.paths.secrets_file.symlink_to(target)
                else:
                    document = _secret_document(unexpected="value")
                    _write_secret_file(config, document)
                with self.assertRaises(RuntimeConfigError):
                    resolve_runtime_secrets(config, environ={})

    def test_secret_values_are_validated_after_precedence_resolution(self):
        invalid_environments: list[tuple[str, dict[str, str]]] = []

        short_password = _environment_secrets()
        short_password["ETRADE_DASHBOARD_PASSWORD"] = "short"
        invalid_environments.append(("short dashboard password", short_password))

        weak_pin = _environment_secrets()
        weak_pin["ETRADE_DASHBOARD_PIN"] = "11111111"
        invalid_environments.append(("weak dashboard pin", weak_pin))

        default_user = _environment_secrets()
        default_user["ETRADE_DASHBOARD_USER"] = "admin"
        invalid_environments.append(("reserved dashboard user", default_user))

        placeholder = _environment_secrets()
        placeholder["ETRADE_DASHBOARD_PASSWORD"] = "changeme"
        invalid_environments.append(("placeholder secret", placeholder))

        repeated_session_key = _environment_secrets()
        repeated_session_key["ETRADE_DASHBOARD_SESSION_SECRET"] = "s" * 64
        invalid_environments.append(
            ("low-diversity dashboard session secret", repeated_session_key)
        )

        with tempfile.TemporaryDirectory() as temporary:
            config = load_runtime_config(_write_config(Path(temporary)))
            for label, environment in invalid_environments:
                with self.subTest(case=label):
                    with self.assertRaises(RuntimeConfigError):
                        resolve_runtime_secrets(
                            config,
                            environ=environment,
                        )

    def test_live_mode_refuses_file_fallback_but_accepts_complete_live_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = load_runtime_config(
                _write_config(base, _document(mode="live"))
            )
            _write_secret_file(config)

            incomplete_environment = _environment_secrets(
                broker_prefix="LIVE"
            )
            incomplete_environment.pop("ETRADE_PASS")
            with self.assertRaisesRegex(
                RuntimeConfigError,
                "service-injected environment",
            ):
                resolve_runtime_secrets(
                    config,
                    environ=incomplete_environment,
                )

            secrets = resolve_runtime_secrets(
                config,
                environ=_environment_secrets(broker_prefix="LIVE"),
            )
            self.assertEqual(secrets.source, "environment")
            self.assertEqual(
                secrets.etrade_consumer_key,
                "environment-consumer-key",
            )


class RuntimeConfigCliAndExampleTests(unittest.TestCase):
    def test_main_reports_machine_readable_success_and_concise_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            valid = _write_config(base)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = runtime_config_main(
                    ["validate", "--config", str(valid)]
                )
            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            output = json.loads(stdout.getvalue())
            self.assertEqual(
                set(output),
                {
                    "schema_version",
                    "mode",
                    "broker_environment",
                    "broker_mutations_authorized",
                    "runtime_root",
                    "source_sha256",
                },
            )
            self.assertEqual(output["mode"], "paper")
            self.assertFalse(output["broker_mutations_authorized"])
            self.assertFalse((base / "runtime").exists())

            invalid_document = _document()
            invalid_document["mode"] = "production"
            invalid = _write_config(
                base,
                invalid_document,
                name="invalid.json",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = runtime_config_main(
                    ["validate", "--config", str(invalid)]
                )
            self.assertEqual(result, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("runtime configuration invalid:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_module_cli_checks_directories_and_service_injected_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            working = base / "working"
            working.mkdir()
            config_path = _write_config(
                base,
                _document(runtime_root="cli-runtime"),
            )
            repository = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            current_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(repository)
                if not current_pythonpath
                else os.pathsep.join((str(repository), current_pythonpath))
            )
            environment.update(_environment_secrets())
            command = [
                sys.executable,
                "-m",
                "live_trading.runtime_config",
                "validate",
                "--config",
                str(config_path),
                "--check-directories",
                "--check-secrets",
                "--check-dashboard-secrets",
            ]

            missing_directories = subprocess.run(
                command,
                cwd=working,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(missing_directories.returncode, 2)
            self.assertIn(
                "runtime directory is unavailable",
                missing_directories.stderr,
            )
            self.assertEqual(list(working.iterdir()), [])

            config = load_runtime_config(config_path)
            _create_runtime_directories(config)
            success = subprocess.run(
                command,
                cwd=working,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(success.stderr, "")
            output = json.loads(success.stdout)
            self.assertEqual(output["runtime_root"], str(config.paths.root))
            self.assertEqual(output["mode"], "paper")
            self.assertEqual(list(working.iterdir()), [])

    def test_packaged_example_is_credential_free_disabled_and_loadable(self):
        resource = importlib.resources.files("live_trading").joinpath(
            "runtime_config.example.json"
        )
        self.assertTrue(resource.is_file())
        with importlib.resources.as_file(resource) as example_path:
            payload_before = example_path.read_bytes()
            document = json.loads(payload_before)
            self.assertEqual(set(document), EXPECTED_TOP_LEVEL_FIELDS)
            self.assertEqual(document["mode"], "paper")
            self.assertEqual(
                document["execution"],
                {
                    "selected_account_id_key": None,
                    "account_allowlist": [],
                    "broker_mutations_enabled": False,
                },
            )
            lowered = payload_before.lower()
            for forbidden in (
                b"consumer_key",
                b"consumer_secret",
                b"etrade_username",
                b"etrade_password",
                b"dashboard_password",
                b"dashboard_pin",
                b"session_secret",
            ):
                with self.subTest(forbidden=forbidden):
                    self.assertNotIn(forbidden, lowered)

            expected_runtime_root = (
                Path(os.path.realpath(example_path.parent)) / "runtime"
            )
            root_existed_before = expected_runtime_root.exists()
            config = load_runtime_config(example_path)
            self.assertEqual(config.mode, "paper")
            self.assertFalse(config.strategy.enabled)
            self.assertFalse(config.model.enabled)
            self.assertFalse(config.execution.broker_mutations_enabled)
            self.assertIsNone(config.selected_account)
            self.assertEqual(config.paths.root, expected_runtime_root)
            self.assertEqual(
                expected_runtime_root.exists(),
                root_existed_before,
            )
            self.assertEqual(example_path.read_bytes(), payload_before)


if __name__ == "__main__":
    unittest.main()
