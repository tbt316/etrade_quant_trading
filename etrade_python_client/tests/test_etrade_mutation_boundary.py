from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_etrade_mutation_boundary import (
    APPLICATION_ROOT,
    Diagnostic,
    NON_HTTP_MUTATION_REFERENCES,
    REQUIRED_TOMBSTONES,
    TombstoneSpec,
    TRANSPORT_INTERNAL_CAPABILITIES,
    TRANSPORT_MUTATION_METHODS,
    VENDOR_MUTATION_CAPABILITIES,
    VENDOR_MUTATION_METHODS,
    scan_paths,
    scan_repository,
    tracked_production_python_files,
)


EXPECTED_TOMBSTONE_REASONS = {
    "order.order.Order.__init__": "order.order.Order.__init__",
    **{
        f"order.order_bo.Order.{name}": f"order.order_bo.Order.{name}"
        for name in (
            "preview_order",
            "preview_order_old",
            "place_order",
            "place_order_old",
            "change_order_limit",
            "wait_and_adjust_until_filled",
            "previous_order",
            "user_select_order",
            "preview_order_menu",
            "cancel_order",
            "place_option_order",
            "cancel_all_order",
            "refresh_order_limit_old",
            "refresh_order_limit",
            "options_selection",
            "view_orders",
            "filter_order",
        )
    },
    (
        "core_api.stock_trade_class.LiveTradeAgent."
        "check_short_availability"
    ): (
        "core_api.stock_trade_class.LiveTradeAgent."
        "check_short_availability"
    ),
    "core_api.stock_trade_class.LiveTradeAgent.place_order": (
        "core_api.stock_trade_class.LiveTradeAgent.place_order"
    ),
    "live_trading.etrade_put_credit_spread.release_margin": (
        "live_trading.etrade_put_credit_spread.release_margin"
    ),
    "live_trading.etrade_cover_call_new.enqueue_manual_trade_request": (
        "legacy dashboard manual-trade queue"
    ),
    "live_trading.etrade_cover_call_new._execute_neutralize_leg": (
        "legacy dashboard neutralize worker"
    ),
    "live_trading.etrade_cover_call_new.release_margin": (
        "legacy automatic margin release"
    ),
    "live_trading.etrade_cover_call_new.submit_manual_open_fast_async": (
        "legacy dashboard manual-open scheduler"
    ),
    "live_trading.etrade_cover_call_new._manual_open_fast_worker": (
        "legacy dashboard manual-open worker"
    ),
    "live_trading.etrade_cover_call_new.submit_order_async": (
        "legacy dashboard order scheduler"
    ),
    "live_trading.etrade_cover_call_new._order_worker": (
        "legacy dashboard order worker"
    ),
    "live_trading.etrade_cover_call_new.monitor_and_nudge_stale_orders": (
        "legacy stale-order nudge worker"
    ),
    "scratch.check_orders.get_today_orders": (
        "scratch.check_orders.get_today_orders"
    ),
}
EXPECTED_NON_HTTP_MUTATION_REFERENCES = {
    ("accounts/accounts.py", "response.request"),
    ("accounts/accounts_bo.py", "response.request"),
    ("ai_agents/report_scraper.py", "route.request"),
    ("ai_agents/report_scraper.py", "self.page.context.request"),
    ("ai_agents/report_scraper.py", "self.page.request"),
    ("backtesting/experiment_manager.py", "args.delete"),
    ("market/market.py", "response.request"),
    ("market/market_bo.py", "response.request"),
    ("order/order_bo.py", "response.request"),
    ("scratch/fslr_30dte_iv.py", "fig.patch"),
    ("scratch/mu_30dte_iv.py", "fig.patch"),
    ("scratch/rklb_20delta_call_iv.py", "fig.patch"),
    ("scratch/run_comparison_2019_2025.py", "fig.patch"),
}
EXPECTED_VENDOR_MUTATION_METHODS = {
    "change_preview_equity_order",
    "change_preview_option_order",
    "perform_request",
    "place_changed_equity_order",
    "place_changed_option_order",
    "place_equity_order",
    "place_option_order",
    "preview_equity_order",
    "preview_option_order",
}
EXPECTED_TRANSPORT_INTERNAL_CAPABILITIES = {
    "_PreparedExchange",
    "_build_request",
    "_exchange_worker",
    "_execute",
    "_isolated_mutation_exchange",
    "_new_pinned_adapter",
    "_prepare_oauth_request",
    "_reconstruct_prepared_request",
    "_run_isolated_exchange",
    "_serialized_prepared_request",
}
EXPECTED_TRANSPORT_MUTATION_METHODS = {
    "preview",
    "place",
    "preview_change",
    "place_change",
    "cancel",
}


def _write_sources(root: Path, sources: dict[str, str]) -> None:
    for relative_path, source in sources.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _codes(diagnostics: list[Diagnostic]) -> list[str]:
    return [diagnostic.code for diagnostic in diagnostics]


def test_required_tombstone_policy_cannot_silently_shrink() -> None:
    assert {
        qualified_name: spec.reason
        for qualified_name, spec in REQUIRED_TOMBSTONES.items()
    } == EXPECTED_TOMBSTONE_REASONS


def test_mutation_capability_policy_cannot_silently_drift() -> None:
    assert set(NON_HTTP_MUTATION_REFERENCES) == (
        EXPECTED_NON_HTTP_MUTATION_REFERENCES
    )
    assert set(VENDOR_MUTATION_CAPABILITIES) == {"ETradeOrder"}
    assert set(VENDOR_MUTATION_METHODS) == EXPECTED_VENDOR_MUTATION_METHODS
    assert set(TRANSPORT_INTERNAL_CAPABILITIES) == (
        EXPECTED_TRANSPORT_INTERNAL_CAPABILITIES
    )
    assert set(TRANSPORT_MUTATION_METHODS) == (
        EXPECTED_TRANSPORT_MUTATION_METHODS
    )


def test_exact_call_only_tombstone_is_accepted(tmp_path: Path) -> None:
    qualified_name = "pkg.legacy.Legacy.mutate"
    _write_sources(
        tmp_path,
        {
            "pkg/legacy.py": (
                "from live_trading.runtime_safety import "
                "reject_legacy_execution\n\n"
                "class Legacy:\n"
                "    def mutate(self):\n"
                '        """Compatibility surface."""\n'
                f'        reject_legacy_execution("{qualified_name}")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["pkg/legacy.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )

    assert diagnostics == []


def test_tombstone_rejects_extra_body_decorator_alias_and_wrong_reason(
    tmp_path: Path,
) -> None:
    qualified_name = "pkg.legacy.Legacy.mutate"
    _write_sources(
        tmp_path,
        {
            "pkg/legacy.py": (
                "from live_trading.runtime_safety import "
                "reject_legacy_execution as reject\n\n"
                "class Legacy:\n"
                "    @staticmethod\n"
                "    def mutate():\n"
                '        reject_legacy_execution("wrong")\n'
                '        session.post("/v1/accounts/1/orders/place")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["pkg/legacy.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )

    codes = _codes(diagnostics)
    assert "TOMBSTONE_IMPORT" in codes
    assert "TOMBSTONE_DECORATOR" in codes
    assert "TOMBSTONE_BODY" in codes
    assert "RAW_HTTP_MUTATION" in codes
    assert "ETRADE_MUTATION_LITERAL" in codes


def test_tombstone_reject_import_cannot_be_shadowed(tmp_path: Path) -> None:
    qualified_name = "pkg.legacy.Legacy.mutate"
    _write_sources(
        tmp_path,
        {
            "pkg/legacy.py": (
                "from live_trading.runtime_safety import "
                "reject_legacy_execution\n"
                "reject_legacy_execution = lambda reason: None\n\n"
                "class Legacy:\n"
                "    def mutate(self):\n"
                f'        reject_legacy_execution("{qualified_name}")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["pkg/legacy.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )

    assert "TOMBSTONE_IMPORT" in _codes(diagnostics)


def test_tombstone_rejector_cannot_be_shadowed_by_a_parameter(
    tmp_path: Path,
) -> None:
    qualified_name = "pkg.legacy.mutate"
    _write_sources(
        tmp_path,
        {
            "pkg/legacy.py": (
                "from live_trading.runtime_safety import "
                "reject_legacy_execution\n\n"
                "def mutate("
                "reject_legacy_execution=lambda reason: None):\n"
                f'    reject_legacy_execution("{qualified_name}")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["pkg/legacy.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )

    assert "TOMBSTONE_SIGNATURE" in _codes(diagnostics)


def test_missing_and_duplicate_tombstones_fail_closed(tmp_path: Path) -> None:
    qualified_name = "pkg.legacy.Legacy.mutate"
    _write_sources(
        tmp_path,
        {
            "pkg/legacy.py": (
                "from live_trading.runtime_safety import "
                "reject_legacy_execution\n\n"
                "class Legacy:\n"
                "    def mutate(self):\n"
                f'        reject_legacy_execution("{qualified_name}")\n'
                "    def mutate(self):\n"
                f'        reject_legacy_execution("{qualified_name}")\n'
            ),
            "pkg/other.py": "value = 1\n",
        },
    )

    duplicate = scan_paths(
        tmp_path,
        ["pkg/legacy.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )
    missing = scan_paths(
        tmp_path,
        ["pkg/other.py"],
        required_tombstones={
            qualified_name: TombstoneSpec(
                "pkg/legacy.py", qualified_name
            )
        },
    )

    assert "TOMBSTONE_DUPLICATE" in _codes(duplicate)
    assert "TOMBSTONE_MISSING" in _codes(missing)


def test_raw_http_methods_literals_and_legacy_calls_are_rejected(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                'PREVIEW = "<PreviewOrderRequest></PreviewOrderRequest>"\n'
                'URL = "/v1/accounts/1/orders/preview.json"\n'
                "session.post(URL)\n"
                "session.put(URL)\n"
                "session.patch(URL)\n"
                "session.delete(URL)\n"
                'session.request("POST", URL)\n'
                "adapter.send(request)\n"
                "client.place_order(order)\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, ["bad.py"], required_tombstones={}
    )

    assert _codes(diagnostics).count("RAW_HTTP_MUTATION") == 6
    assert _codes(diagnostics).count("ETRADE_MUTATION_LITERAL") == 2
    assert _codes(diagnostics).count("LEGACY_MUTATION_CALL") == 1


def test_narrow_non_http_delete_allowlist_is_preserved(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "ai_agents/gemini_analyst.py": (
                "self.client.files.delete(name=file_name)\n"
            ),
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        [
            "ai_agents/gemini_analyst.py",
        ],
        required_tombstones={},
    )

    assert diagnostics == []


def test_http_mutation_import_and_reflection_are_rejected_without_false_defaults(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                "from requests import post as issue\n"
                'getattr(session, "delete")(url)\n'
                'value = getattr(option, "call_put", "PUT")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, ["bad.py"], required_tombstones={}
    )

    assert _codes(diagnostics) == [
        "RAW_HTTP_MUTATION_IMPORT",
        "RAW_HTTP_MUTATION_REFLECTION",
    ]


def test_bound_http_mutation_alias_is_rejected(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                "sender = session.post\n"
                "sender(url)\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, ["bad.py"], required_tombstones={}
    )

    assert _codes(diagnostics) == ["RAW_HTTP_MUTATION_REFERENCE"]


def test_direct_pyetrade_order_capabilities_are_rejected(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                "import pyetrade\n"
                "orders = pyetrade.ETradeOrder(*credentials)\n"
                "orders.preview_equity_order(payload)\n"
                "orders.place_equity_order(payload)\n"
                "callback = orders.perform_request\n"
                'getattr(orders, "place_changed_option_order")(payload)\n'
            ),
            "also_bad.py": (
                "from pyetrade.order import ETradeOrder as OrderClient\n"
                "client = OrderClient(*credentials)\n"
            ),
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["bad.py", "also_bad.py"],
        required_tombstones={},
    )
    codes = _codes(diagnostics)

    assert codes.count("VENDOR_MUTATION_CAPABILITY") >= 2
    assert codes.count("VENDOR_MUTATION_CALL") == 2
    assert "VENDOR_MUTATION_REFERENCE" in codes
    assert "VENDOR_MUTATION_REFLECTION" in codes


def test_central_rejector_cannot_be_turned_into_a_no_op(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "live_trading/runtime_safety.py": (
                "from typing import NoReturn\n"
                "class LegacyExecutionDisabled(RuntimeError):\n"
                "    pass\n"
                "def reject_legacy_execution(surface: str) -> NoReturn:\n"
                "    return None\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["live_trading/runtime_safety.py"],
        required_tombstones={},
    )

    assert "CENTRAL_REJECTOR_BODY" in _codes(diagnostics)


def test_transport_import_call_reference_and_reflection_are_rejected(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                "from live_trading.etrade_broker_transport import "
                "ETradeBrokerTransport\n"
                "transport = ETradeBrokerTransport()\n"
                "transport.preview(auth)\n"
                "callback = transport.place\n"
                'getattr(transport, "preview_change")(auth)\n'
                "module = __import__("
                '"live_trading.etrade_broker_transport")\n'
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, ["bad.py"], required_tombstones={}
    )
    codes = set(_codes(diagnostics))

    assert {
        "TRANSPORT_IMPORT",
        "TRANSPORT_CONSTRUCTION",
        "TRANSPORT_MUTATION_CALL",
        "TRANSPORT_MUTATION_REFERENCE",
        "TRANSPORT_REFLECTION",
    }.issubset(codes)


def test_reexported_and_private_transport_send_capabilities_are_rejected(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "bad.py": (
                "from live_trading.etrade_broker_reader import "
                "_run_isolated_exchange\n"
                "sender = gateway._transport._execute\n"
                "sender(request, authorization)\n"
                "raw = gateway.__dict__['_transport']\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, ["bad.py"], required_tombstones={}
    )
    codes = set(_codes(diagnostics))

    assert {
        "TRANSPORT_CAPABILITY_IMPORT",
        "GATEWAY_TRANSPORT_ACCESS",
        "TRANSPORT_CAPABILITY_REFERENCE",
        "TRANSPORT_CAPABILITY_LOOKUP",
    }.issubset(codes)


def test_relative_transport_import_is_also_rejected(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        {
            "live_trading/bad.py": (
                "from .etrade_broker_transport import ETradeBrokerTransport\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path,
        ["live_trading/bad.py"],
        required_tombstones={},
    )

    assert "TRANSPORT_IMPORT" in _codes(diagnostics)


def test_only_exact_gateway_transport_invocations_are_allowed(
    tmp_path: Path,
) -> None:
    gateway = "live_trading/etrade_order_gateway.py"
    _write_sources(
        tmp_path,
        {
            gateway: (
                "class EtradeOrderGateway:\n"
                "    def _submit_intent(self):\n"
                "        self._transport.preview(auth)\n"
                "        self._transport.place(auth, preview)\n"
                "    def reprice_opening(self):\n"
                "        self._transport.preview_change(order_id, auth)\n"
                "        self._transport.place_change(order_id, auth, preview)\n"
                "    def _cancel_order(self):\n"
                "        self._transport.cancel(auth)\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, [gateway], required_tombstones={}
    )

    assert diagnostics == []


def test_gateway_transport_alias_calls_and_public_property_are_rejected(
    tmp_path: Path,
) -> None:
    gateway = "live_trading/etrade_order_gateway.py"
    _write_sources(
        tmp_path,
        {
            gateway: (
                "class EtradeOrderGateway:\n"
                "    @property\n"
                "    def transport(self):\n"
                "        return self._transport\n"
                "    def _submit_intent(self):\n"
                "        self.transport.preview(auth)\n"
                "    def helper(self):\n"
                "        self._transport.place(auth, preview)\n"
            )
        },
    )

    diagnostics = scan_paths(
        tmp_path, [gateway], required_tombstones={}
    )
    codes = _codes(diagnostics)

    assert "GATEWAY_TRANSPORT_EXPOSURE" in codes
    assert codes.count("TRANSPORT_MUTATION_CALL") == 2


def test_syntax_errors_and_diagnostics_are_stably_sorted(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "z.py": "def broken(:\n",
            "a.py": "session.post(url)\n",
        },
    )

    first = scan_paths(
        tmp_path, ["z.py", "a.py"], required_tombstones={}
    )
    second = scan_paths(
        tmp_path, ["a.py", "z.py", "a.py"], required_tombstones={}
    )

    assert first == second == sorted(first)
    assert [diagnostic.path for diagnostic in first] == ["a.py", "z.py"]
    assert _codes(first) == ["RAW_HTTP_MUTATION", "SYNTAX_ERROR"]


def test_tracked_discovery_includes_scratch_and_excludes_only_tests_and_checker(
    tmp_path: Path,
) -> None:
    _write_sources(
        tmp_path,
        {
            "production.py": "value = 1\n",
            "scratch/archived.py": "value = 2\n",
            "tests/test_production.py": "value = 3\n",
            "scripts/check_etrade_mutation_boundary.py": "value = 4\n",
            "untracked.py": "value = 5\n",
        },
    )
    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "add",
            "production.py",
            "scratch/archived.py",
            "tests/test_production.py",
            "scripts/check_etrade_mutation_boundary.py",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    tracked = tracked_production_python_files(tmp_path)

    assert tracked == ("production.py", "scratch/archived.py")


def test_current_repository_satisfies_the_static_boundary() -> None:
    diagnostics, source_count = scan_repository(APPLICATION_ROOT)

    assert source_count > 100
    assert diagnostics == [], "\n".join(
        diagnostic.render() for diagnostic in diagnostics
    )
