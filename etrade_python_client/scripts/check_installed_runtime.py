#!/usr/bin/env python3
"""Smoke the supported runtime surface from an installed wheel."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import os
import shutil
import sys
import tempfile
from pathlib import Path


PACKAGE_ROOTS = (
    "accounts",
    "ai_agents",
    "backtesting",
    "core_api",
    "data_and_research",
    "live_trading",
    "market",
    "order",
    "polygonio",
    "strategies",
)
READ_ONLY_MODULES = (
    "live_trading.positions_artifact",
    "live_trading.read_only_dashboard",
    "live_trading.runtime_composition",
    "live_trading.runtime_config",
)
SUPPORTED_MODULES = (
    "accounts.accounts_bo",
    "backtesting.backtest_runner",
    "core_api.stock_trade_class",
    "data_and_research.vol_plot",
    "live_trading.etrade_cover_call_new",
    "live_trading.etrade_put_credit_spread",
    "live_trading.positions_artifact_publisher",
    "live_trading.regime_detector_v2",
    "live_trading.runtime_safety",
    "market.market_bo",
    "order.order_bo",
    "polygonio.poly_client",
    "strategies.strategies",
)
FORBIDDEN_READ_ONLY_IMPORT_PREFIXES = (
    "accounts.accounts_bo",
    "aiohttp",
    "core_api.stock_trade_class",
    "data_and_research.polygonio_config",
    "live_trading.etrade_broker_transport",
    "live_trading.etrade_cover_call_new",
    "live_trading.etrade_order_gateway",
    "live_trading.order_intent_ledger",
    "live_trading.positions_artifact_publisher",
    "order.order_bo",
    "polygonio.poly_client",
    "pyetrade",
    "rauth",
    "requests",
    "yfinance",
)
FORBIDDEN_RUNTIME_DISTRIBUTIONS = (
    "pip",
    "poetry-core",
    "setuptools",
    "wheel",
)


class RuntimeSmokeError(RuntimeError):
    """The installed runtime is incomplete or leaked into the source tree."""


def _installed_module(name: str, source_root: Path) -> None:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeSmokeError(f"{name} has no concrete module file")
    module_path = Path(module_file).resolve()
    if source_root == module_path or source_root in module_path.parents:
        raise RuntimeSmokeError(f"{name} resolved from source: {module_path}")
    if "site-packages" not in module_path.parts:
        raise RuntimeSmokeError(
            f"{name} did not resolve from site-packages: {module_path}"
        )


def main() -> int:
    source_root = Path(__file__).resolve().parents[1]
    working_directory = Path.cwd()
    initial_entries = set(working_directory.iterdir())
    if initial_entries:
        raise RuntimeSmokeError(
            "installed runtime smoke requires an empty working directory"
        )
    for name in (*PACKAGE_ROOTS, *READ_ONLY_MODULES):
        _installed_module(name, source_root)
    runtime_example = importlib.resources.files("live_trading").joinpath(
        "runtime_config.example.json"
    )
    if not runtime_example.is_file():
        raise RuntimeSmokeError(
            "installed runtime-configuration example is missing"
        )
    with tempfile.TemporaryDirectory(
        prefix=".read-only-smoke-",
        dir=working_directory,
    ) as temporary:
        private = Path(temporary)
        config_path = private / "runtime-config.json"
        shutil.copyfile(runtime_example, config_path)
        os.chmod(config_path, 0o600)
        runtime_root = private / "runtime"
        for relative in (
            "",
            "state",
            "cache",
            "logs",
            "artifacts",
            "execution",
            "data",
            "model",
        ):
            directory = runtime_root / relative
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        from live_trading.read_only_dashboard import (
            ReadOnlyDashboardApplication,
        )

        application = ReadOnlyDashboardApplication.from_config(
            config_path,
            environ={
                "ETRADE_DASHBOARD_USER": "runtime-smoke-operator",
                "ETRADE_DASHBOARD_PASSWORD": (
                    "runtime-smoke-password-value"
                ),
                "ETRADE_DASHBOARD_SESSION_SECRET": (
                    "runtime-smoke-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R7Ts1Qa"
                ),
                "ETRADE_POSITIONS_ARTIFACT_HMAC_KEY": (
                    "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"
                ),
            },
        )
        if (
            application.runtime.broker_mutations_enabled
            or hasattr(application.runtime, "config")
            or hasattr(application.runtime, "paths")
            or hasattr(application.runtime.settings, "paths")
            or hasattr(application.runtime.positions_artifact_reader, "publish")
            or hasattr(application.runtime.regime_shadow_reader, "publish")
        ):
            raise RuntimeSmokeError(
                "read-only composition retained an unsafe capability"
            )
    forbidden_imports = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_READ_ONLY_IMPORT_PREFIXES
        )
    )
    if forbidden_imports:
        raise RuntimeSmokeError(
            "read-only runtime imported broker or provider capability: "
            f"{forbidden_imports}"
        )
    for name in SUPPORTED_MODULES:
        _installed_module(name, source_root)

    distribution = importlib.metadata.distribution("etrade-quant-trading")
    if distribution.version != "0.1.0":
        raise RuntimeSmokeError(
            f"unexpected installed version: {distribution.version}"
        )
    console_scripts = sorted(
        (entry.name, entry.value)
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    )
    if console_scripts != [
        (
            "etrade-read-only-dashboard",
            "live_trading.read_only_dashboard:main",
        ),
        (
            "etrade-runtime-config",
            "live_trading.runtime_config:main",
        ),
    ]:
        raise RuntimeSmokeError(
            f"unexpected console entry points: {console_scripts}"
        )
    for name in FORBIDDEN_RUNTIME_DISTRIBUTIONS:
        try:
            installed = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeSmokeError(
            f"build-only distribution remains in runtime: "
            f"{installed.metadata['Name']}"
        )
    dashboard = importlib.resources.files("live_trading").joinpath(
        "dashboard_template.html"
    )
    if not dashboard.is_file():
        raise RuntimeSmokeError("installed dashboard template is missing")
    read_only_dashboard = importlib.resources.files("live_trading").joinpath(
        "read_only_dashboard.html"
    )
    if not read_only_dashboard.is_file():
        raise RuntimeSmokeError(
            "installed read-only dashboard template is missing"
        )
    from live_trading.runtime_config import (
        CONFIG_SCHEMA_VERSION,
        load_runtime_config,
    )

    with importlib.resources.as_file(runtime_example) as config_path:
        runtime_config = load_runtime_config(config_path)
    if runtime_config.schema_version != CONFIG_SCHEMA_VERSION:
        raise RuntimeSmokeError(
            "installed runtime-configuration example has the wrong schema"
        )
    if (
        runtime_config.mode != "paper"
        or runtime_config.execution.broker_mutations_enabled is not False
    ):
        raise RuntimeSmokeError(
            "installed runtime-configuration example is not disabled paper mode"
        )
    strategies = sorted(
        path.name
        for path in importlib.resources.files("backtesting")
        .joinpath("strategies")
        .iterdir()
        if path.name.endswith(".yaml")
    )
    if strategies != [
        "baseline_put_spread.yaml",
        "baseline_put_spread_catchup_refill.yaml",
        "put_call_credit_spread.yaml",
    ]:
        raise RuntimeSmokeError(
            f"installed strategy payload differs: {strategies}"
        )

    created_entries = sorted(
        path.name
        for path in set(working_directory.iterdir()) - initial_entries
    )
    if created_entries:
        raise RuntimeSmokeError(
            f"runtime imports created paths: {created_entries}"
        )

    print(
        "installed runtime smoke passed: "
        f"{len(PACKAGE_ROOTS)} roots, "
        f"{len(READ_ONLY_MODULES) + len(SUPPORTED_MODULES)} modules"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeSmokeError as error:
        print(f"installed runtime smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
