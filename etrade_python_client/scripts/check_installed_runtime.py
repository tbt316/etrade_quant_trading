#!/usr/bin/env python3
"""Smoke the supported runtime surface from an installed wheel."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import stat
import sys
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
SUPPORTED_MODULES = (
    "accounts.accounts_bo",
    "backtesting.backtest_runner",
    "core_api.stock_trade_class",
    "data_and_research.vol_plot",
    "live_trading.etrade_cover_call_new",
    "live_trading.etrade_put_credit_spread",
    "live_trading.regime_detector_v2",
    "live_trading.runtime_config",
    "live_trading.runtime_safety",
    "market.market_bo",
    "order.order_bo",
    "polygonio.poly_client",
    "strategies.strategies",
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
    for name in (*PACKAGE_ROOTS, *SUPPORTED_MODULES):
        _installed_module(name, source_root)

    distribution = importlib.metadata.distribution("etrade-quant-trading")
    if distribution.version != "0.1.0":
        raise RuntimeSmokeError(
            f"unexpected installed version: {distribution.version}"
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
    runtime_example = importlib.resources.files("live_trading").joinpath(
        "runtime_config.example.json"
    )
    if not runtime_example.is_file():
        raise RuntimeSmokeError(
            "installed runtime-configuration example is missing"
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

    created_entries = set(working_directory.iterdir()) - initial_entries
    allowed_logs = {"etrade_trader.log", "python_client.log"}
    unexpected = sorted(
        path.name for path in created_entries if path.name not in allowed_logs
    )
    if unexpected:
        raise RuntimeSmokeError(
            f"runtime imports created unexpected paths: {unexpected}"
        )
    for path in created_entries:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeSmokeError(
                f"runtime import log is not owner-only regular file: {path}"
            )

    print(
        "installed runtime smoke passed: "
        f"{len(PACKAGE_ROOTS)} roots, {len(SUPPORTED_MODULES)} modules"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeSmokeError as error:
        print(f"installed runtime smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
