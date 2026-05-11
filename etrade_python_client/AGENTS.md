# Repository Instructions for Coding Agents

## Market Regime Detection Work

Before modifying, reviewing, or running market-regime detection, EV probability, or regime-aware backtest code, read:

- `docs/market_regime_detect_specs.md`
- `backtesting/strategy_registry.md`

This applies at minimum to:

- `live_trading/ev_engine.py`
- `live_trading/ev_plots.py`
- `live_trading/data_ingestion.py`
- `live_trading/pca_fusion.py`
- `backtesting/backtest_runner.py`
- `scratch/*regime*`
- `scratch/*backtest*`

Treat `docs/market_regime_detect_specs.md` as the source of truth for non-anticipativity, timestamp discipline, causal scaling/PCA, HMM filtering, regime label provenance, return bucket availability, cache validity, and backtest regime-map lagging.

For any regime-aware historical backtest, explicitly record:

- training/calibration end date
- test date range
- whether HMM inference is walk-forward or fixed out-of-sample filtering
- whether daily regime values are lagged before trade entry
- whether return/probability buckets include only outcomes resolved before each trade date

Do not present a regime-aware backtest as valid if any of those fields are unknown.
