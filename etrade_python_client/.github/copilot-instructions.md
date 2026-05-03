# Copilot Instructions for EtradePythonClient

## Project Overview
This repository contains a Python-based options trading and backtesting system integrating Polygon.io, E*TRADE APIs, and yfinance. It supports advanced strike selection, rolling logic, multiprocessing for backtests, reinforcement learning stubs, and robust order tracking via CSV logs.

## Key Architectural Patterns
- **Modular Scripts:** Each major workflow (backtesting, live trading, option chain analysis, plotting) is implemented as a separate script for clarity and maintainability.
- **API Integration:** Polygon.io and E*TRADE APIs are used for market data and order execution. yfinance is used for historical price data.
- **Data Caching:** Option chain and price data are cached using pickle and CSV for performance and reliability.
- **Multiprocessing:** Backtests can be parallelized across tickers using `ProcessPoolExecutor`.
- **Logging & Error Handling:** Extensive use of logging and defensive coding to handle API errors, NoneType, IndexError, and EOFError scenarios.
- **Order Tracking:** Orders and their statuses are tracked in CSV files, with periodic updates from executed order queries.
- **Plotting:** Day-by-day option effectiveness is visualized using matplotlib, with close prices polled from yfinance.
- **Reinforcement Learning:** RL stubs are provided for future parameter optimization using Stable-Baselines3 PPO and Gym.

## Coding Conventions
- **Python 3.10+** is recommended.
- **Pandas** is used for data manipulation; **numpy** for numerical operations.
- **Function Naming:** Use descriptive names (e.g., `find_closest_premium_strike`, `monthly_recursive_backtest`).
- **Type Conversion:** Always convert API results to expected types (float, int) and check for None before processing.
- **Error Handling:** Use try/except blocks around API calls and file operations. Log errors for debugging.
- **CSV Updates:** When updating order status, always check for matching executed orders and update price/quantity fields.
- **Multiprocessing:** Use `ProcessPoolExecutor` for parallel ticker backtests; avoid global state.

## AI Agent Productivity Tips
- **Workspace Analysis:** Use semantic and grep search to locate relevant scripts and functions.
- **Code Refactoring:** Modularize logic for reusability (e.g., strike selection, order status updates).
- **Testing:** Validate changes with unit tests and by running main scripts. Check for new errors after edits.
- **Documentation:** Update this file and add docstrings/comments to new functions for future maintainability.
- **Error Resolution:** When encountering tracebacks, prioritize defensive coding and type checks.
- **Integration Points:** When adding new features (e.g., RL, new roll methods), ensure compatibility with existing data structures and logging.

## Common Workflows
- **Backtesting:** Run `monthly_recursive_backtest` or `backtest_options_sync_or_async` for multi-ticker, multi-core backtests.
- **Order Tracking:** Use `update_csv_order_statuses` to sync CSV logs with executed orders from E*TRADE.
- **Plotting:** Use scripts like `vol_plot.py` and custom matplotlib logic for day-by-day effectiveness plots.
- **API Mapping:** Use helper functions to cross-reference tickers between Polygon.io and E*TRADE APIs.

## Advanced Features
- **Rolling Logic:** Supports searching for options with expiration longer than 1 week when rolling assignments.
- **Volume Filtering:** Strike selection functions filter by volume before returning results.
- **RL Parameter Selection:** RL stubs provided for future integration; see `Stable-Baselines3` and `Gym` usage.

## Onboarding Checklist for AI Agents
1. Review this file and main scripts for architectural patterns.
2. Use semantic/grep search to locate functions and workflows.
3. Validate code changes with error checks and test runs.
4. Update documentation and add comments for new logic.
5. Ensure compatibility with CSV logging and API integration.
6. Ask for user feedback on unclear or incomplete sections.

---
For questions or improvements, update this file and notify the user for review.
