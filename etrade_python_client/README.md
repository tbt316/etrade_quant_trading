# Etrade Python Client

## Architecture & Directory Structure

The project code is organized into 5 logical modules (packages) to clearly separate live trading logic, backtesting features, underlying broker API connections, data-gathering scripts, and standalone AI agents.

```text
etrade_python_client/
├── live_trading/             # Core trading scripts running live against E*TRADE
│   ├── etrade_cover_call_new.py      # Main cover call agent (ENTRY POINT)
│   ├── etrade_put_credit_spread.py   # Put credit spread logic
│   ├── etrade_check_option.py        # Option checker logic
│   ├── spy_position_tracker.py       # SPY specific position tracker
│   ├── portfolio_manager.py          # General portfolio tracking
│   ├── etrade_option_chains.py       # Option chain puller for live E*TRADE
│   └── email_snapshot.py             # Automates sending status emails
│
├── backtesting/              # Offline historical testing systems (System B & others)
│   ├── polygonio_dailytrade.py       # Main Polygon backtest engine
│   ├── polygon_multi.py              # Multi-ticker orchestrator
│   ├── monthly_cpu_bound.py          # Multiprocessing tasks
│   ├── backtest_bo.py                # YFinance backtest logic
│   └── option_limit_backtest.py      # Limit backtest logic
│
├── core_api/                 # Shared foundation connecting to brokers/platforms
│   ├── auth.py                       # E*TRADE OAuth flow
│   └── stock_trade_class.py          # Core Selenium+API E*TRADE wrapper
│
├── data_and_research/        # Scripts evaluating data, scraping, or plotting
│   ├── polygonio_improvequery.py     # Polygon helper/data loader (Shared!)
│   ├── polygonio_config.py           # Polygon API keys/config
│   ├── option_assign_probability.py  # Calculates assignment risk
│   ├── vol_plot.py                   # Volatility plotting
│   ├── history_option.py             # Pulls historical option info
│   ├── option_price.py               # Evaluates option prices
│   ├── check_volume_sp500.py         # S&P500 volume checker
│   ├── daily_return.py               # Daily returns calculator
│   ├── find_option_volume.py         # Option volume screener
│   ├── return_hist.py                # Historical return analysis
│   ├── annual_volatility_calc.py     # Volatility calculator
│   ├── ticker_event_query.py         # Pulls events (earnings/dividends)
│   ├── download_ticker.py            # YFinance data downloader
│   └── polygon_trial.py              # Sandbox/experiment script
│
└── ai_agents/                # Standalone AI analysis & web scraping tools
    ├── gemini_analyst.py             # AI market analyst runner
    ├── main_analyst_agent.py         # Orchestrator for agent logic
    ├── integrate_browser_use.py      # Browser automation for agents
    └── report_scraper.py             # Scraping external analyst reports
```

## Documentation
- [Market Regime Detection Specification](docs/market_regime_detect_specs.md): Institutional-grade regulations for causal regime modeling, manifold stabilization, and non-anticipative features.

## Running the Live Agent
The main live trading agent should be run using Python 3. Ensure the virtual environment (`venv`) is activated.

Execution example:
`python3 -m live_trading.etrade_cover_call_new --no-sandbox --trade --username [USERNAME] --...`
