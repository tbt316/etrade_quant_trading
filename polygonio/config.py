from dataclasses import dataclass
import os

# ---------------------------------------------------------
# Config / Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    # Trading Costs & Risk
    option_trade_cost: float = 0.5
    spread_cost: float = 0.1
    min_profit: float = 0.05
    max_profit: float = 0.2

    # Data & Backtest Parameters
    lookback_window: int = 5
    validation_month_forward: int = 6
    initial_capital: int = 100_000
    cover_call_max_positions: int = 20

    # Option Chain & Pricing
    option_chain_force_update: bool = False
    skip_earnings: bool = False
    skip_missing_strike_trade: bool = False
    option_range: float = 0.1
    use_trade_data: bool = True
    price_interpolate: bool = True

    # Filters
    vol_threshold: float = -1
    iv_threshold_min: float = 0

    # Premium Pricing Mode
    # choose from: "close", "mid", "trade"
    premium_price_mode: str = "trade"

    # External API Keys
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")

    # Debug flags (tweak at runtime if needed)
    debug_closure: bool = False            # print per-position close/trigger details
    debug_closure_verbose: bool = False    # include per-leg quotes used for close-cost
    debug_open_sizing: bool = False        # print capital/margin sizing decisions for opens
    debug_plot: bool = False               # print plot input diagnostics
    debug_plot_verbose: bool = False       # more detailed plot diagnostics


# ---------------------------------------------------------
# Field Maps
# ---------------------------------------------------------

PREMIUM_FIELD_MAP = {
    "close": "close_price",
    "mid": "mid_price",
    "trade": "trade_price",
}

DELTA_FIELD_MAP = {
    "close": "close_price_delta",
    "mid": "mid_price_delta",
    "trade": "trade_price_delta",
}


# ---------------------------------------------------------
# Singleton Accessor
# ---------------------------------------------------------

_settings_instance: Settings | None = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
