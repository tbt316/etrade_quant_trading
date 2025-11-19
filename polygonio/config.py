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
    initial_capital: int = 10_000
    cover_call_max_positions: int = 20

    # Option Chain & Pricing
    option_chain_force_update: bool = False
    skip_earnings: bool = False
    skip_missing_strike_trade: bool = False
    option_range: float = 0.1
    use_trade_data: bool = True
    price_interpolate: bool = True
    # Execution model: steer between bid/ask for fills (0..1). 0.5 = mid.
    # Lower than 0.5 steers buys toward ask and sells toward bid.
    execution_steer: float = 0.4

    # Filters
    vol_threshold: float = -1
    iv_threshold_min: float = 0

    # Premium Pricing Mode
    # choose from: "close", "mid", "trade"
    premium_price_mode: str = "trade"
    # Override polygon data source: "trade" (trades) or "quote" (bid/ask mid). Leave blank to use mode above.
    premium_price_source: str = "quote"

    # Premium sampling time controls (for "trade"/"mid" modes)
    # Target intraday time (local) to sample around when no explicit time is provided
    premium_time_target: str = "12:45:00"  # HH:MM:SS
    # premium_time_target: str = "07:30:00"  # HH:MM:SS
    # +/- window in seconds applied around the target time
    premium_time_window_secs: int = 30
    # Cutoff time (records after this time ignored client-side)
    premium_time_cutoff: str = "13:00:00"  # HH:MM:SS
    # Adaptive time search controls for cases with no data exactly at target time
    premium_time_step_secs: int = 120       # step size in seconds when searching around target time
    premium_time_max_steps: int = 4         # number of steps outward (tries 0, +1, -1, +2, -2, ...)
    # Require front/back SIP timestamps to be within this delta when accepting a pair
    premium_pair_delta_secs: int = 120
    premium_close_max_back_days: int = 15
    # Calendar expiry search window (+/- days) when locating front/back contracts
    premium_expiry_window_days: int = 20
    # Strike search bandwidth around spot when pre-filtering contract lookup
    premium_strike_spot_pct: float = 0.05
    # Minutes to step backward when retrying premium sampling after a mismatch
    premium_time_backoff_minutes: int = 60
    # Number of backward steps to try when searching for aligned premiums
    premium_time_backoff_steps: int = 3

    # Close pricing tolerance multiplier (for open-position daily marks)
    premium_close_pair_delta_multiplier: float = 5.0
    # Maximum acceptable bid/ask spread as pct of mid when using quotes (0 disables)
    premium_max_spread_pct: float = 0.1
    # Minimum premium delta (points) between front/back legs to allow an entry
    minimum_front_back_spread: float = 0.5

    # Calendar strategy defaults
    calendar_front_dte_default: int = 30
    calendar_back_dte_default: int = 60
    calendar_dte_tolerance_days: int = 30
    calendar_weekday_default: str = "Friday"
    calendar_ff_entry_threshold: float = 0.1
    calendar_ff_exit_threshold: float = 0.0
    calendar_take_profit_pct: float = 0.5
    calendar_stop_loss_pct: float = -0.5
    calendar_max_daily_positions: int = 1
    calendar_ticker_debt_pct: float = 0.1  # debt cap per ticker as pct of total capital
    calendar_position_debt_pct: float = 0.05  # max debt per position as pct of capital
    # calendar_ticker_pool: tuple[str, ...] = ("SPY","SLV","QQQ","IWM","EEM","XLF","GLD","USO","HYG","EFA","FXI","GDX","EWZ","XOP","XLE","VXX","TLT","LQD","UUP","SQQQ",)
    calendar_ticker_pool: tuple[str, ...] = ("GLD","SLV","IBIT","VXX","UVXY","SPXU","SQQQ",)
    # calendar_ticker_pool: tuple[str, ...] = ("AAPL","MSFT","NVDA","AMZN","META","GOOGL","AMD","INTC","MU",)#"BAC","JPM","F","T","XOM","PFE","C","KO","GE",)
    
    calendar_gap_tolerance_pct: float = 0.3
    debug_calendar_timing: bool = False
    debug_backtest_timing: bool = False
    debug_evaluate_candidates: bool = False

    # Rates & dividends
    # Default dividend yield (continuous, annualized). Used if no better estimate available.
    dividend_yield_default: float = 0.0
    # Source for risk-free rate inference (currently 'yahoo' only)
    risk_free_source: str = "yahoo"

    # External API Keys
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")

    # Debug flags (tweak at runtime if needed)
    debug_closure: bool = False            # print per-position close/trigger details
    debug_closure_verbose: bool = False    # include per-leg quotes used for close-cost
    debug_open_sizing: bool = False        # print capital/margin sizing decisions for opens
    debug_plot: bool = False               # print plot input diagnostics
    debug_plot_verbose: bool = False       # more detailed plot diagnostics
    plot_option_pair_price_history: bool = False  # plot leg price history when a pair closes at loss
    debug_iv_solver: bool = False          # enter pdb when IV solver fails
    debug_polygon_quote: bool = False      # print quote poll summary
    debug_iv: bool = False                 # print IV inputs and enter debugger before each solve
    # Calendar pair selection debugging (prints per-ticker reasons when filtered)
    debug_calendar_pair_selection: bool = False
    # Minimum bid/ask size to accept an entry (0 disables the check)
    bid_ask_size_limit: float = 10


# ---------------------------------------------------------
# Field Maps
# ---------------------------------------------------------

PREMIUM_FIELD_MAP = {
    "close": "close_price",
    "mid": "mid_price",
    "trade": "trade_price",
    "quote": "mid_price",
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


def resolve_premium_mode(settings: Settings | None = None) -> str:
    """Return the effective premium mode ('trade', 'quote', 'mid', 'close')."""
    s = settings or get_settings()
    override = (getattr(s, "premium_price_source", "") or "").strip().lower()
    base = (getattr(s, "premium_price_mode", "trade") or "trade").strip().lower()
    mode = override or base
    if mode not in PREMIUM_FIELD_MAP:
        mode = "trade"
    return mode


def resolve_premium_field(settings: Settings | None = None) -> str:
    """Return the payload field to inspect for pricing."""
    mode = resolve_premium_mode(settings)
    return PREMIUM_FIELD_MAP.get(mode, "trade_price")
