"""
Massive API (formerly Polygon.io) configuration.
Loads API key from environment, .env file, or falls back to legacy config.
"""
import os


def _offline_only_enabled() -> bool:
    return os.environ.get("MASSIVE_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def get_api_key() -> str:
    """Load API key with cascading fallback."""
    if _offline_only_enabled():
        return ""

    # 1. Check environment variable
    key = os.environ.get("MASSIVE_API_KEY")
    if key:
        return key

    # 2. Try dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("MASSIVE_API_KEY")
        if key:
            return key
    except ImportError:
        pass

    # 3. Fallback to legacy Polygon config
    try:
        from data_and_research import polygonio_config
        return polygonio_config.API_KEY
    except ImportError:
        raise RuntimeError(
            "No Massive/Polygon API key found. "
            "Set MASSIVE_API_KEY env var or configure polygonio_config.py"
        )


API_KEY = get_api_key()
BASE_URL = "https://api.polygon.io"
