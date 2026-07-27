"""Quarantined historical live-order inspection script.

The original scratch utility loaded production OAuth credentials and
constructed pyetrade's order-capable client directly. Read-only order
inspection must use the hardened broker reader once that workflow is exposed.
"""

from live_trading.runtime_safety import reject_legacy_execution


def get_today_orders() -> None:
    """Reject the retired direct-pyetrade order path."""

    reject_legacy_execution("scratch.check_orders.get_today_orders")


if __name__ == "__main__":
    get_today_orders()
