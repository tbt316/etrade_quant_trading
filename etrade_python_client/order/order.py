"""Quarantined compatibility surface for the obsolete interactive order client."""

from __future__ import annotations

from live_trading.runtime_safety import reject_legacy_execution


class Order:
    """Compatibility name retained for imports; construction always fails closed."""

    def __init__(self, *args, **kwargs):
        """Reject construction before storing or inspecting caller data."""

        reject_legacy_execution("order.order.Order.__init__")
