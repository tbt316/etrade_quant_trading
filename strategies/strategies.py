from __future__ import annotations
"""
Strategy layer
==============

This module centralizes the *shape* of each strategy and which option sides
(calls/puts) are needed. It does **not** change how you pick strikes or
premiums — keep using your existing selection logic. You pass those selected
strikes/premiums here and we return a normalized position dict and margin.

That mirrors what your old `backtest_options_sync_or_async(...)` did with
`trade_type` branches, but in a clean, testable module.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Tuple

Side = Literal["call", "put"]


# -------------------------
# Data structures
# -------------------------

@dataclass
class Leg:
    side: Side                # "call" | "put"
    action: Literal["sell", "buy"]
    strike: float
    premium: float            # positive number; sign handling is done in PnL later


@dataclass
class Position:
    trade_type: str
    underlying: str
    expiration: str           # YYYY-MM-DD
    opened_at: str            # pricing/as-of date YYYY-MM-DD
    qty: int                  # contracts multiplier (positive = short net credit positions use +qty)
    legs: List[Leg]
    required_margin: float    # in dollars

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["legs"] = [asdict(l) for l in self.legs]
        return d


# -------------------------
# Helpers (pure)
# -------------------------

def _vertical_margin(short_strike: float, long_strike: float, contract_size: int = 100) -> float:
    """Max loss for a 1-lot vertical = (|short-long| * 100).
    Caller should multiply by quantity outside if needed. Our Position.required_margin
    is typically per *position* at given qty, so we multiply by qty when building.
    """
    width = abs(short_strike - long_strike)
    return width * contract_size


def _iron_condor_margin(
    short_call: float,
    long_call: float,
    short_put: float,
    long_put: float,
    contract_size: int = 100,
) -> float:
    """IC margin = max(call spread width, put spread width) * 100 (per contract)."""
    call_w = abs(short_call - long_call)
    put_w = abs(short_put - long_put)
    return max(call_w, put_w) * contract_size


# -------------------------
# Strategy interfaces
# -------------------------

class Strategy:
    name: str

    def sides_needed(self) -> List[Side]:
        raise NotImplementedError

    def build_position(
        self,
        *,
        underlying: str,
        expiration: str,
        opened_at: str,
        qty: int,
        # Provide whichever legs are relevant for the strategy; unused may be None
        short_call: Optional[Tuple[float, float]] = None,  # (strike, premium)
        long_call: Optional[Tuple[float, float]] = None,
        short_put: Optional[Tuple[float, float]] = None,
        long_put: Optional[Tuple[float, float]] = None,
    ) -> Position:
        raise NotImplementedError


# -------------------------
# Concrete strategies (logic mirrors your original shape)
# -------------------------

class IronCondorStrategy(Strategy):
    name = "iron_condor"

    def sides_needed(self) -> List[Side]:
        return ["call", "put"]

    def build_position(
        self,
        *,
        underlying: str,
        expiration: str,
        opened_at: str,
        qty: int,
        short_call: Tuple[float, float],
        long_call: Tuple[float, float],
        short_put: Tuple[float, float],
        long_put: Tuple[float, float],
    ) -> Position:
        sc_k, sc_p = short_call
        lc_k, lc_p = long_call
        sp_k, sp_p = short_put
        lp_k, lp_p = long_put

        legs = [
            Leg("call", "sell", sc_k, sc_p),
            Leg("call", "buy",  lc_k, lc_p),
            Leg("put",  "sell", sp_k, sp_p),
            Leg("put",  "buy",  lp_k, lp_p),
        ]
        margin_one = _iron_condor_margin(sc_k, lc_k, sp_k, lp_k)
        return Position(
            trade_type=self.name,
            underlying=underlying,
            expiration=expiration,
            opened_at=opened_at,
            qty=qty,
            legs=legs,
            required_margin=margin_one * qty,
        )


class PutCreditSpreadStrategy(Strategy):
    name = "put_credit_spread"

    def sides_needed(self) -> List[Side]:
        return ["put"]

    def build_position(
        self,
        *,
        underlying: str,
        expiration: str,
        opened_at: str,
        qty: int,
        short_put: Tuple[float, float],
        long_put: Tuple[float, float],
    ) -> Position:
        sp_k, sp_p = short_put
        lp_k, lp_p = long_put
        legs = [
            Leg("put", "sell", sp_k, sp_p),
            Leg("put", "buy",  lp_k, lp_p),
        ]
        margin_one = _vertical_margin(sp_k, lp_k)
        return Position(
            trade_type=self.name,
            underlying=underlying,
            expiration=expiration,
            opened_at=opened_at,
            qty=qty,
            legs=legs,
            required_margin=margin_one * qty,
        )


class CallCreditSpreadStrategy(Strategy):
    name = "call_credit_spread"

    def sides_needed(self) -> List[Side]:
        return ["call"]

    def build_position(
        self,
        *,
        underlying: str,
        expiration: str,
        opened_at: str,
        qty: int,
        short_call: Tuple[float, float],
        long_call: Tuple[float, float],
    ) -> Position:
        sc_k, sc_p = short_call
        lc_k, lc_p = long_call
        legs = [
            Leg("call", "sell", sc_k, sc_p),
            Leg("call", "buy",  lc_k, lc_p),
        ]
        margin_one = _vertical_margin(sc_k, lc_k)
        return Position(
            trade_type=self.name,
            underlying=underlying,
            expiration=expiration,
            opened_at=opened_at,
            qty=qty,
            legs=legs,
            required_margin=margin_one * qty,
        )


class CoveredCallStrategy(Strategy):
    name = "covered_call"

    def sides_needed(self) -> List[Side]:
        return ["call"]

    def build_position(
        self,
        *,
        underlying: str,
        expiration: str,
        opened_at: str,
        qty: int,
        short_call: Tuple[float, float],
        long_call: Optional[Tuple[float, float]] = None,
    ) -> Position:
        """Covered call is modeled as a short call leg plus underlying shares held externally.
        In your original code, the long leg for CC was set equal to the short leg (no width) to
        keep the data structure uniform; margin is typically handled by broker as covered.
        Here we set required_margin to 0 for the option piece; any stock margin is managed elsewhere.
        """
        sc_k, sc_p = short_call
        legs = [Leg("call", "sell", sc_k, sc_p)]
        return Position(
            trade_type=self.name,
            underlying=underlying,
            expiration=expiration,
            opened_at=opened_at,
            qty=qty,
            legs=legs,
            required_margin=0.0,
        )


# -------------------------
# Factory / registry
# -------------------------

def get_strategy(trade_type: str) -> Strategy:
    t = (trade_type or "").lower()
    if t == "iron_condor":
        return IronCondorStrategy()
    if t in {"pcs", "put_credit_spread"}:
        return PutCreditSpreadStrategy()
    if t in {"ccs", "call_credit_spread"}:
        return CallCreditSpreadStrategy()
    if t in {"cc", "covered_call"}:
        return CoveredCallStrategy()
    raise ValueError(f"Unknown trade_type: {trade_type}")


# -------------------------
# Convenience: which sides to fetch (drop-in replacement)
# -------------------------

def sides_for_trade_type(trade_type: str) -> List[Side]:
    """Match your original branching used to decide call/put fetching.
    - iron_condor -> [call, put]
    - put_credit_spread -> [put]
    - call_credit_spread / covered_call -> [call]
    """
    return get_strategy(trade_type).sides_needed()

