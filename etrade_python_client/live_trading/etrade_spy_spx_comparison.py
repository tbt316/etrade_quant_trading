import os
import sys
import json
from datetime import datetime, timedelta

# adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accounts.accounts_bo import Accounts
from live_trading.etrade_cover_call_new import oauth

TARGET_PREMIUM_DOLLARS = 1000.0
TARGET_SHORT_DELTA = -0.15
TARGET_LONG_DELTA = -0.05

def fetch_puts(accounts, symbol, expiration_date, price_near):
    url = f"{accounts.base_url}/v1/market/optionchains.json"
    params = {
        "symbol": symbol,
        "expiryYear": expiration_date.year,
        "expiryMonth": expiration_date.month,
        "expiryDay": expiration_date.day,
        "includeWeekly": True,
        "optionCategory": "ALL",
        "chainType": "PUT",
        "strikePriceNear": price_near,
        "noOfStrikes": 400,
    }
    response = accounts.session.get(url, params=params)
    if response.status_code != 200: return []
    data = response.json()
    pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
    
    puts = []
    for pair in pairs:
        opt = pair.get("Put")
        if opt:
            greeks = opt.get("OptionGreeks", {})
            delta = float(greeks.get("delta", 0.0)) if greeks.get("delta") is not None else 0.0
            if delta < 0:
                bid = float(opt.get("bid"))
                ask = float(opt.get("ask"))
                puts.append({
                    "strike": float(opt.get("strikePrice")),
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "spread": ask - bid,
                    "delta": delta
                })
    return sorted(puts, key=lambda x: x["strike"], reverse=True)

def get_closest_by_delta(options, target_delta):
    if not options: return None
    return min(options, key=lambda x: abs(x["delta"] - target_delta))

def calculate_ev(net_credit, spread_width, short_delta, long_delta):
    prob_full_profit = 1.0 - abs(short_delta)
    prob_max_loss = abs(long_delta)
    prob_partial = abs(short_delta) - abs(long_delta)
    
    max_risk = spread_width - net_credit
    avg_partial_loss = (spread_width / 2.0) - net_credit 
    
    ev = (prob_full_profit * net_credit) - (prob_max_loss * max_risk) - (prob_partial * avg_partial_loss)
    return ev * 100 

def main():
    use_sandbox = False
    print("Authenticating with E*TRADE...")
    session, base_url = oauth(use_sandbox, auto_login=False) 
    if not session:
        print("Failed to authenticate.")
        return

    accounts = Accounts(session, base_url)
    
    spy_ticker = "SPY"
    spx_ticker = "SPX"
    
    print("\nFetching stock prices...")
    spy_price = accounts.get_stock_price(spy_ticker)
    spx_price = accounts.get_stock_price(spx_ticker)

    spx_exps = accounts.get_available_expirations(spx_ticker)
    if not spx_exps:
        spx_ticker = "SPXW"

    target_exp = "2026-05-29"
    exp_date_obj = datetime.strptime(target_exp, "%Y-%m-%d")

    print(f"Fetching options chain centered slightly OTM for {target_exp}...")
    
    spy_options = fetch_puts(accounts, spy_ticker, exp_date_obj, spy_price * 0.90)
    spx_options = fetch_puts(accounts, spx_ticker, exp_date_obj, spx_price * 0.90)

    # Clean out quotes without valid asks (prevent weird edge case fills)
    spy_options = [o for o in spy_options if o["ask"] > 0]
    spx_options = [o for o in spx_options if o["ask"] > 0]

    # Baseline SPY and SPX leg assignment via strict Delta targeting (-0.15 short, -0.05 long)
    spy_short_opt = get_closest_by_delta(spy_options, TARGET_SHORT_DELTA)
    spx_short_opt = get_closest_by_delta(spx_options, TARGET_SHORT_DELTA)

    spy_long_opt = get_closest_by_delta(spy_options, TARGET_LONG_DELTA)
    spx_long_opt = get_closest_by_delta(spx_options, TARGET_LONG_DELTA)

    if not spy_short_opt or not spx_short_opt or not spy_long_opt or not spx_long_opt:
        print("Failed to find valid legs for comparison!")
        return

    print("\n" + "="*60)
    print("  DELTA-MATCHED SPREAD COMPARISON (USING MIDPOINT PRICING)  ")
    print("="*60)

    # Calculate using proper midpoints 
    spy_net_credit = max(0, spy_short_opt["mid"] - spy_long_opt["mid"])
    spx_net_credit = max(0, spx_short_opt["mid"] - spx_long_opt["mid"])
    
    spy_spread_width = spy_short_opt["strike"] - spy_long_opt["strike"]
    spx_spread_width = spx_short_opt["strike"] - spx_long_opt["strike"]

    spy_margin_per_contract = spy_spread_width * 100
    spx_margin_per_contract = spx_spread_width * 100

    spy_ev = calculate_ev(spy_net_credit, spy_spread_width, spy_short_opt["delta"], spy_long_opt["delta"])
    spx_ev = calculate_ev(spx_net_credit, spx_spread_width, spx_short_opt["delta"], spx_long_opt["delta"])

    spy_qty = max(1, round(TARGET_PREMIUM_DOLLARS / (spy_net_credit * 100))) if spy_net_credit > 0 else 0
    spx_qty = max(1, round(TARGET_PREMIUM_DOLLARS / (spx_net_credit * 100))) if spx_net_credit > 0 else 0

    print(f"Goal Target Premium: ~${TARGET_PREMIUM_DOLLARS}\n")

    print(f"--- SPY Credit Spread ---")
    print(f"Width          : ${spy_spread_width:.2f}")
    print(f"Short Leg [{spy_short_opt['strike']}] : Bid ${spy_short_opt['bid']:.2f} / Ask ${spy_short_opt['ask']:.2f} (Spread: ${spy_short_opt['spread']:.2f}) -> Mid: ${spy_short_opt['mid']:.2f} [Delta {spy_short_opt['delta']:.4f}]")
    print(f"Long Leg  [{spy_long_opt['strike']}] : Bid ${spy_long_opt['bid']:.2f} / Ask ${spy_long_opt['ask']:.2f} (Spread: ${spy_long_opt['spread']:.2f}) -> Mid: ${spy_long_opt['mid']:.2f} [Delta {spy_long_opt['delta']:.4f}]")
    print(f"Net Credit     : ${spy_net_credit:.2f} / contract (Midpoint)")
    print(f"Estimated EV   : ${spy_ev:.2f} / contract")
    print(f"Contract Qty   : {spy_qty}")
    print(f"Total Premium  : ${spy_qty * spy_net_credit * 100:.2f}")
    print(f"Total Portfolio EV: ${spy_qty * spy_ev:.2f}")
    print(f"Required Margin: ${spy_qty * spy_margin_per_contract:,.2f}")
    
    print(f"\n--- {spx_ticker} Credit Spread ---")
    print(f"Width          : ${spx_spread_width:.2f}")
    print(f"Short Leg [{spx_short_opt['strike']}] : Bid ${spx_short_opt['bid']:.2f} / Ask ${spx_short_opt['ask']:.2f} (Spread: ${spx_short_opt['spread']:.2f}) -> Mid: ${spx_short_opt['mid']:.2f} [Delta {spx_short_opt['delta']:.4f}]")
    print(f"Long Leg  [{spx_long_opt['strike']}] : Bid ${spx_long_opt['bid']:.2f} / Ask ${spx_long_opt['ask']:.2f} (Spread: ${spx_long_opt['spread']:.2f}) -> Mid: ${spx_long_opt['mid']:.2f} [Delta {spx_long_opt['delta']:.4f}]")
    print(f"Net Credit     : ${spx_net_credit:.2f} / contract (Midpoint)")
    print(f"Estimated EV   : ${spx_ev:.2f} / contract")
    print(f"Contract Qty   : {spx_qty}")
    print(f"Total Premium  : ${spx_qty * spx_net_credit * 100:.2f}")
    print(f"Total Portfolio EV: ${spx_qty * spx_ev:.2f}")
    print(f"Required Margin: ${spx_qty * spx_margin_per_contract:,.2f}")
    
    if spy_qty > 0 and spx_qty > 0:
        print("\n--- Efficiency ---")
        print(f"SPY Margin / Premium Ratio: { (spy_qty * spy_margin_per_contract) / (spy_qty * spy_net_credit * 100):.2f}")
        print(f"SPX Margin / Premium Ratio: { (spx_qty * spx_margin_per_contract) / (spx_qty * spx_net_credit * 100):.2f}")

if __name__ == "__main__":
    main()
