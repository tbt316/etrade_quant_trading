        print("\n─── Phase 4: Executing strategy sequentially ───")
        current_cash = initial_capital
        active_trades: List[SpreadTrade] = []
        previously_panic = False
        
        for td, candidates in trade_plan:
            if td not in underlying_prices.index:
                continue
                
            spot = underlying_prices.loc[td]
            daily_events = []
            
            # 0. Track Regime
            current_regime = regimes.get(td, -1) if regimes else -1
            result.regime_history.append((td, current_regime))
            
            current_regime_name = result.regime_labels.get(current_regime, "")
            is_panic = "Panic / Crisis" in current_regime_name
            
            # 1. Update active trades (Exits)
            still_active = []
            closed_today = []
            daily_total_unrealized = 0.0
            
            # Pre-fetch all needed quotes for active trades in parallel
            unique_tickers = set()
            for trade in active_trades:
                unique_tickers.add(trade.short_ticker)
                unique_tickers.add(trade.long_ticker)
            
            quotes_map = {}
            if unique_tickers:
                quote_tasks = {ticker: client.fetch_eod_quote(ticker, td) for ticker in unique_tickers}
                quote_results = await asyncio.gather(*quote_tasks.values(), return_exceptions=True)
                quotes_map = dict(zip(quote_tasks.keys(), quote_results))

            for trade in active_trades:
                exp_dt = datetime.strptime(trade.expiration, "%Y-%m-%d")
                curr_dt = datetime.strptime(td, "%Y-%m-%d")
                current_dte = (exp_dt - curr_dt).days
                
                short_ticker = trade.short_ticker
                long_ticker = trade.long_ticker
                
                short_cached = cache.get_ohlcv(short_ticker, td)
                long_cached = cache.get_ohlcv(long_ticker, td)
                
                exit_triggered = False
                exit_reason = ""
                
                short_quote = quotes_map.get(short_ticker)
                long_quote = quotes_map.get(long_ticker)
                if isinstance(short_quote, Exception): short_quote = None
                if isinstance(long_quote, Exception): long_quote = None

                if short_quote and long_quote:
                    current_short_mid = short_quote["mid"]
                    current_long_mid = long_quote["mid"]
                else:
                    # Data Gap Handling
                    result.data_gap_count += 1
                    current_short_mid = short_cached.get("close", 0) if short_cached else 0
                    current_long_mid = long_cached.get("close", 0) if long_cached else 0
                    
                    if not current_short_mid or not current_long_mid:
                        result.critical_gap_count += 1
                        current_short_mid = trade.short_entry_mid
                        current_long_mid = trade.long_entry_mid
                
                current_debit = current_short_mid - current_long_mid
                gain_pct = (trade.net_credit - current_debit) / trade.net_credit if trade.net_credit > 0 else 0
                    
                if not trade.hold_to_expiration:
                    # Early profit exit
                    if gain_pct >= trade.profit_target:
                        exit_triggered = True
                        exit_reason = "early_profit"
                
                    # Scheduled exit
                    is_panic_trade = "Panic / Crisis" in trade.entry_regime_name
                    if not is_panic_trade and current_dte <= close_dte and not exit_triggered:
                        # NEW: Only exit at half-DTE if the position is in profit.
                        # If in loss, skip and follow profit target (or hold to expiration).
                        if gain_pct > 0:
                            if hold_itm_to_expiration and spot <= trade.short_strike:
                                trade.hold_to_expiration = True
                            else:
                                exit_triggered = True
                                exit_reason = "scheduled"
                
                if current_dte == 0 and not exit_triggered:
                    exit_triggered = True
                    exit_reason = "expired"
                    
                if exit_triggered:
                    trade.exit_date = td
                    trade.short_exit_mid = current_short_mid
                    trade.net_debit_close = current_debit
                    trade.pnl_per_share = trade.net_credit - trade.net_debit_close
                    trade.pnl_per_contract = trade.pnl_per_share * 100
                    trade.exit_dte = current_dte
                    trade.status = "closed"
                    trade.exit_reason = exit_reason
                    
                    current_cash -= trade.net_debit_close * 100 * trade.num_contracts
                    closed_today.append(trade)
                    print(f"  {td}: CLOSE {trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p | PnL=${trade.pnl_per_contract*trade.num_contracts:,.0f}")
                    
                    if path_logger:
                        daily_events.append({"type": "close", "reason": exit_reason, "trade": asdict(trade)})
                else:
                    still_active.append(trade)
                    daily_total_unrealized += current_debit * 100 * trade.num_contracts
            
            active_trades = still_active
            result.trades.extend(closed_today)
            
            # 2. Portfolio Update
            current_margin_usage = sum(t.margin_required for t in active_trades)
            current_nlv = current_cash - daily_total_unrealized
            result.cash_history.append((td, current_cash))
            result.nvl_history.append((td, current_nlv))
            result.margin_history.append((td, current_margin_usage))
            
            # 3. Entry Logic
            allowed_margin = current_nlv * margin_limit_pct
            if current_margin_usage < allowed_margin:
                eff_short_delta = target_short_delta
                eff_spread_width = spread_width
                eff_qty_multiplier = 1.0
                eff_profit_target = early_profit_pct
                eff_target_dte = target_dte
                
                if dynamic_delta_variant and is_panic:
                    eff_short_delta = max(-0.95, min(-0.01, target_short_delta * panic_delta_multiplier))
                    eff_spread_width = spread_width * panic_width_multiplier
                    eff_qty_multiplier = panic_qty_multiplier
                    eff_target_dte = panic_dte_target
                    print(f"  {td}: [PANIC MODE] Targets adjusted: Delta={eff_short_delta:.2f}, Width=${eff_spread_width:.1f}")

                selected_exp = None
                chain_data = None
                possible_exps = find_target_expiration_friday(td, eff_target_dte)
                for exp in possible_exps:
                    if exp in contracts_by_exp:
                        chain = cache.get_chain_for_date(underlying, exp, "put", td)
                        if chain and len(chain) >= 10:
                            selected_exp = exp
                            chain_data = chain
                            break

                if selected_exp:
                    exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
                    dte_days = (exp_dt - datetime.strptime(td, "%Y-%m-%d")).days
                    dte_years = dte_days / 365.0

                    strikes = [r["strike"] for r in chain_data]
                    close_prices = [r["close"] for r in chain_data]
                    valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]
                    
                    if len(valid) >= 5:
                        v_strikes, v_prices = zip(*valid)
                        delta_chain = compute_chain_deltas(spot, list(v_strikes), list(v_prices), dte_years, risk_free_rate, "put")
                        
                        if delta_chain:
                            short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][1] - eff_short_delta))
                            short_strike = delta_chain[short_idx][0]
                            short_delta = delta_chain[short_idx][1]
                            long_strike = short_strike - eff_spread_width
                            
                            # Find long mid
                            long_mid = next((r["close"] for r in chain_data if abs(r["strike"] - long_strike) < 0.01), 0.0)
                            if not long_mid:
                                nearest_long = min(strikes, key=lambda s: abs(s - long_strike))
                                long_mid = next((r["close"] for r in chain_data if abs(r["strike"] - nearest_long) < 0.01), 0.0)
                                long_strike = nearest_long
                            
                            short_mid = delta_chain[short_idx][2]
                            net_credit = short_mid - long_mid
                            
                            if net_credit > 0:
                                margin_per_lot = (short_strike - long_strike) * 100
                                if backtest_qty > 0:
                                    num_contracts = int(backtest_qty * eff_qty_multiplier) or 1
                                else:
                                    num_contracts = int((allowed_margin - current_margin_usage) // margin_per_lot)
                                
                                if num_contracts > 0:
                                    trade = SpreadTrade(
                                        short_ticker=next(r["option_ticker"] for r in chain_data if abs(r["strike"] - short_strike) < 0.01),
                                        long_ticker=next(r["option_ticker"] for r in chain_data if abs(r["strike"] - long_strike) < 0.01),
                                        short_strike=short_strike,
                                        long_strike=long_strike,
                                        expiration=selected_exp,
                                        entry_date=td,
                                        entry_price=spot,
                                        net_credit=net_credit,
                                        num_contracts=num_contracts,
                                        margin_required=margin_per_lot * num_contracts,
                                        status="open",
                                        entry_regime=current_regime,
                                        entry_regime_name=current_regime_name,
                                        entry_dte=dte_days,
                                        short_entry_mid=short_mid,
                                        long_entry_mid=long_mid,
                                        profit_target=eff_profit_target
                                    )
                                    
                                    active_trades.append(trade)
                                    result.trades.append(trade)
                                    current_cash += net_credit * 100 * num_contracts
                                    current_margin_usage += trade.margin_required
                                    
                                    result.daily_leg_premiums.append((td, short_mid, long_mid))
                                    print(f"  {td}: OPEN {num_contracts}x {short_strike}/{long_strike}p | Credit=${net_credit:.2f} | Margin=${trade.margin_required:,.0f}")

                                    if path_logger:
                                        chain_snapshot = []
                                        for r in chain_data:
                                            d_val = next((d[1] for d in delta_chain if abs(d[0] - r["strike"]) < 0.01), 0.0)
                                            if abs(r["strike"] - short_strike) < 30:
                                                chain_snapshot.append({
                                                    "strike": float(r["strike"]),
                                                    "close": float(r["close"]) if r["close"] else 0.0,
                                                    "delta": float(d_val)
                                                })
                                        
                                        daily_events.append({
                                            "type": "entry",
                                            "expiration": selected_exp,
                                            "short_strike": short_strike,
                                            "long_strike": long_strike,
                                            "net_credit": float(net_credit),
                                            "num_contracts": num_contracts,
                                            "chain_snapshot": chain_snapshot
                                        })
            
            # ALWAYS Log Day
            if path_logger:
                path_logger.log_day(
                    date_str=td,
                    spot=spot,
                    regime=current_regime,
                    regime_name=current_regime_name or f"Regime {current_regime}",
                    nlv=current_nlv,
                    cash=current_cash,
                    margin=current_margin_usage,
                    active_trades=active_trades,
                    events=daily_events,
                    abnormalities=[a for a in result.abnormalities if a["date"] == td]
                )
            
            previously_panic = is_panic
