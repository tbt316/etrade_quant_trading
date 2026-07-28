# Historical Mark Evidence Boundary

The backtester has two explicit historical mark policies:

- `strict_nbbo` accepts only an observed Massive exact-contract quote with
  finite `bid > 0`, `ask >= bid`, an explicit UTC event timestamp inside the
  requested regular NYSE session, exact request ticker binding, and a quote no
  older than `execution.max_quote_age_minutes` at the exact exchange-calendar
  close. Calendar-derived closes include early-close sessions.
- `research_fallback` may use an explicit `OBSERVED_NBBO`,
  `SYNCHRONIZED_MINUTE_AGGREGATE`, `TRADE_PRINT`, `THEORETICAL`, or
  `DAILY_CLOSE` source. An older in-session observed quote may remain a
  research mark, but diagnostics record its age and strict validation remains
  false.

Every mark bundle must contain exactly two or three unique contracts from one
pricing date. Multi-leg timestamps must independently fit inside
`execution.max_time_delta_minutes`. Missing, close-stale, crossed, non-finite,
wrong-day, wrong-contract, duplicate-contract, mixed-date, post-close, or
time-skewed evidence fails strict mark validation.

Strict NBBO is not an executable-fill claim. The provider quote can have zero
displayed size, the modeled order quantity can exceed displayed size, and the
default model uses a midpoint. Diagnostics therefore expose
`strict_nbbo_mark_validated`; the deprecated `strict_nbbo_authorized` field is
always false. `HISTORICAL_MARK_NOT_EXECUTABLE_FILL` is persisted with the
assumption. Trade records follow the same rule with
`entry_fill_strict_nbbo_mark_validated` and
`exit_fill_strict_nbbo_mark_validated`; their legacy `*_authorized` fields are
always false. `historical_execution_proven`, `entry_execution_proven`, and
`exit_execution_proven` are also always false.

The legacy `option_prices` cache does not retain a quote/trade event timestamp
or source. Its `bid`, `ask`, `mid`, and `is_synchronized` columns therefore
cannot support strict mark validation and are not interpreted as a quote,
trade, theoretical price, or synchronized aggregate. Its unambiguous OHLCV
`close` column may be used only as an explicitly labeled `DAILY_CLOSE`
research fallback.

Use `execution.entry_fill_mode` and `execution.exit_fill_mode` when entry and
exit mark policies differ. `execution.fill_mode` sets both and conflicts with
different explicit entry/exit values. The standalone runner supports
`--fill-mode`, `--entry-fill-mode`, and `--exit-fill-mode`.
`execution.require_entry_nbbo` is rejected, even when false, because it
silently left exit semantics ambiguous. Migrate it as follows:

```yaml
execution:
  entry_fill_mode: strict_nbbo
  exit_fill_mode: research_fallback  # or strict_nbbo, explicitly chosen
```

Set `execution.max_time_delta_minutes` to bind multi-leg synchronization (the
older `entry.max_time_delta_minutes` location remains readable for
compatibility). Set `execution.max_quote_age_minutes` to bind close freshness.
Both defaults are five minutes; quote age must be positive and finite, while a
zero leg-skew threshold is allowed.

For programmatic callers, an explicit function argument has highest
precedence, followed by `strategy_config.execution`, followed by the legacy
`strategy_config.entry.max_time_delta_minutes` location for leg skew, then the
five-minute default. The standalone CLI follows the same rule:
`--max-time-delta-minutes` and `--max-quote-age-minutes` explicitly override
the loaded strategy configuration. The underscore spellings remain accepted
for compatibility.

The generic theoretical mark is contract-type aware: put marks use the
Black-Scholes put function and call marks use the call function. Target and
reference ticker flags must match the explicit option type.

The Massive client receives parsed JSON and does not retain exact raw provider
response bytes. Mark diagnostics disclose
`NO_RAW_PROVIDER_BYTES_RETAINED`; this boundary proves in-process validation,
not durable provider replay or execution.

Expiration uses `EXPIRATION_CLOSE_MARK_RESEARCH_ONLY`, calculated from the
same-day underlying close solely to preserve exploratory backtest continuity.
It does not claim settlement. Contract settlement style and official reference
are `UNVERIFIED`, with AM-settlement risk called out for index products such as
SPX. Each such close records a critical abnormality and never sets strict NBBO
mark validation.

Overall backtest `causal_validity` remains `UNVERIFIED` pending provider
entitlement and availability proof, raw-response retention/parser replay,
complete OHLCV range truth, quantity-aware execution assumptions, and verified
expiration settlement style/reference.
