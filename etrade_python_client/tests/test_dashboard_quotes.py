import unittest
from types import SimpleNamespace

from accounts.accounts_bo import (
    Accounts,
    _extract_cboe_vix_closes,
    _market_quote_price,
    _missing_market_close_dates,
    _quote_metadata,
)


class _Response:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.request = SimpleNamespace(headers={})

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload=None, responses=None):
        self.payload = payload
        self.responses = list(responses or [])
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return _Response(self.payload)


class DashboardQuoteTests(unittest.TestCase):
    def test_market_quote_returns_price_and_freshness_metadata(self):
        session = _Session({
            "QuoteResponse": {
                "QuoteData": [{
                    "Product": {"symbol": "SPX", "securityType": "INDX"},
                    "quoteStatus": "REALTIME",
                    "dateTimeUTC": 1_784_300_000,
                    "All": {
                        "lastTrade": 7_491.35,
                        "timeOfLastTrade": 1_784_299_995,
                    },
                }]
            }
        })
        accounts = Accounts(session, "https://api.etrade.com", consumer_key="")

        prices, metadata = accounts.get_stock_prices(["SPX"], include_metadata=True)

        self.assertEqual(prices, {"SPX": 7_491.35})
        self.assertEqual(metadata["SPX"]["source"], "E*TRADE")
        self.assertEqual(metadata["SPX"]["status"], "REALTIME")
        self.assertEqual(metadata["SPX"]["timestamp"], 1_784_299_995)
        self.assertEqual(session.calls[0][1]["params"], {"detailFlag": "ALL"})

    def test_quote_timestamp_accepts_milliseconds_and_preserves_delay_status(self):
        metadata = _quote_metadata(
            {
                "quoteStatus": "DELAYED",
                "All": {"timeOfLastTrade": 1_784_299_995_000},
            },
            7_480.25,
        )

        self.assertEqual(metadata["timestamp"], 1_784_299_995)
        self.assertEqual(metadata["status"], "DELAYED")

    def test_market_quote_uses_the_newest_regular_or_extended_trade(self):
        regular_is_newer = {
            "All": {
                "lastTrade": 7_491.35,
                "timeOfLastTrade": 1_784_300_000,
                "ehQuote": {
                    "lastPrice": 7_480.10,
                    "timeOfLastTrade": 1_784_200_000,
                },
            }
        }
        extended_is_newer = {
            "All": {
                "lastTrade": 201.25,
                "timeOfLastTrade": 1_784_300_000,
                "ehQuote": {
                    "lastPrice": 202.10,
                    "timeOfLastTrade": 1_784_300_500,
                },
            }
        }

        self.assertEqual(_market_quote_price(regular_is_newer), 7_491.35)
        self.assertEqual(_market_quote_price(extended_is_newer), 202.10)

    def test_vix_only_cache_gap_triggers_market_history_refresh(self):
        price_cache = {
            "SPY": {"2026-07-22": 747.41, "2026-07-23": 738.18},
            "SPX": {"2026-07-22": 7_498.96, "2026-07-23": 7_408.30},
            "VIX": {"2026-07-22": 16.50},
        }

        missing_dates = _missing_market_close_dates(
            ["2026-07-22", "2026-07-23", "2026-07-24 (Live)"],
            price_cache,
            today_str="2026-07-24",
            market_active=True,
        )

        self.assertEqual(missing_dates, ["2026-07-23"])

    def test_cboe_vix_history_parser_returns_requested_closes(self):
        csv_text = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "07/22/2026,17.42,19.49,16.64,16.64\n"
            "07/23/2026,17.67,20.31,17.32,18.70\n"
        )

        closes = _extract_cboe_vix_closes(csv_text, ["2026-07-23"])

        self.assertEqual(closes, {"2026-07-23": 18.70})

    def test_portfolio_retries_after_expired_token(self):
        expired_session = _Session(responses=[
            _Response({}, status_code=401, text="oauth_problem=token_expired"),
        ])
        refreshed_session = _Session({
            "PortfolioResponse": {
                "AccountPortfolio": [],
            }
        })
        accounts = Accounts(expired_session, "https://api.etrade.test", consumer_key="")
        accounts.account = {"accountIdKey": "account-key"}
        accounts.auth_refresh_callback = lambda reason: (
            refreshed_session,
            "https://api.etrade.test",
        )

        positions = accounts.portfolio(require_success=True)

        self.assertEqual(positions, [])
        self.assertEqual(len(expired_session.calls), 1)
        self.assertEqual(len(refreshed_session.calls), 1)

    def test_required_portfolio_fetch_raises_instead_of_returning_false_empty(self):
        session = _Session(responses=[
            _Response({}, status_code=401, text="oauth_problem=token_rejected"),
        ])
        accounts = Accounts(session, "https://api.etrade.test", consumer_key="")
        accounts.account = {"accountIdKey": "account-key"}

        with self.assertRaisesRegex(RuntimeError, "status code 401"):
            accounts.portfolio(require_success=True)

    def test_consumer_key_is_required_only_for_calls_that_use_its_header(self):
        session = _Session({})
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )

        with self.assertRaisesRegex(RuntimeError, "Missing E\\*TRADE production consumer key"):
            accounts.check_earning_date(SimpleNamespace(symbol="SPY"))

        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
