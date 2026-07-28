import unittest
from datetime import date
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
        self.content = text.encode("utf-8")
        self.request = SimpleNamespace(headers={})

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload=None, responses=None):
        self.payload = payload
        self.responses = list(responses or [])
        self.calls = []
        self.auth = None

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
        refreshed_session = _Session(responses=[
            _Response(None, status_code=204),
        ])
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

    def test_required_portfolio_rejects_malformed_success_payload(self):
        session = _Session({
            "PortfolioResponse": {},
        })
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "snapshot is incomplete",
        ):
            accounts.portfolio(require_success=True)

    def test_portfolio_accepts_only_empty_first_page_204_as_no_positions(self):
        session = _Session(responses=[
            _Response(None, status_code=204),
        ])
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        self.assertEqual(accounts.portfolio(), [])

        ambiguous = _Session({
            "PortfolioResponse": {
                "AccountPortfolio": [],
            },
        })
        accounts.session = ambiguous
        with self.assertRaisesRegex(
            RuntimeError,
            "account-level portfolio proof",
        ):
            accounts.portfolio()

        accounts.session = _Session(responses=[
            _Response(
                None,
                status_code=204,
                text="unexpected body",
            ),
        ])
        with self.assertRaisesRegex(
            RuntimeError,
            "empty first-page HTTP 204",
        ):
            accounts.portfolio()

        accounts.session = _Session(responses=[
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalNoOfPages": 2,
                        "nextPageNo": "2",
                        "Position": [{
                            "positionId": 11,
                            "Product": {
                                "symbol": "SPY",
                                "securityType": "EQ",
                            },
                            "quantity": 1,
                        }],
                    }],
                },
            }),
            _Response(None, status_code=204),
        ])
        with self.assertRaisesRegex(
            RuntimeError,
            "empty first-page HTTP 204",
        ):
            accounts.portfolio()

    def test_required_portfolio_uses_explicit_pagination_metadata(self):
        position = {
            "positionId": "11",
            "Product": {
                "symbol": "SPY",
                "securityType": "EQ",
            },
            "quantity": 1,
            "Complete": {
                "price": 700.0,
                "adjPrice": 700.0,
            },
            "marketValue": 700.0,
            "positionType": "LONG",
        }
        session = _Session(responses=[
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalNoOfPages": 2,
                        "nextPageNo": "2",
                        "Position": [position],
                    }],
                },
            }),
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalNoOfPages": 2,
                        "Position": [{
                            **position,
                            "positionId": "12",
                        }],
                    }],
                },
            }),
        ])
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        positions = accounts.portfolio(
            minimal=True,
            require_success=True,
        )

        self.assertEqual(
            [item.position_id for item in positions],
            ["11", "12"],
        )
        self.assertEqual(len(session.calls), 2)
        expected_common = {
            "view": "COMPLETE",
            "count": 50,
            "sortBy": "SYMBOL",
            "sortOrder": "ASC",
            "marketSession": "REGULAR",
            "totalsRequired": "false",
            "lotsRequired": "false",
        }
        self.assertEqual(
            session.calls[0][1]["params"],
            {**expected_common, "pageNumber": 1},
        )
        self.assertEqual(
            session.calls[1][1]["params"],
            {**expected_common, "pageNumber": 2},
        )

    def test_portfolio_preserves_standard_option_contract_identity(self):
        session = _Session({
            "PortfolioResponse": {
                "AccountPortfolio": [{
                    "accountId": "123",
                    "totalNoOfPages": 1,
                    "Position": [{
                        "positionId": "11",
                        "osiKey": "SPY---260821C00650000",
                        "Product": {
                            "symbol": "SPY",
                            "securityType": "OPTN",
                            "callPut": "CALL",
                            "expiryYear": 2026,
                            "expiryMonth": 8,
                            "expiryDay": 21,
                            "strikePrice": 650,
                        },
                        "quantity": -1,
                        "Complete": {
                            "adjPrice": 1.25,
                            "optionsAdjustedFlag": False,
                            "optionMultiplier": 100,
                            "deliverablesStr": "100 shares",
                        },
                    }],
                }],
            },
        })
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        positions = accounts.portfolio(minimal=True)

        self.assertEqual(len(positions), 1)
        self.assertEqual(
            positions[0].osi_key,
            "SPY---260821C00650000",
        )
        self.assertEqual(positions[0].option_multiplier, 100)
        self.assertIs(positions[0].options_adjusted_flag, False)
        self.assertEqual(
            positions[0].option_deliverables,
            "100 shares",
        )

    def test_manual_spread_selection_requests_standard_unadjusted_contracts_and_retains_osi(self):
        def put(strike, bid, ask, delta):
            return {
                "osiKey": f"SPY---260821P{strike * 1000:08d}",
                "strikePrice": strike,
                "bid": bid,
                "ask": ask,
                "volume": 10,
                "openInterest": 100,
                "OptionGreeks": {
                    "delta": delta,
                    "gamma": 0.01,
                    "iv": 0.20,
                },
            }

        session = _Session({
            "OptionChainResponse": {
                "OptionPair": [
                    {"Put": put(495, 2.00, 2.20, -0.15)},
                    {"Put": put(490, 1.00, 1.20, -0.10)},
                    {"Put": put(485, 0.50, 0.70, -0.05)},
                ],
            },
        })
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.get_stock_price = lambda _ticker: 500

        spread = accounts.get_option_spread_by_price(
            "SPY",
            "Put",
            days_to_expire=25,
            target_premium=0,
            hedge_ratio=1,
            hedge_spread=10,
            qty=1,
            target_delta=0.15,
            target_expiration=date(2026, 8, 21),
        )

        self.assertIsNotNone(spread)
        self.assertEqual(
            session.calls[0][1]["params"]["skipAdjusted"],
            True,
        )
        self.assertEqual(
            session.calls[0][1]["params"]["optionCategory"],
            "STANDARD",
        )
        self.assertEqual(
            spread["sell_option"].osi_key,
            "SPY---260821P00495000",
        )
        self.assertEqual(
            spread["buy_option"].osi_key,
            "SPY---260821P00485000",
        )

    def test_required_portfolio_rejects_duplicate_position_across_pages(self):
        position = {
            "positionId": 11,
            "Product": {
                "symbol": "SPY",
                "securityType": "EQ",
            },
            "quantity": 1,
        }
        session = _Session(responses=[
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalPages": 2,
                        "nextPageNo": 2,
                        "Position": [position],
                    }],
                },
            }),
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalPages": 2,
                        "Position": [position],
                    }],
                },
            }),
        ])
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "duplicate position id",
        ):
            accounts.portfolio(
                minimal=True,
                require_success=True,
            )

        session = _Session(responses=[
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalPages": 2,
                        "nextPageNo": "2",
                        "Position": [position],
                    }],
                },
            }),
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalPages": 2,
                        "Position": [{
                            **position,
                            "positionId": "011",
                        }],
                    }],
                },
            }),
        ])
        accounts.session = session
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid or duplicate position id",
        ):
            accounts.portfolio(
                minimal=True,
                require_success=True,
            )

    def test_required_portfolio_rejects_changed_pagination_field(self):
        position = {
            "positionId": 11,
            "Product": {
                "symbol": "SPY",
                "securityType": "EQ",
            },
            "quantity": 1,
        }
        session = _Session(responses=[
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalNoOfPages": 2,
                        "nextPageNo": 2,
                        "Position": [position],
                    }],
                },
            }),
            _Response({
                "PortfolioResponse": {
                    "AccountPortfolio": [{
                        "accountId": "123",
                        "totalPages": 2,
                        "Position": [{
                            **position,
                            "positionId": 12,
                        }],
                    }],
                },
            }),
        ])
        accounts = Accounts(
            session,
            "https://api.etrade.test",
            consumer_key="",
        )
        accounts.account = {
            "accountId": "123",
            "accountIdKey": "account-key",
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "metadata changed",
        ):
            accounts.portfolio(
                minimal=True,
                require_success=True,
            )

    def test_portfolio_never_returns_a_partial_page_set(self):
        first_page = {
            "PortfolioResponse": {
                "AccountPortfolio": [{
                    "accountId": "123",
                    "totalNoOfPages": 2,
                    "nextPageNo": 2,
                    "Position": [{
                        "positionId": 11,
                        "Product": {
                            "symbol": "SPY",
                            "securityType": "EQ",
                        },
                        "quantity": 1,
                    }],
                }],
            },
        }
        for require_success in (False, True):
            with self.subTest(require_success=require_success):
                session = _Session(responses=[
                    _Response(first_page),
                    _Response(
                        {},
                        status_code=500,
                        text="temporarily unavailable",
                    ),
                ])
                accounts = Accounts(
                    session,
                    "https://api.etrade.test",
                    consumer_key="",
                )
                accounts.account = {
                    "accountId": "123",
                    "accountIdKey": "account-key",
                }

                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed on page 2",
                ):
                    accounts.portfolio(
                        minimal=True,
                        require_success=require_success,
                    )

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
