import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from live_trading import spy_position_tracker


class _Response:
    text = ""

    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _OrderInstance:
    base_url = "https://api.etrade.test"
    account = {"accountIdKey": "account-key"}
    consumer_key = "consumer-key"

    def __init__(self):
        self.session = _Session([])


class SpyPositionTrackerSyncTests(unittest.TestCase):
    def test_fetch_uses_one_date_window_and_completes_pagination(self):
        session = _Session([
            _Response(200, {
                "OrdersResponse": {
                    "Order": [{"orderId": 1}],
                    "marker": "next-page",
                }
            }),
            _Response(200, {
                "OrdersResponse": {
                    "Order": [{"orderId": 2}],
                }
            }),
        ])

        orders, complete, metadata = spy_position_tracker._fetch_executed_orders(
            session,
            "https://api.etrade.test/orders.json",
            {"consumerKey": "key"},
            date(2026, 7, 3),
            date(2026, 7, 17),
            sleep_fn=lambda _: None,
        )

        self.assertTrue(complete)
        self.assertEqual([order["orderId"] for order in orders], [1, 2])
        self.assertEqual(metadata["pages_fetched"], 2)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0][1]["params"]["fromDate"], "07032026")
        self.assertEqual(session.calls[0][1]["params"]["toDate"], "07172026")
        self.assertNotIn("marker", session.calls[0][1]["params"])
        self.assertEqual(session.calls[1][1]["params"]["marker"], "next-page")

    def test_rate_limit_is_retried_before_success(self):
        session = _Session([
            _Response(429, {"Error": {"message": "Rate limit exceeded"}}),
            _Response(200, {"OrdersResponse": {"Order": []}}),
        ])
        delays = []

        orders, complete, metadata = spy_position_tracker._fetch_executed_orders(
            session,
            "https://api.etrade.test/orders.json",
            {"consumerKey": "key"},
            date(2026, 7, 3),
            date(2026, 7, 17),
            sleep_fn=delays.append,
        )

        self.assertTrue(complete)
        self.assertEqual(orders, [])
        self.assertEqual(metadata["pages_fetched"], 1)
        self.assertEqual(delays, [0.5])

    def test_failed_later_page_discards_partial_fetch(self):
        session = _Session([
            _Response(200, {
                "OrdersResponse": {
                    "Order": [{"orderId": 1}],
                    "marker": "next-page",
                }
            }),
            _Response(500, {"Error": {"message": "Temporary outage"}}),
            _Response(500, {"Error": {"message": "Temporary outage"}}),
            _Response(500, {"Error": {"message": "Temporary outage"}}),
        ])

        orders, complete, metadata = spy_position_tracker._fetch_executed_orders(
            session,
            "https://api.etrade.test/orders.json",
            {"consumerKey": "key"},
            date(2026, 7, 3),
            date(2026, 7, 17),
            sleep_fn=lambda _: None,
        )

        self.assertFalse(complete)
        self.assertEqual(orders, [])
        self.assertEqual(metadata["pages_fetched"], 1)
        self.assertEqual(len(session.calls), 4)

    def test_failed_sync_preserves_confirmed_cache_window(self):
        today = date.today()
        flow_date = (today - timedelta(days=2)).isoformat()
        trade = {
            "date": flow_date,
            "action": "SELL_OPEN",
            "symbol": "SPX_2026-07-17_PUT_6000",
            "quantity": 1,
            "price": 20,
            "cash_impact": 2000,
            "order_id": 123,
            "sec_type": "OPTN",
        }
        original_cache = {
            "daily_cash_flows": {flow_date: 2000.0},
            "trade_details": {flow_date: [trade]},
            "days_with_close_events": [flow_date],
            "last_update_date": (today - timedelta(days=1)).isoformat(),
            "sync_health": {
                "status": "ok",
                "last_successful_at": "2026-07-16T16:30:00-04:00",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "spy_gains_cache.json"
            cache_path.write_text(json.dumps(original_cache))
            with (
                patch.object(spy_position_tracker, "SPY_GAINS_CACHE_FILE", str(cache_path)),
                patch.object(
                    spy_position_tracker,
                    "_fetch_executed_orders",
                    return_value=([], False, {
                        "pages_fetched": 0,
                        "error": "Rate limit exceeded",
                    }),
                ),
            ):
                result = spy_position_tracker.calculate_spy_daily_gains(
                    _OrderInstance(),
                    start_date=(today - timedelta(days=30)).isoformat(),
                )
                saved_cache = json.loads(cache_path.read_text())

        self.assertEqual(saved_cache["daily_cash_flows"], original_cache["daily_cash_flows"])
        self.assertEqual(saved_cache["trade_details"], original_cache["trade_details"])
        self.assertEqual(saved_cache["days_with_close_events"], original_cache["days_with_close_events"])
        self.assertEqual(saved_cache["last_update_date"], original_cache["last_update_date"])
        self.assertEqual(saved_cache["sync_health"]["status"], "error")
        self.assertEqual(
            saved_cache["sync_health"]["last_successful_at"],
            original_cache["sync_health"]["last_successful_at"],
        )
        self.assertEqual(result["cash_flows"][flow_date], 2000.0)


if __name__ == "__main__":
    unittest.main()
