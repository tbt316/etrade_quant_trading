from __future__ import annotations
import json
import logging
import configparser
import random
import re
from datetime import datetime,date,timedelta
import numpy as np
from accounts.accounts_bo import StockPosition, is_etrade_token_expired_response
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random
import logging
import requests
from xml.etree import ElementTree as ET
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from live_trading.runtime_safety import (
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
    configure_owner_only_logger,
    payload_fingerprint,
    redact_http_headers,
    reject_legacy_execution,
    resolve_etrade_consumer_key,
)

# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# logger settings
logger = configure_owner_only_logger('my_logger')


class Order:

    @staticmethod
    def _option_tick_size(price: float) -> float:
        try:
            return 0.05 if abs(float(price)) < 3 else 0.10
        except Exception:
            return 0.10

    @classmethod
    def _snap_option_limit_price(cls, target_price: float, current_price: float | None = None) -> float:
        target = abs(float(target_price))
        current = abs(float(current_price)) if current_price is not None else target
        tick = cls._option_tick_size(max(target, current))
        tick_dec = Decimal(str(tick))
        target_dec = Decimal(str(target))

        if abs(target - current) < 1e-9:
            snapped = (Decimal(str(current)) / tick_dec).to_integral_value(rounding=ROUND_HALF_UP) * tick_dec
            return float(max(tick_dec, snapped))

        rounding = ROUND_FLOOR if target < current else ROUND_CEILING
        snapped = (target_dec / tick_dec).to_integral_value(rounding=rounding) * tick_dec

        if target < current and float(snapped) >= current - 1e-9:
            snapped = Decimal(str(max(tick, current - tick)))
        elif target > current and float(snapped) <= current + 1e-9:
            snapped = Decimal(str(current + tick))

        return float(max(tick_dec, snapped))

    def __init__(
        self,
        session,
        account,
        base_url,
        use_sandbox,
        consumer_key=None,
        runtime_safety: RuntimeSafetyBoundary | None = None,
    ):
        self.session = session
        self.account = account
        self.base_url = base_url
        self.use_sandbox = use_sandbox
        self.runtime_safety = runtime_safety
        if runtime_safety is not None:
            if runtime_safety.use_sandbox != use_sandbox:
                raise RuntimeSafetyError("order client environment conflicts with runtime safety boundary")
            runtime_safety.verify_account(account)
        config_key = "SANDBOX_CONSUMER_KEY" if self.use_sandbox else "PROD_CONSUMER_KEY"
        self.consumer_key = resolve_etrade_consumer_key(
            self.use_sandbox,
            consumer_key=consumer_key,
            config_value=config["DEFAULT"].get(config_key),
        )

    def _assert_order_api_access(self) -> None:
        """Revalidate the arm and exact account at every order API boundary."""

        if self.runtime_safety is None:
            raise RuntimeSafetyError(
                "order API access requires an authenticated RuntimeSafetyBoundary"
            )
        if self.runtime_safety.use_sandbox != self.use_sandbox:
            raise RuntimeSafetyError("order client environment conflicts with runtime safety boundary")
        self.runtime_safety.verify_account(self.account)

    def _refresh_auth_session_if_possible(self, reason):
        callback = getattr(self, "auth_refresh_callback", None)
        if not callable(callback):
            return False
        refreshed = callback(reason)
        if not refreshed:
            return False
        self.session, self.base_url = refreshed
        return True

    def preview_order(self, order):
        """
        Preview an order by calling the E*TRADE preview order API.

        Parameters:
        - order (dict): Dictionary containing the order details.

        Returns:
        - dict: Response from the preview order API, or None if the preview fails.
        """
        reject_legacy_execution("order.order_bo.Order.preview_order")


    def preview_order_old(self, order):
        """
        Call preview order API based on selecting from different given options

        :param self: Pass in authenticated session and information on selected account
        """
        reject_legacy_execution("order.order_bo.Order.preview_order_old")

    def place_order(self, order, preview_only=False):
        """
        Place an order based on the preview order response.

        Parameters:
        - order (dict): Dictionary containing the order details.

        Returns:
        - str: The order ID if the order is placed successfully, or None if it fails.
        """
        reject_legacy_execution("order.order_bo.Order.place_order")

    def place_order_old(self, order):
        """
        Place order based on the preview order response
        """
        reject_legacy_execution("order.order_bo.Order.place_order_old")

    def get_order_details_by_id(self, order_id: int) -> dict | None:
        """
        Fetch a single order's details by ID using the dedicated details endpoint.
        Falls back to scanning OPEN/EXECUTED lists if the details endpoint is unavailable.
        """
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/{order_id}.json"
        headers = {"consumerKey": self.consumer_key}
        try:
            r = self.session.get(url, header_auth=True, headers=headers)
            if r.status_code == 200:
                data = r.json()
                return data.get("OrdersResponse", {}).get("Order", [{}])[0]
        except Exception:
            pass

        # Fallback: search open orders first, then executed
        for status in ("OPEN", "EXECUTED"):
            url_list = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
            params = {"status": status}
            rl = self.session.get(url_list, header_auth=True, params=params, headers=headers)
            if rl.status_code == 200:
                data = rl.json().get("OrdersResponse", {}).get("Order", [])
                for od in data:
                    if od.get("orderId") == order_id:
                        return od
        return None

    def _build_instruments_xml_from_detail(self, detail: dict) -> str:
        instruments = detail.get("Instrument", [])
        instruments_xml = ""
        for instrument in instruments:
            product = instrument.get('Product', {})
            leg_xml = f"""
            <Instrument>
                <Product>
                    <securityType>{product.get('securityType','')}</securityType>
                    <symbol>{product.get('symbol','')}</symbol>
                    <callPut>{product.get('callPut','')}</callPut>
                    <expiryYear>{product.get('expiryYear','')}</expiryYear>
                    <expiryMonth>{product.get('expiryMonth','')}</expiryMonth>
                    <expiryDay>{product.get('expiryDay','')}</expiryDay>
                    <strikePrice>{product.get('strikePrice','')}</strikePrice>
                </Product>
                <orderAction>{instrument.get('orderAction','')}</orderAction>
                <quantityType>QUANTITY</quantityType>
                <quantity>{instrument.get('orderedQuantity', instrument.get('quantity', 0))}</quantity>
            </Instrument>"""
            instruments_xml += leg_xml
        return instruments_xml

    def change_order_limit(self, order_id: int, new_limit_price: float, order_detail: dict, order_type: str) -> tuple[bool, int | None]:
        """
        Preview and place a change to an existing order's limit price.
        Returns True if the place-change call succeeds (HTTP 200), False otherwise.
        """
        reject_legacy_execution("order.order_bo.Order.change_order_limit")

    def _find_replacement_order_id(self, replaced_order_id: int) -> int | None:
        """Search OPEN then EXECUTED orders for an order whose details.replacesOrderId equals replaced_order_id."""
        headers = {"consumerKey": self.consumer_key}
        list_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        for status in ("OPEN", "EXECUTED"):
            try:
                r = self.session.get(list_url, header_auth=True, params={"status": status}, headers=headers)
                if r.status_code != 200:
                    continue
                orders = r.json().get("OrdersResponse", {}).get("Order", [])
                for od in orders:
                    for d in od.get("OrderDetail", []):
                        if d.get("replacesOrderId") == replaced_order_id:
                            return od.get("orderId")
            except Exception:
                continue
        return None

    def wait_and_adjust_until_filled(self, order_id: int, step: float = 0.05, interval_sec: int = 10, max_checks: int = 120, initial_wait: bool = True) -> tuple[bool, int]:
        """
        Poll order status every `interval_sec`; if still OPEN, adjust limit by `step`.
        - For NET_CREDIT, decrease the limit.
        - For NET_DEBIT or BUY LIMIT, increase the limit.
        Stops when EXECUTED, EXPIRED, REJECTED, or after `max_checks` adjustments.
        """
        reject_legacy_execution("order.order_bo.Order.wait_and_adjust_until_filled")

    def previous_order(self, session, account, prev_orders):
        """
        Calls preview order API based on a list of previous orders

        :param session: authenticated session
        :param account: information on selected account
        :param prev_orders: list of instruments from previous orders
        """
        reject_legacy_execution("order.order_bo.Order.previous_order")

    @staticmethod
    def print_orders(response, status):
        """
        Formats and displays a list of order

        :param response: response object of a list of orders
        :param status: order status related to the response object
        :return a list of previous orders
        """
        prev_orders = []
        if response is not None and "OrdersResponse" in response and "Order" in response["OrdersResponse"]:
            for order in response["OrdersResponse"]["Order"]:
                if order is not None and "OrderDetail" in order:
                    for details in order["OrderDetail"]:
                        if details is not None and "Instrument" in details:
                            for instrument in details["Instrument"]:
                                order_str = ""
                                order_obj = {"priceType": None,
                                             "orderTerm": None,
                                             "order_indicator": None,
                                             "order_type": None,
                                             "security_type": None,
                                             "symbol": None,
                                             "orderAction": None,
                                             "quantity": None}
                                if order is not None and 'orderType' in order:
                                    order_obj["order_type"] = order["orderType"]

                                if order is not None and 'orderId' in order:
                                    order_str += "Order #" + str(order["orderId"]) + " : "

                                if instrument is not None and 'Product' in instrument \
                                        and 'securityType' in instrument["Product"]:
                                    order_str += "Type: " + instrument["Product"]["securityType"] + " | "
                                    order_obj["security_type"] = instrument["Product"]["securityType"]

                                if instrument is not None and 'orderAction' in instrument:
                                    order_str += "Order Type: " + instrument["orderAction"] + " | "
                                    order_obj["orderAction"] = instrument["orderAction"]

                                if instrument is not None and 'orderedQuantity' in instrument:
                                    order_str += "Quantity(Exec/Entered): " + str("{:,}".format(instrument["orderedQuantity"])) + " | "
                                    order_obj["quantity"] = instrument["orderedQuantity"]

                                if instrument is not None and 'Product' in instrument and 'symbol' in instrument["Product"]:
                                    order_str += "Symbol: " + instrument["Product"]["symbol"] + " | "
                                    order_obj["symbol"] = instrument["Product"]["symbol"]

                                if details is not None and 'priceType' in details:
                                    order_str += "Price Type: " + details["priceType"] + " | "
                                    order_obj["priceType"] = details["priceType"]

                                if details is not None and 'orderTerm' in details:
                                    order_str += "Term: " + details["orderTerm"] + " | "
                                    order_obj["orderTerm"] = details["orderTerm"]

                                if details is not None and 'limitPrice' in details:
                                    order_str += "Price: " + str('${:,.2f}'.format(details["limitPrice"])) + " | "
                                    order_obj["limitPrice"] = details["limitPrice"]

                                if status == "Open" and details is not None and 'netBid' in details:
                                    order_str += "Bid: " + details["netBid"] + " | "
                                    order_obj["bid"] = details["netBid"]

                                if status == "Open" and details is not None and 'netAsk' in details:
                                    order_str += "Ask: " + details["netAsk"] + " | "
                                    order_obj["ask"] = details["netAsk"]

                                if status == "Open" and details is not None and 'netPrice' in details:
                                    order_str += "Last Price: " + details["netPrice"] + " | "
                                    order_obj["netPrice"] = details["netPrice"]

                                if status == "indiv_fills" and instrument is not None and 'filledQuantity' in instrument:
                                    order_str += "Quantity Executed: " + str("{:,}".format(instrument["filledQuantity"])) + " | "
                                    order_obj["quantity"] = instrument["filledQuantity"]

                                if status != "open" and status != "expired" and status != "rejected" and instrument is not None \
                                        and "averageExecutionPrice" in instrument:
                                    order_str += "Price Executed: " + str('${:,.2f}'.format(instrument["averageExecutionPrice"])) + " | "

                                if status != "expired" and status != "rejected" and details is not None and 'status' in details:
                                    order_str += "Status: " + details["status"]

                                print(order_str)
                                prev_orders.append(order_obj)
        return prev_orders

    @staticmethod
    def print_orders_customized(response, status):
        """
        Formats and displays a list of orders

        :param response: response object of a list of orders
        :param status: order status related to the response object
        :return a list of previous orders
        """
        prev_orders = []
        if response is not None and "OrdersResponse" in response and "Order" in response["OrdersResponse"]:
            for order in response["OrdersResponse"]["Order"]:
                if order is not None and "OrderDetail" in order:
                    for details in order["OrderDetail"]:
                        order_str = ""
                        order_obj = {"priceType": None,
                                        "orderTerm": None,
                                        "excuted_time": None,
                                        "order_indicator": None,
                                        "order_type": None,
                                        "security_type": None,
                                        "symbol": None,
                                        "orderAction": None,
                                        "quantity": None}

                        if order is not None and 'orderType' in order:
                            order_obj["order_type"] = order["orderType"]

                        if order is not None and 'orderId' in order:
                            order_str += "Order #" + str(order["orderId"]) + " : "
                        
                        if order is not None and 'executedTime' in details:
                            order_str += "Date:" + str(datetime.datetime.fromtimestamp(int(details["executedTime"]/1000))) + " | "
                            order_obj["excuted_time"] = details["executedTime"]
                            
                        if details is not None and "Instrument" in details:
                            for instrument in details["Instrument"]:
                                if instrument is not None and 'Product' in instrument \
                                        and 'securityType' in instrument["Product"]:
                                    order_str += "Type: " + instrument["Product"]["securityType"] + " | "
                                    order_obj["security_type"] = instrument["Product"]["securityType"]

                                if instrument is not None and 'orderAction' in instrument:
                                    order_str += "Type: " + instrument["orderAction"] + " | "
                                    order_obj["orderAction"] = instrument["orderAction"]

                                if instrument is not None and 'orderedQuantity' in instrument:
                                    order_str += "Quantity: " + str("{:,}".format(instrument["orderedQuantity"])) + " | "
                                    order_obj["quantity"] = instrument["orderedQuantity"]

                                if instrument is not None and 'Product' in instrument and 'symbol' in instrument["Product"]:
                                    order_str += "Symbol: " + instrument["Product"]["symbol"] + " | "
                                    order_obj["symbol"] = instrument["Product"]["symbol"]

                                if details is not None and 'priceType' in details:
                                    order_str += "Price Type: " + details["priceType"] + " | "
                                    order_obj["priceType"] = details["priceType"]

                                if details is not None and 'orderTerm' in details:
                                    # order_str += "Term: " + details["orderTerm"] + " | "
                                    order_obj["orderTerm"] = details["orderTerm"]

                                if details is not None and 'limitPrice' in details:
                                    # order_str += "Price: " + str('${:,.2f}'.format(details["limitPrice"])) + " | "
                                    order_obj["limitPrice"] = details["limitPrice"]

                                if status == "Open" and details is not None and 'netBid' in details:
                                    order_str += "Bid: " + details["netBid"] + " | "
                                    order_obj["bid"] = details["netBid"]

                                if status == "Open" and details is not None and 'netAsk' in details:
                                    order_str += "Ask: " + details["netAsk"] + " | "
                                    order_obj["ask"] = details["netAsk"]

                                if status == "Open" and details is not None and 'netPrice' in details:
                                    order_str += "Last Price: " + details["netPrice"] + " | "
                                    order_obj["netPrice"] = details["netPrice"]

                                if status == "indiv_fills" and instrument is not None and 'filledQuantity' in instrument:
                                    order_str += "Quantity Executed: " + str("{:,}".format(instrument["filledQuantity"])) + " | "
                                    order_obj["quantity"] = instrument["filledQuantity"]

                                if status != "open" and status != "expired" and status != "rejected" and instrument is not None \
                                        and "averageExecutionPrice" in instrument:
                                    order_str += "Price Executed: " + str('${:,.2f}'.format(instrument["averageExecutionPrice"])) + " | "

                                if status != "expired" and status != "rejected" and details is not None and 'status' in details:
                                    order_str += "Status: " + details["status"]

                                print(order_str)
                                prev_orders.append(order_obj)
        return prev_orders

    def options_selection(self, options):
        """
        Formats and displays different options in a menu

        :param options: List of options to display
        :return the number user selected
        """
        reject_legacy_execution("order.order_bo.Order.options_selection")

    def user_select_order(self):
        """
            Provides users options to select to preview orders
            :param self test
            :return user's order selections
            """
        reject_legacy_execution("order.order_bo.Order.user_select_order")

    def preview_order_menu(self, session, account, prev_orders):
        """
        Provides the different options for preview orders: select new order or select from previous order

        :param session: authenticated session
        :param account: information on selected account
        :param prev_orders: list of instruments from previous orders
        """
        reject_legacy_execution("order.order_bo.Order.preview_order_menu")

    def cancel_order(self):
        """
        Calls cancel order API to cancel an existing order
        :param self: Pass parameter with authenticated session and information on selected account
        """
        reject_legacy_execution("order.order_bo.Order.cancel_order")

    def view_orders(self):
        """
        Calls orders API to provide the details for the orders

        :param self: Pass in authenticated session and information on selected account
        """
        reject_legacy_execution("order.order_bo.Order.view_orders")

    def filter_order(self):
        """
        Calls orders API to provide the details for the orders

        :param self: Pass in authenticated session and information on selected account
        """
        reject_legacy_execution("order.order_bo.Order.filter_order")

    def get_open_orders(self) -> list:
        """
        Get all open orders for the current account.
        :return: List of open orders, each represented as a dictionary
        """
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        headers = {"consumerKey": self.consumer_key}
        params = {"status": "OPEN"}

        response = self.session.get(url, header_auth=True, params=params, headers=headers)
        if is_etrade_token_expired_response(response) and self._refresh_auth_session_if_possible("open orders fetch"):
            response = self.session.get(url, header_auth=True, params=params, headers=headers)
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))
        logger.debug("Response Body: %s", response.text)

        open_orders_list = []

        if response.status_code == 200:
            data = response.json()
            open_orders = data.get("OrdersResponse", {}).get("Order", [])

            for order in open_orders:
                for detail in order.get("OrderDetail", []):
                    # Handle multiple instruments (spread orders)
                    for instrument in detail.get('Instrument', []):
                        expiry_year = instrument['Product'].get('expiryYear')
                        expiry_month = instrument['Product'].get('expiryMonth')
                        expiry_day = instrument['Product'].get('expiryDay')
                        if expiry_year and expiry_month and expiry_day:
                            expiry_date = f"{expiry_year}-{expiry_month:02d}-{expiry_day:02d}"
                        else:
                            expiry_date = None
                        
                        order_dict = {
                            "orderId": order['orderId'],
                            "placedTime": order.get('placedTime'),
                            "orderTerm": detail.get('orderTerm'),
                            "priceType": detail.get('priceType'),
                            "limitPrice": detail.get('limitPrice'),
                            "orderType": order.get('orderType'),
                            "symbol": instrument['Product']['symbol'],
                            "securityType": instrument['Product']['securityType'],
                            "orderAction": instrument.get('orderAction'),
                            "quantity": instrument.get('orderedQuantity', instrument.get('quantity')),
                            "strikePrice": instrument['Product'].get('strikePrice'),
                            "expiryDate": expiry_date,
                            "callPut": instrument['Product'].get('callPut'),
                            "clientOrderId": order.get('clientOrderId'),
                            "allDetail": detail # Keep full detail for change_order_limit
                        }
                        open_orders_list.append(order_dict)
        elif response.status_code == 204:
            logger.debug("No open orders found (204 No Content)")
        else:
            print(f"Error fetching open orders: {response.status_code} - {response.text}")

        return open_orders_list

    def get_cancelled_orders(self, start_date: str) -> list:
        """
        Get cancelled orders starting from a certain date.
        :param start_date: The date from which to start fetching cancelled orders (inclusive)
        :return: List of cancelled order legs, each represented as a dictionary
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        headers = {"consumerKey": self.consumer_key}
        params = {"status": "CANCELLED"}

        response = self.session.get(url, header_auth=True, params=params, headers=headers)
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))
        logger.debug("Response Body: %s", response.text)

        cancelled_orders_list = []
        if response.status_code == 200:
            data = response.json()
            cancelled_orders = data.get("OrdersResponse", {}).get("Order", [])

            for order in cancelled_orders:
                for detail in order.get("OrderDetail", []):
                    order_time = detail.get("cancelledTime") or detail.get("placedTime")
                    if not order_time:
                        continue
                    order_date = datetime.fromtimestamp(order_time / 1000).date()
                    if order_date < start_date:
                        continue
                    for instrument in detail.get('Instrument', []):
                        product = instrument.get('Product', {})
                        expiry_year = product.get('expiryYear')
                        expiry_month = product.get('expiryMonth')
                        expiry_day = product.get('expiryDay')
                        if expiry_year is not None and expiry_month is not None and expiry_day is not None:
                            expiry_date = f"{expiry_year}-{expiry_month:02d}-{expiry_day:02d}"
                        else:
                            expiry_date = None
                        cancelled_orders_list.append({
                            "order_id": order['orderId'],
                            "cancelled_price": instrument.get('averageExecutionPrice') or detail.get('limitPrice'),
                            "cancelled_quantity": instrument.get('cancelledQuantity') or instrument.get('orderedQuantity') or instrument.get('quantity'),
                            "cancelled_date": datetime.fromtimestamp(order_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            "price_type": detail.get('priceType'),
                            "symbol": product.get('symbol'),
                            "equity_type": product.get('securityType'),
                            "client_order_id": order.get('clientOrderId'),
                            "order_action": instrument.get('orderAction'),
                            "strike_price": product.get('strikePrice'),
                            "expiry_date": expiry_date,
                            "option_type": product.get('callPut'),
                            "symbol_description": instrument.get('symbolDescription')
                        })
        elif response.status_code == 204:
            logger.debug("No cancelled orders found (204 No Content)")
        else:
            logger.error("Failed to fetch cancelled orders. Status Code: %s, Response: %s", response.status_code, response.text)

        return cancelled_orders_list

    def get_executed_orders(self, start_date: str) -> list:
        """
        Get all executed orders starting from a certain date
        :param start_date: The date from which to start fetching executed orders (inclusive)
        :return: List of executed orders, each represented as a dictionary
        """
        # Convert start_date to datetime
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # URL for the API endpoint
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"

        # Add parameters and header information
        headers = {"consumerKey": self.consumer_key}
        params = {"status": "EXECUTED"}

        # Make API call for GET request
        response = self.session.get(url, header_auth=True, params=params, headers=headers)
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))
        logger.debug("Response Body: %s", response.text)

        executed_orders_list = []

        if response.status_code == 200:
            data = response.json()
            executed_orders = data.get("OrdersResponse", {}).get("Order", [])

            for order in executed_orders:
                for detail in order.get("OrderDetail", []):
                    executed_time = detail.get("executedTime")
                    if executed_time:
                        # Convert the executedTime to a date
                        executed_date = datetime.fromtimestamp(executed_time / 1000).date()
                        if executed_date >= start_date:
                            # Handle multiple instruments (spread orders)
                            for instrument in detail.get('Instrument', []):
                                expiry_year = instrument['Product'].get('expiryYear')
                                expiry_month = instrument['Product'].get('expiryMonth')
                                expiry_day = instrument['Product'].get('expiryDay')
                                if expiry_year is not None and expiry_month is not None and expiry_day is not None:
                                    expiry_date = f"{expiry_year}-{expiry_month:02d}-{expiry_day:02d}"
                                else:
                                    expiry_date = None
                                order_dict = {
                                    "order_id": order['orderId'],
                                    "executed_price": instrument.get('averageExecutionPrice'),
                                    "executed_quantity": instrument.get('filledQuantity'),
                                    "executed_date": datetime.fromtimestamp(executed_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                                    "price_type": detail['priceType'],
                                    "symbol": instrument['Product']['symbol'],
                                    "equity_type": instrument['Product']['securityType'],
                                    "client_order_id": order.get('clientOrderId'),
                                    # Add additional fields for options
                                    "order_action": instrument.get('orderAction'),
                                    "strike_price": instrument['Product'].get('strikePrice'),
                                    "expiry_date": expiry_date,
                                    "option_type": instrument['Product'].get('callPut'),
                                    "symbol_description": instrument.get('symbolDescription')
                                }
                                executed_orders_list.append(order_dict)

        else:
            logger.error("Failed to fetch orders. Status Code: %s, Response: %s", response.status_code, response.text)

        return executed_orders_list

    def get_opened_orders(self, start_date: str) -> list:
        """
        Get all opened orders starting from a certain date
        :param start_date: The date from which to start fetching opened orders (inclusive)
        :return: List of opened orders, each represented as a dictionary
        """
        # Convert start_date to datetime
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # URL for the API endpoint
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"

        # Add parameters and header information
        headers = {"consumerKey": self.consumer_key}
        params = {"status": "OPEN"}

        # Make API call for GET request
        response = self.session.get(url, header_auth=True, params=params, headers=headers)
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))
        logger.debug("Response Body: %s", response.text)

        opened_orders_list = []

        if response.status_code == 200:
            data = response.json()
            opened_orders = data.get("OrdersResponse", {}).get("Order", [])

            for order in opened_orders:
                for detail in order.get("OrderDetail", []):
                    placed_time = detail.get("placedTime")
                    if placed_time:
                        # Convert the placedTime to a date
                        placed_date = datetime.datetime.fromtimestamp(placed_time / 1000).date()
                        if placed_date >= start_date:
                            order_dict = {
                                "order_id": order['orderId'],
                                "placed_price": detail.get('limitPrice', detail.get('stopPrice', '')),
                                "placed_quantity": detail['Instrument'][0].get('orderedQuantity'),
                                "placed_date": datetime.datetime.fromtimestamp(placed_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                                "price_type": detail['priceType'],
                                "symbol": detail['Instrument'][0]['Product']['symbol'],
                                "equity_type": detail['Instrument'][0]['Product']['securityType'],
                                "client_order_id": order.get('clientOrderId')
                            }
                            print(order_dict)
                            opened_orders_list.append(order_dict)

        else:
            logger.error("Failed to fetch orders. Status Code: %s, Response: %s", response.status_code, response.text)

        return opened_orders_list

    def place_option_order(self, ticker_symbol: str, strike_price: float, expiration_date: str, call_put: str, quantity: int = 1, price_type: str = "MARKET", limit_price: float = None, order_action: str = "BUY_TO_OPEN", order_term: str = "GOOD_FOR_DAY"):
            """
            Place an order for an option using the E*TRADE API.

            Parameters:
            - ticker_symbol (str): The underlying stock symbol.
            - strike_price (float): The strike price of the option.
            - expiration_date (str): The expiration date of the option in "YYYY-MM-DD" format.
            - call_put (str): The type of option ('CALL' or 'PUT').
            - quantity (int): The number of option contracts to trade. Default is 1.
            - price_type (str): The price type of the order, e.g., 'LIMIT' or 'MARKET'. Default is 'LIMIT'.
            - limit_price (float): The limit price for the order (if price_type is 'LIMIT'). Default is None.
            - order_action (str): The action for the order, e.g., 'BUY_TO_OPEN' or 'SELL_TO_CLOSE'. Default is 'BUY_TO_OPEN'.
            - order_term (str): The term for the order, e.g., 'GOOD_FOR_DAY'. Default is 'GOOD_FOR_DAY'.

            Returns:
            - dict: The response from the E*TRADE API if the order is successful, or None if it fails.
            """
            reject_legacy_execution("order.order_bo.Order.place_option_order")

    def option_gain_new(self, start_date: str, generate_plot=False):
        """
        Calculate the cash flow for executed option trades conducted on or after the given start date.
        
        This function aggregates only the trades executed on or after the start_date and calculates the cash flow
        by summing the trade amounts. It does not calculate detailed gains or losses using cost basis tracking.
        The time range covers from the start_date to the current date, broken into 5-day chunks to fetch orders.
        The aggregated cash flows (inflow vs outflow) are then printed, and optionally a plot is generated.
        
        :param start_date: The starting date (YYYY-MM-DD) from which to consider executed trades.
        :param generate_plot: If True, display a bar chart of total cash flow by underlying asset.
        :return: None. Prints total cash flow by underlying asset and overall total cash flow.
        """
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        from_date = start_date_dt.strftime('%m%d%Y')
        to_date = datetime.now().strftime('%m%d%Y')
    
        # Base URL and Headers
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        headers = {"consumerKey": self.consumer_key}
    
        # Break the period from start_date to current date into 5-day chunks
        date_ranges = []
        current_start = start_date_dt
        while current_start <= datetime.now().date():
            current_end = min(current_start + timedelta(days=1), datetime.now().date())
            date_ranges.append((current_start.strftime('%m%d%Y'), current_end.strftime('%m%d%Y')))
            current_start = current_end + timedelta(days=1)
    
        # Aggregate open orders data
        open_orders_data = {"OrdersResponse": {"Order": []}}
    
        for start, end in date_ranges:
            marker = None
            while True:
                params_open_trades = {
                    "status": "EXECUTED",
                    "count": 100,
                    "fromDate": start,
                    "toDate": end,
                    # "securityType": "OPTN",
                }
                if marker:
                    params_open_trades['marker'] = marker

                print(f"Fetching executed trades from {start} to {end} (Marker: {marker})...")
        
                response_open = self.session.get(url, params=params_open_trades, headers=headers)
                if response_open.status_code != 200:
                    logger.error("Failed to fetch open orders. Status Code: %s, Response: %s", response_open.status_code, response_open.text)
                    break
        
                # Parse and append data
                chunk_data = response_open.json()
                if "OrdersResponse" in chunk_data and "Order" in chunk_data["OrdersResponse"]:
                    orders = chunk_data["OrdersResponse"]["Order"]
                    if orders:
                        open_orders_data["OrdersResponse"]["Order"].extend(orders)
                        if "marker" in chunk_data["OrdersResponse"] and chunk_data["OrdersResponse"]["marker"]:
                            marker = chunk_data["OrdersResponse"]["marker"]
                        else:
                            break 
                    else:
                        break
                else:
                    break
    
        # Process open orders data
        open_orders = open_orders_data.get("OrdersResponse", {}).get("Order", [])
        print(f"Total orders fetched: {len(open_orders)}")
    
        # Cost basis tracker using both symbol and quantity as keys
        cash_flow_by_underlying = defaultdict(float)  # Track total cash flow per underlying asset
        
        # ===== FIRST PASS: Collect all trades by type =====
        option_trades = []  # All option trades
        assignment_eq_trades = []  # EQ from OPTION_ASSIGNMENT (assignment BUYs/SELLs)
        regular_eq_trades = []  # Regular EQ orders (potential close-outs)
        
        for order in open_orders:
            order_type = order.get("orderType")
            order_id = order.get("orderId")
            for detail in order.get("OrderDetail", []):
                if detail.get("status") != "EXECUTED":
                    continue
                executed_time = detail.get('executedTime')
                executed_date = None
                if executed_time:
                    executed_date = datetime.fromtimestamp(executed_time / 1000).strftime('%Y-%m-%d')
                
                for instrument in detail.get("Instrument", []):
                    sec_type = instrument['Product']['securityType']
                    filled_quantity = float(instrument["filledQuantity"])
                    order_action = instrument["orderAction"]
                    symbol = instrument['Product']['symbol']
                    average_execution_price = instrument.get("averageExecutionPrice")
                    
                    if average_execution_price is None:
                        continue
                    
                    # Extract underlying
                    underlying_asset = symbol
                    try:
                        if "--" in symbol:
                            underlying_asset = symbol.split("--")[0]
                        else:
                            match = re.match(r"([A-Za-z]+)", symbol)
                            underlying_asset = match.group(1) if match else symbol
                    except:
                        pass
                    
                    trade_info = {
                        "order_id": order_id,
                        "order_type": order_type,
                        "sec_type": sec_type,
                        "symbol": symbol,
                        "underlying": underlying_asset,
                        "action": order_action,
                        "quantity": filled_quantity,
                        "price": average_execution_price,
                        "executed_date": executed_date
                    }
                    
                    if sec_type == "OPTN":
                        option_trades.append(trade_info)
                    elif sec_type == "EQ":
                        if order_type == "OPTION_ASSIGNMENT":
                            assignment_eq_trades.append(trade_info)
                        else:
                            regular_eq_trades.append(trade_info)
        
        # ===== SECOND PASS: Match assignment EQ with close-out EQ =====
        used_eq_orders = set()  # Track matched close-out orders
        
        for assign_trade in assignment_eq_trades:
            assign_symbol = assign_trade["underlying"]
            assign_qty = assign_trade["quantity"]
            assign_action = assign_trade["action"]
            assign_price = assign_trade["price"]
            
            # Add assignment trade to cash flow
            trade_amount = assign_qty * assign_price
            if assign_action in ["SELL", "SELL_TO_COVER"]:
                cash_flow_by_underlying[assign_symbol] += trade_amount
            elif assign_action in ["BUY", "BUY_TO_COVER"]:
                cash_flow_by_underlying[assign_symbol] -= trade_amount
            
            # Find matching close-out trade
            # If assignment is BUY, look for SELL with same qty
            # If assignment is SELL, look for BUY with same qty
            for eq_trade in regular_eq_trades:
                if eq_trade["order_id"] in used_eq_orders:
                    continue
                if eq_trade["underlying"] != assign_symbol:
                    continue
                if eq_trade["quantity"] != assign_qty:
                    continue
                
                # Check for opposite action
                eq_action = eq_trade["action"]
                is_match = False
                if assign_action in ["BUY", "BUY_TO_COVER"] and eq_action in ["SELL", "SELL_TO_COVER"]:
                    is_match = True
                elif assign_action in ["SELL", "SELL_TO_COVER"] and eq_action in ["BUY", "BUY_TO_COVER"]:
                    is_match = True
                
                if is_match:
                    # Add close-out trade to cash flow
                    closeout_amount = eq_trade["quantity"] * eq_trade["price"]
                    if eq_action in ["SELL", "SELL_TO_COVER"]:
                        cash_flow_by_underlying[assign_symbol] += closeout_amount
                    elif eq_action in ["BUY", "BUY_TO_COVER"]:
                        cash_flow_by_underlying[assign_symbol] -= closeout_amount
                    
                    used_eq_orders.add(eq_trade["order_id"])
                    print(f"[Assignment Match] {assign_symbol}: Assignment {assign_action} {assign_qty:.0f} @ ${assign_price:.2f} matched with {eq_action} @ ${eq_trade['price']:.2f}")
                    break
        
        # ===== THIRD PASS: Process all option trades =====
        for trade in option_trades:
            underlying = trade["underlying"]
            qty = trade["quantity"]
            price = trade["price"]
            action = trade["action"]
            
            trade_amount = qty * price * 100  # Options use 100x multiplier
            
            if action in ["SELL_OPEN", "SELL_CLOSE", "SELL", "SELL_TO_COVER"]:
                cash_flow_by_underlying[underlying] += trade_amount
            elif action in ["BUY_OPEN", "BUY_CLOSE", "BUY", "BUY_TO_COVER"]:
                cash_flow_by_underlying[underlying] -= trade_amount

        # ===== FETCH CURRENT PORTFOLIO WITH LOTS for position annotations =====
        # {symbol: {"quantity": qty, "marketValue": value, "lots": [{date, qty, price, marketValue}]}}
        current_equity_positions = {}
        try:
            portfolio_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/portfolio.json"
            portfolio_params = {"view": "COMPLETE", "count": 100, "lotsRequired": "true"}
            portfolio_response = self.session.get(portfolio_url, params=portfolio_params, header_auth=True)
            if portfolio_response.status_code == 200:
                portfolio_data = portfolio_response.json()
                if "PortfolioResponse" in portfolio_data and "AccountPortfolio" in portfolio_data["PortfolioResponse"]:
                    for acctPortfolio in portfolio_data["PortfolioResponse"]["AccountPortfolio"]:
                        for position in acctPortfolio.get("Position", []):
                            product = position.get("Product", {})
                            if product.get("securityType") == "EQ":
                                symbol = product.get("symbol", "")
                                qty = float(position.get("quantity", 0))
                                market_value = float(position.get("marketValue", 0))
                                current_price = float(position.get("Quick", {}).get("lastTrade", 0))
                                
                                # Extract lot information
                                lots = []
                                for lot in position.get("PositionLot", []):
                                    lot_date_ms = lot.get("acquiredDate")
                                    lot_date = None
                                    if lot_date_ms:
                                        lot_date = datetime.fromtimestamp(lot_date_ms / 1000).strftime('%Y-%m-%d')
                                    lot_qty = float(lot.get("remainingQty", 0))
                                    lot_price = float(lot.get("price", 0))
                                    lot_market_value = lot_qty * current_price
                                    lots.append({
                                        "date": lot_date,
                                        "quantity": lot_qty,
                                        "price": lot_price,
                                        "marketValue": lot_market_value
                                    })
                                
                                current_equity_positions[symbol] = {
                                    "quantity": qty,
                                    "marketValue": market_value,
                                    "currentPrice": current_price,
                                    "lots": lots
                                }
        except Exception as e:
            print(f"[option_gain_new] Could not fetch portfolio: {e}")
        
        # Track unmatched assignment buys (no close-out found)
        unmatched_assignments = {}  # {underlying: [assignment trades without matching sells]}
        for assign_trade in assignment_eq_trades:
            assign_symbol = assign_trade["underlying"]
            assign_qty = assign_trade["quantity"]
            assign_action = assign_trade["action"]
            
            # Check if this assignment had a match
            matched = False
            for eq_trade in regular_eq_trades:
                if eq_trade["order_id"] in used_eq_orders:
                    if eq_trade["underlying"] == assign_symbol and eq_trade["quantity"] == assign_qty:
                        matched = True
                        break
            
            if not matched and assign_action in ["BUY", "BUY_TO_COVER"]:
                if assign_symbol not in unmatched_assignments:
                    unmatched_assignments[assign_symbol] = []
                unmatched_assignments[assign_symbol].append(assign_trade)

        # Print total cash flow by underlying asset
        print("\nTotal Cash Flow by Underlying Asset:")
        for underlying, cash_flow in cash_flow_by_underlying.items():
            annotation = ""
            # Check if there are unmatched assignments for this underlying
            if underlying in unmatched_assignments:
                # Try to match assignment dates with portfolio lot dates
                matched_lots = []
                if underlying in current_equity_positions:
                    pos = current_equity_positions[underlying]
                    for assign_trade in unmatched_assignments[underlying]:
                        assign_date = assign_trade.get("executed_date")
                        assign_qty = assign_trade["quantity"]
                        assign_price = assign_trade["price"]
                        
                        # Find lot with matching date
                        for lot in pos.get("lots", []):
                            if lot["date"] == assign_date and lot["quantity"] == assign_qty:
                                matched_lots.append({
                                    "date": assign_date,
                                    "qty": assign_qty,
                                    "costBasis": assign_price,
                                    "currentMktVal": lot["marketValue"],
                                    "currentPrice": pos.get("currentPrice", 0)
                                })
                                break
                
                if matched_lots:
                    # Format explicit annotation for matched lots
                    lot_details = []
                    for ml in matched_lots:
                        lot_details.append(f"{ml['date']}: {ml['qty']:.0f}sh @ ${ml['costBasis']:.2f}, now ${ml['currentMktVal']:,.2f}")
                    annotation = f" [Open Assignment: {'; '.join(lot_details)}]"
                elif underlying in current_equity_positions:
                    # Fallback: show all lots for this symbol
                    pos = current_equity_positions[underlying]
                    unmatched_qty = sum(t["quantity"] for t in unmatched_assignments[underlying])
                    annotation = f" [Assignment {unmatched_qty:.0f}sh still open, total position: {pos['quantity']:.0f}sh, MktVal: ${pos['marketValue']:,.2f}]"
                else:
                    unmatched_qty = sum(t["quantity"] for t in unmatched_assignments[underlying])
                    annotation = f" [Unmatched Assignment: {unmatched_qty:.0f} shares - position may have been sold]"
            
            print(f"{underlying}: {'Inflow' if cash_flow >= 0 else 'Outflow'} ${abs(cash_flow):.2f}{annotation}")
    
        # Calculate overall cash flow
        total_cash_flow = sum(cash_flow_by_underlying.values())
        print(f"\nOverall Total Cash Flow: {'Inflow' if total_cash_flow >= 0 else 'Outflow'} ${abs(total_cash_flow):.2f}")
    
        if generate_plot:
            # Generate bar chart for total cash flow by underlying asset
            underlying_assets = list(cash_flow_by_underlying.keys())
            cash_flows = list(cash_flow_by_underlying.values())
    
            plt.figure(figsize=(12, 7))
            plt.bar(underlying_assets, cash_flows, color='skyblue', label='Total Cash Flow')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('Underlying Asset')
            plt.ylabel('Cash Flow ($)')
            plt.title('Total Cash Flow by Underlying Asset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            plt.show()    

    def option_gain(self, start_date: str, generate_plot=False):
        """
        Calculate detailed daily and overall gains or losses for closed option trades since the given start date.
        
        This function operates in two phases. First, it processes open orders to build a cost basis tracker and
        calculates daily cash flows. Second, it processes closed trades to match the filled quantities against
        the stored cost basis entries, computing gains or losses for each trade and then aggregating these per day.
        The final output shows per-day gains/losses as well as the overall gain/loss. A plot can also be
        generated to visualize the daily cash flow.
    
        :param start_date: The date (YYYY-MM-DD) from which to start calculating gains/losses for closed trades.
        :param generate_plot: If True, display a bar chart for daily cash flow.
        :return: None. Prints trade details, daily gains/losses, and overall gain/loss.
        """
        # ...existing code...
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        six_months_ago_dt = start_date_dt - timedelta(days=90)
        six_months_ago = six_months_ago_dt.strftime('%m%d%Y')
        from_date = start_date_dt.strftime('%m%d%Y')
        to_date = datetime.now().strftime('%m%d%Y')

        # Base URL and Headers
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        headers = {"consumerKey": self.consumer_key}

        # Break the six-month range into 5-day chunks
        date_ranges = []
        current_start = six_months_ago_dt
        while current_start < datetime.now().date():
            current_end = min(current_start + timedelta(days=5), datetime.now().date())
            date_ranges.append((current_start.strftime('%m%d%Y'), current_end.strftime('%m%d%Y')))
            current_start = current_end + timedelta(days=1)

        # Aggregate open orders data
        open_orders_data = {"OrdersResponse": {"Order": []}}

        for start, end in date_ranges:
            params_open_trades = {
                "status": "EXECUTED",
                "count": 100,
                "fromDate": start,
                "toDate": end,
                "securityType": "OPTN",
            }

            response_open = self.session.get(url, params=params_open_trades, headers=headers)
            if response_open.status_code != 200:
                logger.error("Failed to fetch open orders. Status Code: %s, Response: %s", response_open.status_code, response_open.text)
                continue

            # Parse and append data
            chunk_data = response_open.json()
            if "OrdersResponse" in chunk_data and "Order" in chunk_data["OrdersResponse"]:
                open_orders_data["OrdersResponse"]["Order"].extend(chunk_data["OrdersResponse"]["Order"])

        # Process open orders data
        open_orders = open_orders_data.get("OrdersResponse", {}).get("Order", [])

        # Cost basis tracker using both symbol and quantity as keys
        cost_basis_tracker = {}
        daily_cash_flow = defaultdict(float)  # New: Track daily cash flow

        # Process open trades to track cost basis
        for order in open_orders:
            for detail in order.get("OrderDetail", []):
                if detail.get("status") != "EXECUTED":
                    continue
                for instrument in detail.get("Instrument", []):
                    order_action = instrument["orderAction"]
                    if instrument['Product']['securityType'] != "OPTN":
                        continue
                    product_id = instrument["Product"]["productId"]["symbol"]
                    filled_quantity = float(instrument["filledQuantity"])
                    average_execution_price = instrument.get("averageExecutionPrice")
                    if average_execution_price is None:
                        print(f"Missing averageExecutionPrice for instrument: {product_id}")
                        continue

                    executed_time = datetime.fromtimestamp(detail["executedTime"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    executed_date = executed_time.split(" ")[0]

                    # Calculate trade amount
                    trade_amount = filled_quantity * average_execution_price * 100  # Options contract multiplier
                    if order_action in ["SELL_OPEN", "SELL_CLOSE"]:
                        daily_cash_flow[executed_date] += trade_amount  # Positive cash flow
                        if order_action == "SELL_OPEN":
                            trade_amount = -trade_amount  # Negative cost basis for SELL_OPEN
                    if order_action in ["BUY_OPEN", "BUY_CLOSE"]:
                        daily_cash_flow[executed_date] -= trade_amount  # Negative cash flow
                        if order_action == "BUY_OPEN":
                            trade_amount = trade_amount  # Positive cost basis for SELL_OPEN

                    if order_action in ["BUY_OPEN", "SELL_OPEN"]:
                        # Handle opening trades
                        cost_basis_key = product_id
                        if cost_basis_key not in cost_basis_tracker:
                            cost_basis_tracker[cost_basis_key] = []  # Initialize as a list if not already present

                        # Append the new entry to the list for this product_id
                        cost_basis_tracker[cost_basis_key].append({
                            "quantity": filled_quantity,
                            "cost_basis": trade_amount,
                        })

                        print(f"Added to cost_basis_tracker: {cost_basis_key} - Quantity: {filled_quantity}, Cost Basis: ${trade_amount:.2f} {executed_time}")
        # Fetch executed orders starting from start_date for closed trades
        params_closed_trades = {
            "status": "EXECUTED",
            "count": 100,
            "fromDate": from_date,
            "toDate": datetime.now().strftime('%m%d%Y'),
            "securityType": "OPTN",
        }

        response_closed = self.session.get(url, params=params_closed_trades, headers=headers)
        if response_closed.status_code != 200:
            logger.error("Failed to fetch closed orders. Status Code: %s, Response: %s", response_closed.status_code, response_closed.text)
            return

        closed_orders_data = response_closed.json()
        closed_orders = closed_orders_data.get("OrdersResponse", {}).get("Order", [])

        # Dictionary to store information about closed trades
        close_trades = {}

        # Calculate gains/losses for closed trades
        for order in closed_orders:
            for detail in order.get("OrderDetail", []):
                if detail.get("status") != "EXECUTED":
                    continue

                for instrument in detail.get("Instrument", []):
                    product_id = instrument["Product"]["productId"]["symbol"]
                    filled_quantity = float(instrument["filledQuantity"])
                    order_action = instrument["orderAction"]
                    if order_action not in ["SELL_CLOSE", "BUY_CLOSE"]:
                        continue
                    average_execution_price = instrument.get("averageExecutionPrice", 0)

                    # Calculate trade amount
                    trade_amount = filled_quantity * average_execution_price * 100  # Options contract multiplier
                    if order_action == "SELL_CLOSE":
                        trade_amount = trade_amount  # Positive for SELL_CLOSE
                    if order_action == "BUY_CLOSE":
                        trade_amount = -trade_amount  # Negative for BUY

                    cost_basis_key = product_id
                    if cost_basis_key not in cost_basis_tracker:
                        executed_time = datetime.fromtimestamp(detail["executedTime"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"No cost basis available for {product_id} with quantity {filled_quantity}. {executed_time} Skipping gain/loss calculation")
                        continue

                    total_cost_basis = 0
                    remaining_quantity = filled_quantity

                    print(f"Processing order: {product_id},{filled_quantity}")
                    while remaining_quantity > 0:
                        if not cost_basis_tracker[cost_basis_key]:
                            print(f"Insufficient cost basis entries for {product_id} to match filled quantity {filled_quantity}.")
                            breakpoint()
                            break

                        cost_basis_entry = cost_basis_tracker[cost_basis_key][0]
                        entry_quantity = cost_basis_entry["quantity"]
                        entry_cost_basis = cost_basis_entry["cost_basis"]

                        if entry_quantity > remaining_quantity:
                            proportional_cost = (remaining_quantity / entry_quantity) * entry_cost_basis
                            total_cost_basis += proportional_cost
                            cost_basis_tracker[cost_basis_key][0]["quantity"] -= remaining_quantity
                            print(f"{cost_basis_entry[cost_basis_key]},entry_quantity: {entry_quantity}, remaining_quantity: {remaining_quantity}")
                            remaining_quantity = 0
                        else:
                            print(f"removing entry: {cost_basis_key}, {cost_basis_tracker[cost_basis_key]},remain: {remaining_quantity}, filled qty: {filled_quantity}, entry_quantity: {entry_quantity}")
                            total_cost_basis += entry_cost_basis
                            remaining_quantity -= entry_quantity
                            print(f"cost basis tracker entry before pop: {cost_basis_tracker[cost_basis_key]}")
                            cost_basis_tracker[cost_basis_key].pop(0)
                            print(f"cost basis tracker entry after pop: {cost_basis_tracker[cost_basis_key]}")

                        # Check if the cost_basis_key exists and process it
                        if cost_basis_key in cost_basis_tracker:
                            if not cost_basis_tracker[cost_basis_key]:  # Check if the list is empty
                                del cost_basis_tracker[cost_basis_key]  # Safely delete the key if empty
                        else:
                            print(f"Cost basis key '{cost_basis_key}' does not exist in the tracker. Skipping.")
                            continue

                    if remaining_quantity > 0:
                        print(f"Not enough cost basis entries for {product_id} to match filled quantity {filled_quantity}. Remaining quantity: {remaining_quantity}")
                        continue

                    gain_loss = trade_amount - total_cost_basis

                    # Store trade details in close_trades
                    executed_time = datetime.fromtimestamp(detail["executedTime"] / 1000).strftime('%Y-%m-%d %H:%M:%S')

                    close_trade_key = (
                        product_id,
                        executed_time,  # Include executed_time for complete uniqueness
                    )
                    # Store the closed trade details in the dictionary
                    close_trades[close_trade_key] = {
                        "action": order_action,
                        "quantity": filled_quantity,
                        "executed_time": executed_time,
                        "trade_amount": trade_amount,
                        "cost_basis": total_cost_basis,
                        "gain_loss": gain_loss,
                    }

        # Print all close trades
        for trade_key, trade_info in close_trades.items():
            print(f"Trade Details for {trade_key}:")
            for key, value in trade_info.items():
                print(f"  {key}: {value}")

        # Calculate total gain/loss
        total_gain_loss = sum(trade_info["gain_loss"] for trade_info in close_trades.values())
        print(f"\nTotal Gain/Loss: {'Gain' if total_gain_loss >= 0 else 'Loss'} ${abs(total_gain_loss):.2f}")

        # Aggregate daily gain/loss
        daily_gain_loss = defaultdict(float)
        for trade_info in close_trades.values():
            trade_date = trade_info["executed_time"].split(" ")[0]  # Extract date only
            daily_gain_loss[trade_date] += trade_info["gain_loss"]

        # Sort daily gain/loss by date
        sorted_daily_gain_loss = dict(sorted(daily_gain_loss.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d')))

        # Print daily gain/loss
        print("\nDaily Gain/Loss:")
        for date, gain_loss in sorted_daily_gain_loss.items():
            print(f"{date}: {'Gain' if gain_loss >= 0 else 'Loss'} ${abs(gain_loss):.2f}")

        # Plot daily gain/loss
        dates = list(sorted_daily_gain_loss.keys())
        values = list(sorted_daily_gain_loss.values())

        # Filter daily cash flow to include only dates on or after start_date
        filtered_daily_cash_flow = {
            date: cash_flow
            for date, cash_flow in sorted(daily_cash_flow.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
            if datetime.strptime(date, '%Y-%m-%d').date() >= start_date_dt
        }

        # Print filtered daily cash flow
        print("\nFiltered Daily Cash Flow:")
        for date, cash_flow in filtered_daily_cash_flow.items():
            print(f"{date}: {'Inflow' if cash_flow >= 0 else 'Outflow'} ${abs(cash_flow):.2f}")

        # Calculate average daily cash flow for filtered dates
        total_cash_flow = sum(filtered_daily_cash_flow.values())
        average_cash_flow = sum(filtered_daily_cash_flow.values()) / len(filtered_daily_cash_flow) if filtered_daily_cash_flow else 0
        print(f"\nTotal cash flow: ${total_cash_flow:.2f} Average Daily Cash Flow (filtered): ${average_cash_flow:.2f}")

        if generate_plot == True:
            # Generate bar chart for filtered daily cash flow
            dates = list(filtered_daily_cash_flow.keys())
            cash_flows = list(filtered_daily_cash_flow.values())

            plt.figure(figsize=(12, 7))
            plt.bar(dates, cash_flows, color='skyblue', label='Daily Cash Flow')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Cash Flow ($)')
            plt.title('Filtered Daily Cash Flow')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            plt.show()

    def cancel_all_order(self):
        """
        Queries for all open orders and cancels them after user confirmation.
        
        :param self: Pass parameter with authenticated session and information on selected account
        :return: None
        """
        reject_legacy_execution("order.order_bo.Order.cancel_all_order")

    def refresh_order_limit_old(self):
        """
        Refreshes the limit price of all open spread orders by adjusting NET_CREDIT orders down by 0.01
        and NET_DEBIT orders up by 0.01 every 10 seconds until no open orders remain.
        Uses the E*TRADE API v1 change order endpoint.
        """
        reject_legacy_execution("order.order_bo.Order.refresh_order_limit_old")



    def refresh_order_limit(self):
        """
        Refreshes the limit prices of open spread orders by adjusting them slightly.
        Incorporates the preview step before placing the order change as per E*TRADE API v1.
        """
        reject_legacy_execution("order.order_bo.Order.refresh_order_limit")
