from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
import configparser
import random
import re
from datetime import datetime,date,timedelta
import numpy as np
from accounts.accounts_bo import StockPosition
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random
import logging
import requests
from xml.etree import ElementTree as ET

# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# logger settings
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("python_client.log", maxBytes=5 * 1024 * 1024, backupCount=3)
FORMAT = "%(asctime)-15s %(message)s"
fmt = logging.Formatter(FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)


class Order:

    def __init__(self, session, account, base_url, use_sandbox):
        self.session = session
        self.account = account
        self.base_url = base_url
        self.use_sandbox = use_sandbox
        if self.use_sandbox:
            self.consumer_key = config["DEFAULT"]["SANDBOX_CONSUMER_KEY"]
        else: 
            self.consumer_key = config["DEFAULT"]["PROD_CONSUMER_KEY"]

    def preview_order(self, order):
        """
        Preview an order by calling the E*TRADE preview order API.

        Parameters:
        - order (dict): Dictionary containing the order details.

        Returns:
        - dict: Response from the preview order API, or None if the preview fails.
        """
        # URL for the API endpoint
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/preview"

        # Headers
        headers = {
            "Content-Type": "application/xml",
            "Accept": "application/json",
            "consumerKey": self.consumer_key,
        }

        # Determine if the order is for an option or a spread
        is_option = order["securityType"] == "OPTN"
        is_spread = order.get("orderAction") == "SPREAD"

        if not is_option:
            # Payload for an equity order
            payload = f"""<?xml version="1.0" encoding="UTF-8"?>
            <PreviewOrderRequest>
                <orderType>EQ</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    <stopPrice></stopPrice>
                    <limitPrice>{order.get('limitPrice', '')}</limitPrice>
                    <Instrument>
                        <Product>
                            <securityType>EQ</securityType>
                            <symbol>{order['symbol']}</symbol>
                        </Product>
                        <orderAction>{order['orderAction']}</orderAction>
                        <quantityType>QUANTITY</quantityType>
                        <quantity>{abs(order['quantity'])}</quantity>
                    </Instrument>
                </Order>
            </PreviewOrderRequest>
            """
        elif is_spread:
            # Payload for a spread order with multiple legs
            legs_xml = ""
            for leg in order["legs"]:
                if leg['symbol'] == "BRK.B":
                    leg['symbol'] = "BRKB"

                leg_xml = f"""
                <Instrument>
                    <Product>
                        <securityType>OPTN</securityType>
                        <symbol>{leg['symbol']}</symbol>
                        <callPut>{leg['callPut']}</callPut>
                        <expiryYear>{leg['expiryYear']}</expiryYear>
                        <expiryMonth>{leg['expiryMonth']}</expiryMonth>
                        <expiryDay>{leg['expiryDay']}</expiryDay>
                        <strikePrice>{leg['strikePrice']}</strikePrice>
                    </Product>
                    <orderAction>{leg['orderAction']}</orderAction>
                    <quantityType>QUANTITY</quantityType>
                    <quantity>{abs(leg['quantity'])}</quantity>
                </Instrument>
                """
                legs_xml += leg_xml
            if order['spreadType'] == "VERTICAL":
                spread_or_optn = 'SPREADS'
            else:
                spread_or_optn = 'OPTN'
            payload = f"""<?xml version="1.0" encoding="UTF-8"?>
            <PreviewOrderRequest>
                <orderType>{spread_or_optn}</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <limitPrice>{order['limitPrice']}</limitPrice>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    {legs_xml}
                </Order>
            </PreviewOrderRequest>
            """
        else:
            # Payload for a single-leg option order
            if order['symbol'] == "BRK.B":
                order['symbol'] = "BRKB"
            payload = f"""<?xml version="1.0" encoding="UTF-8"?>
            <PreviewOrderRequest>
                <orderType>OPTN</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <limitPrice>{order['limitPrice']}</limitPrice>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    <Instrument>
                        <Product>
                            <securityType>OPTN</securityType>
                            <symbol>{order['symbol']}</symbol>
                            <callPut>{order['callPut']}</callPut>
                            <expiryYear>{order['expiryYear']}</expiryYear>
                            <expiryMonth>{order['expiryMonth']}</expiryMonth>
                            <expiryDay>{order['expiryDay']}</expiryDay>
                            <strikePrice>{order['strikePrice']}</strikePrice>
                        </Product>
                        <orderAction>{order['orderAction']}</orderAction>
                        <quantityType>QUANTITY</quantityType>
                        <quantity>{abs(order['quantity'])}</quantity>
                    </Instrument>
                </Order>
            </PreviewOrderRequest>
            """
        # Make the API call for POST request
        response = self.session.post(url, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)

        # Print the status code and response text for debugging
        if response.status_code != 200:
            print("Preview order Status Code:", response.status_code)
            if "insufficient" in response.text.lower():
                print("Insufficient funds to place order.",order)
                return "INSUFFICIENT_FUNDS"
            elif "create an unapproved options" in response.text.lower():
                print("Uncovered option position.",order)
                return "UNCOVERED_OPTION"
            else:
                print(order)
                print(response.text)
                print(payload)

        # Handle and parse response
        if response and response.status_code == 200:
            data = response.json()
            logger.debug("Response Body: %s", json.dumps(data, indent=4, sort_keys=True))

            if data and "PreviewOrderResponse" in data and "PreviewIds" in data["PreviewOrderResponse"]:
                for preview_id in data["PreviewOrderResponse"]["PreviewIds"]:
                    print("Preview ID: " + str(preview_id["previewId"]))

                # Adding preview_id to order for further processing
                data["PreviewOrderResponse"]["preview_id"] = preview_id["previewId"]
                return data["PreviewOrderResponse"]

            else:
                # Handle errors
                if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"]:
                    print("Error1: " + data["Error"]["message"])
                else:
                    print("Error1: Preview Order API service error")
        else:
            # Handle errors
            if response:
                try:
                    data = response.json()
                    if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"]:
                        print("Error3: " + data["Error"]["message"])
                    else:
                        print("Error3: Preview Order API service error")
                except ValueError:
                    print("Error parsing response as JSON.")


    def preview_order_old(self, order):
        """
        Call preview order API based on selecting from different given options

        :param self: Pass in authenticated session and information on selected account
        """
        # User's order selection
        # order = self.user_select_order()

        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders/preview.json"
        # Add parameters and header information
        headers = {"Content-Type": "application/xml", "consumerKey": self.consumer_key}

        # Add payload for POST Request
        payload = """<PreviewOrderRequest>
                       <orderType>EQ</orderType>
                       <clientOrderId>{0}</clientOrderId>
                       <Order>
                           <allOrNone>false</allOrNone>
                           <priceType>{1}</priceType>
                           <orderTerm>{2}</orderTerm>
                           <marketSession>REGULAR</marketSession>
                           <stopPrice></stopPrice>
                           <limitPrice>{3}</limitPrice>
                           <Instrument>
                               <Product>
                                   <securityType>EQ</securityType>
                                   <symbol>{4}</symbol>
                               </Product>
                               <orderAction>{5}</orderAction>
                               <quantityType>QUANTITY</quantityType>
                               <quantity>{6}</quantity>
                           </Instrument>
                       </Order>
                   </PreviewOrderRequest>"""
        payload = payload.format(order["client_order_id"], order["priceType"], order["orderTerm"],
                                 order["limitPrice"], order["symbol"], order["orderAction"], order["quantity"])

        # Make API call for POST request
        print(payload)
        response = self.session.post(url, header_auth=True, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)
        print(response)

        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            print("\nPreview Order:")

            if data is not None and "PreviewOrderResponse" in data and "PreviewIds" in data["PreviewOrderResponse"]:
                for previewids in data["PreviewOrderResponse"]["PreviewIds"]:
                    print("Preview ID: " + str(previewids["previewId"]))
            else:
                # Handle errors
                data = response.json()
                if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                    print("Error1: " + data["Error"]["message"])
                else:
                    print("Error1: Preview Order API service error")

            if data is not None and "PreviewOrderResponse" in data and "Order" in data["PreviewOrderResponse"]:
                for orders in data["PreviewOrderResponse"]["Order"]:
                    # order["limitPrice"] = orders["limitPrice"]

                    if orders is not None and "Instrument" in orders:
                        for instrument in orders["Instrument"]:
                            if instrument is not None and "orderAction" in instrument:
                                print("Action: " + instrument["orderAction"])
                            if instrument is not None and "quantity" in instrument:
                                print("Quantity: " + str(instrument["quantity"]))
                            if instrument is not None and "Product" in instrument \
                                    and "symbol" in instrument["Product"]:
                                print("Symbol: " + instrument["Product"]["symbol"])
                            if instrument is not None and "symbolDescription" in instrument:
                                print("Description: " + str(instrument["symbolDescription"]))

                if orders is not None and "priceType" in orders and "limitPrice" in orders:
                    print("Price Type: " + orders["priceType"])
                    if orders["priceType"] == "MARKET":
                        print("Price: MKT")
                    else:
                        print("Price: " + str(orders["limitPrice"]))
                if orders is not None and "orderTerm" in orders:
                    print("Duration: " + orders["orderTerm"])
                if orders is not None and "estimatedCommission" in orders:
                    print("Estimated Commission: " + str(orders["estimatedCommission"]))
                if orders is not None and "estimatedTotalAmount" in orders:
                    print("Estimated Total Cost: " + str(orders["estimatedTotalAmount"]))

                orders['preview_id'] = previewids["previewId"]
                return orders
            else:
                # Handle errors
                data = response.json()
                if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                    print("Error2: " + data["Error"]["message"])
                else:
                    print("Error2: Preview Order API service error")
        else:
            # Handle errors
            data = response.json()
            if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                print("Error3: " + data["Error"]["message"])
                if "Hard to Borrow" in data["Error"]["message"]:
                    return 'Hard_to_Borrow'
            else:
                print("Error3: Preview Order API service error")

    def place_order(self, order, preview_only=False):
        """
        Place an order based on the preview order response.

        Parameters:
        - order (dict): Dictionary containing the order details.

        Returns:
        - str: The order ID if the order is placed successfully, or None if it fails.
        """
        # Preview the order first to get the preview ID
        preview_order = self.preview_order(order)
        if preview_order == "INSUFFICIENT_FUNDS" or preview_order == "UNCOVERED_OPTION":
            print(f"Cannot preview order due to {preview_order}")
            return preview_order
        if not preview_order or "preview_id" not in preview_order:
            print("Preview order failed. Cannot place order.")
            return
        else:
            order["preview_id"] = preview_order["preview_id"]

        if preview_only == True:
            # print(preview_order)
            return preview_order["preview_id"]

        # URL for the API endpoint
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/place"

        # Headers
        headers = {
            "Content-Type": "application/xml",
            "Accept": "application/json",
            "consumerKey": self.consumer_key,
        }

        # Determine if the order is for an option, equity, or spread
        is_option = order["securityType"] == "OPTN"
        is_spread = order.get("orderAction") == "SPREAD"

        # Generate payload
        if not is_option:
            # Payload for an equity order
            payload = f"""
            <PlaceOrderRequest>
                <orderType>EQ</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <PreviewIds>
                    <previewId>{order['preview_id']}</previewId>
                </PreviewIds>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    <limitPrice>{order.get('limitPrice', '')}</limitPrice>
                    <Instrument>
                        <Product>
                            <securityType>EQ</securityType>
                            <symbol>{order['symbol']}</symbol>
                        </Product>
                        <orderAction>{order['orderAction']}</orderAction>
                        <quantityType>QUANTITY</quantityType>
                        <quantity>{order['quantity']}</quantity>
                    </Instrument>
                </Order>
            </PlaceOrderRequest>
            """
        elif is_spread:
            # Payload for a spread order with multiple legs
            legs_xml = ""
            for leg in order["legs"]:
                leg_xml = f"""
                <Instrument>
                    <Product>
                        <securityType>OPTN</securityType>
                        <symbol>{leg['symbol']}</symbol>
                        <callPut>{leg['callPut']}</callPut>
                        <expiryYear>{leg['expiryYear']}</expiryYear>
                        <expiryMonth>{leg['expiryMonth']}</expiryMonth>
                        <expiryDay>{leg['expiryDay']}</expiryDay>
                        <strikePrice>{leg['strikePrice']}</strikePrice>
                    </Product>
                    <orderAction>{leg['orderAction']}</orderAction>
                    <quantityType>QUANTITY</quantityType>
                    <quantity>{abs(leg['quantity'])}</quantity>
                </Instrument>
                """
                legs_xml += leg_xml

            if order['spreadType'] == "VERTICAL":
                spread_or_optn = 'SPREADS'
            else:
                spread_or_optn = 'OPTN'

            payload = f"""<?xml version="1.0" encoding="UTF-8"?>
            <PlaceOrderRequest>
                <orderType>{spread_or_optn}</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <PreviewIds>
                    <previewId>{order['preview_id']}</previewId>
                </PreviewIds>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <limitPrice>{order['limitPrice']}</limitPrice>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    {legs_xml}
                </Order>
            </PlaceOrderRequest>
            """
        else:
            # Payload for a single-leg option order
            if order['symbol'] == "BRK.B":
                order['symbol'] = "BRKB"
            payload = f"""
            <PlaceOrderRequest>
                <orderType>OPTN</orderType>
                <clientOrderId>{order['client_order_id']}</clientOrderId>
                <PreviewIds>
                    <previewId>{order['preview_id']}</previewId>
                </PreviewIds>
                <Order>
                    <allOrNone>false</allOrNone>
                    <priceType>{order['priceType']}</priceType>
                    <limitPrice>{order['limitPrice']}</limitPrice>
                    <orderTerm>{order['orderTerm']}</orderTerm>
                    <marketSession>REGULAR</marketSession>
                    <Instrument>
                        <Product>
                            <securityType>OPTN</securityType>
                            <symbol>{order['symbol']}</symbol>
                            <callPut>{order['callPut']}</callPut>
                            <expiryYear>{order['expiryYear']}</expiryYear>
                            <expiryMonth>{order['expiryMonth']}</expiryMonth>
                            <expiryDay>{order['expiryDay']}</expiryDay>
                            <strikePrice>{order['strikePrice']}</strikePrice>
                        </Product>
                        <orderAction>{order['orderAction']}</orderAction>
                        <quantityType>QUANTITY</quantityType>
                        <quantity>{abs(order['quantity'])}</quantity>
                    </Instrument>
                </Order>
            </PlaceOrderRequest>
            """

        # Make API call for POST request
        response = self.session.post(url, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)

        if response.status_code != 200:
            print("Preview order Status Code:", response.status_code)
            if "insufficient" in response.text.lower():
                print("Insufficient funds to place order.",order)
                return "INSUFFICIENT_FUNDS"
            elif "uncovered" in response.text.lower():
                print("Uncovered option position.",order)
                return "UNCOVERED_OPTION"
            else:
                print(order)
                print(response.text)
                print(payload)

        # Handle and parse response
        if response and response.status_code == 200:
            data = response.json()
            logger.debug("Response Body: %s", json.dumps(data, indent=4, sort_keys=True))
            print("\nPlace Order:")

            if data and "PlaceOrderResponse" in data and "OrderIds" in data["PlaceOrderResponse"]:
                for order_ids in data["PlaceOrderResponse"]["OrderIds"]:
                    print("Order ID: " + str(order_ids["orderId"]))
                    return order_ids["orderId"]
            else:
                print("Error: Place Order API service error")
        else:
            # Handle errors
            try:
                error_response = response.json()
                print("Error: ", error_response.get("Error", {}).get("message", "Unknown error"))
            except ValueError:
                print("Error parsing response as JSON.")

        return None

    def place_order_old(self, order):
        """
        Place order based on the preview order response
        """
        # Preview the order first to get the preview ID
        previw_order = self.preview_order(order)
        if "preview_id" not in previw_order:
            print("Preview order failed. Cannot place order.")
            return
        else:
            order["preview_id"] = previw_order["preview_id"]

        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders/place.json"

        # Add parameters and header information
        headers = {"Content-Type": "application/xml", "consumerKey": self.consumer_key}

        # Add payload for POST Request
        payload = """<PlaceOrderRequest>
                       <orderType>EQ</orderType>
                       <clientOrderId>{0}</clientOrderId>
                       <PreviewIds>
                           <previewId>{1}</previewId>
                       </PreviewIds>
                       <Order>
                           <allOrNone>false</allOrNone>
                           <priceType>{2}</priceType>
                           <orderTerm>{3}</orderTerm>
                           <marketSession>REGULAR</marketSession>
                           <stopPrice></stopPrice>
                           <limitPrice>{4}</limitPrice>
                           <Instrument>
                               <Product>
                                   <securityType>EQ</securityType>
                                   <symbol>{5}</symbol>
                               </Product>
                               <orderAction>{6}</orderAction>
                               <quantityType>QUANTITY</quantityType>
                               <quantity>{7}</quantity>
                           </Instrument>
                       </Order>
                   </PlaceOrderRequest>"""
        payload = payload.format(order["client_order_id"], order["preview_id"], order["priceType"],
                                 order["orderTerm"], order["limitPrice"], order["symbol"], order["orderAction"], order["quantity"])

        # Make API call for POST request
        response = self.session.post(url, header_auth=True, headers=headers, data=payload)
        logger.debug("Request Header: %s", response.request.headers)
        logger.debug("Request payload: %s", payload)

        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            print("\nPlace Order:")

            if data is not None and "PlaceOrderResponse" in data and "OrderIds" in data["PlaceOrderResponse"]:
                for order_ids in data["PlaceOrderResponse"]["OrderIds"]:
                    print("Order ID: " + str(order_ids["orderId"]))
                    order_id = order_ids["orderId"]
            else:
                # Handle errors
                data = response.json()
                if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                    print("Error1: " + data["Error"]["message"])
                else:
                    print("Error1: Place Order API service error")

            if data is not None and "PlaceOrderResponse" in data and "Order" in data["PlaceOrderResponse"]:
                for orders in data["PlaceOrderResponse"]["Order"]:
                    if orders is not None and "Instrument" in orders:
                        for instrument in orders["Instrument"]:
                            if instrument is not None and "orderAction" in instrument:
                                print("Action: " + instrument["orderAction"])
                            if instrument is not None and "quantity" in instrument:
                                print("Quantity: " + str(instrument["quantity"]))
                            if instrument is not None and "Product" in instrument \
                                    and "symbol" in instrument["Product"]:
                                print("Symbol: " + instrument["Product"]["symbol"])
                            if instrument is not None and "symbolDescription" in instrument:
                                print("Description: " + str(instrument["symbolDescription"]))

                if orders is not None and "priceType" in orders and "limitPrice" in orders:
                    print("Price Type: " + orders["priceType"])
                    if orders["priceType"] == "MARKET":
                        print("Price: MKT")
                    else:
                        print("Price: " + str(orders["limitPrice"]))
                if orders is not None and "orderTerm" in orders:
                    print("Duration: " + orders["orderTerm"])
                if orders is not None and "estimatedCommission" in orders:
                    print("Estimated Commission: " + str(orders["estimatedCommission"]))
                if orders is not None and "estimatedTotalAmount" in orders:
                    print("Estimated Total Cost: " + str(orders["estimatedTotalAmount"]))
                return order_id
            else:
                # Handle errors
                data = response.json()
                if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                    print("Error2: " + data["Error"]["message"])
                else:
                    print("Error2: Place Order API service error")
        else:
            # Handle errors
            data = response.json()
            if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                print("Error3: " + data["Error"]["message"])
            else:
                print("Error3: Place Order API service error")

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
        new_client_order_id = str(random.randint(1000000000, 9999999999))
        price_type = order_detail.get("priceType", "LIMIT")
        order_term = order_detail.get("orderTerm", "GOOD_FOR_DAY")
        instruments_xml = self._build_instruments_xml_from_detail(order_detail)

        preview_payload = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <PreviewOrderRequest>
            <orderType>{order_type}</orderType>
            <clientOrderId>{new_client_order_id}</clientOrderId>
            <Order>
                <allOrNone>false</allOrNone>
                <priceType>{price_type}</priceType>
                <limitPrice>{new_limit_price:.2f}</limitPrice>
                <orderTerm>{order_term}</orderTerm>
                <marketSession>REGULAR</marketSession>{instruments_xml}
            </Order>
        </PreviewOrderRequest>"""

        preview_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/preview"
        preview_headers = {"Content-Type": "application/xml", "Accept": "application/json", "consumerKey": self.consumer_key}
        pr = self.session.post(preview_url, headers=preview_headers, data=preview_payload)
        if pr.status_code != 200:
            print(f"Failed to preview change for order {order_id}: {pr.text}")
            return False, None
        # Try JSON first; fall back to XML if needed
        preview_id = None
        try:
            data = pr.json()
            por = data.get("PreviewOrderResponse") if isinstance(data, dict) else None
            if por:
                ids = por.get("PreviewIds")
                if isinstance(ids, list) and ids:
                    preview_id = ids[0].get("previewId")
                elif isinstance(ids, dict):
                    preview_id = ids.get("previewId")
        except ValueError:
            pass
        if not preview_id:
            try:
                root = ET.fromstring(pr.text)
                preview_id_elem = root.find(".//previewId")
                if preview_id_elem is not None:
                    preview_id = preview_id_elem.text
            except ET.ParseError:
                pass
        if not preview_id:
            print(f"Failed to parse preview response for order {order_id}: {pr.text}")
            return False, None

        place_payload = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <PlaceOrderRequest>
            <orderType>{order_type}</orderType>
            <clientOrderId>{new_client_order_id}</clientOrderId>
            <PreviewIds>
                <previewId>{preview_id}</previewId>
            </PreviewIds>
            <Order>
                <allOrNone>false</allOrNone>
                <priceType>{price_type}</priceType>
                <limitPrice>{new_limit_price:.2f}</limitPrice>
                <orderTerm>{order_term}</orderTerm>
                <marketSession>REGULAR</marketSession>
                {instruments_xml}
            </Order>
        </PlaceOrderRequest>"""

        place_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/{order_id}/change/place"
        place_headers = {"Content-Type": "application/xml", "Accept": "application/json", "consumerKey": self.consumer_key}
        pl = self.session.put(place_url, headers=place_headers, data=place_payload)
        if pl.status_code == 200:
            new_id = None
            try:
                data = pl.json()
                # Try common shapes
                for key in ("PlaceChangeOrderResponse", "PlaceOrderResponse"):
                    if key in data and "OrderIds" in data[key]:
                        ids = data[key]["OrderIds"]
                        if isinstance(ids, list) and ids:
                            new_id = ids[0].get("orderId")
                        elif isinstance(ids, dict):
                            new_id = ids.get("orderId")
                        break
            except ValueError:
                # Non-JSON; ignore
                pass
            print(f"Updated order {order_id} to new limit {new_limit_price:.2f}{' (new id ' + str(new_id) + ')' if new_id else ''}")
            return True, new_id
        print(f"Failed to update order {order_id}: {pl.text}")
        return False, None

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
        checks = 0
        # Allow the order some time before the first adjustment
        if initial_wait:
            time.sleep(interval_sec)
        while True:
            od = self.get_order_details_by_id(order_id)
            if not od:
                print(f"Order {order_id}: details not found; will retry...")
                checks += 1
                if checks >= max_checks:
                    print(f"Order {order_id}: giving up after {max_checks} checks")
                    return False, order_id
                time.sleep(interval_sec)
                continue

            status = None
            detail = None
            for d in od.get("OrderDetail", []):
                status = d.get("status")
                detail = d
                break

            if status in ("EXECUTED", "FILLED"):
                print(f"Order {order_id}: executed.")
                return True, order_id
            if status in ("CANCELLED", "REJECTED", "EXPIRED"):
                # If it was cancelled due to replace, follow the new order id
                repl_id = self._find_replacement_order_id(order_id)
                if repl_id:
                    print(f"Order {order_id}: status {status}; continuing with replacement order {repl_id}.")
                    order_id = repl_id
                    checks += 1
                    if checks >= max_checks:
                        print(f"Order {order_id}: reached max checks {max_checks}; stopping.")
                        return
                    time.sleep(interval_sec)
                    continue
                print(f"Order {order_id}: status {status}; stopping adjustments.")
                return False, order_id

            price_type = detail.get("priceType", "LIMIT")
            limit_price = float(detail.get("limitPrice", 0) or 0)
            order_type = od.get("orderType", "SPREADS")
            # Determine bump direction
            if price_type == "NET_CREDIT":
                new_limit = max(0.01, limit_price - step)
            elif price_type == "NET_DEBIT":
                new_limit = limit_price + step
            else:
                # Infer from action: any SELL -> decrease, otherwise increase
                actions = [i.get('orderAction','') for i in detail.get('Instrument', [])]
                if any(a.startswith('SELL') for a in actions):
                    new_limit = max(0.01, limit_price - step)
                else:
                    new_limit = limit_price + step

            ok, new_id = self.change_order_limit(order_id, new_limit, detail, order_type)
            if ok and new_id:
                order_id = new_id
            checks += 1
            if checks >= max_checks:
                print(f"Order {order_id}: reached max checks {max_checks}; stopping.")
                return False, order_id
            time.sleep(interval_sec)

    def previous_order(self, session, account, prev_orders):
        """
        Calls preview order API based on a list of previous orders

        :param session: authenticated session
        :param account: information on selected account
        :param prev_orders: list of instruments from previous orders
        """

        if prev_orders is not None:
            while True:

                # Display previous instruments for user selection
                print("")
                count = 1
                for order in prev_orders:
                    print(str(count) + ")\tOrder Action: " + order["orderAction"] + " | "
                          + "Security Type: " + str(order["security_type"]) + " | "
                          + "Term: " + str(order["orderTerm"]) + " | "
                          + "Quantity: " + str(order["quantity"]) + " | "
                          + "Symbol: " + order["symbol"] + " | "
                          + "Price Type: " + order["priceType"])
                    count = count + 1
                print(str(count) + ")\t" "Go Back")
                options_select = input("Please select an option: ")

                if options_select.isdigit() and 0 < int(options_select) < len(prev_orders) + 1:

                    # URL for the API endpoint
                    url = self.base_url + "/v1/accounts/" + account["accountIdKey"] + "/orders/preview.json"

                    # Add parameters and header information
                    headers = {"Content-Type": "application/xml", "consumerKey": self.consumer_key}

                    # Add payload for POST Request
                    payload = """<PreviewOrderRequest>
                                   <orderType>{0}</orderType>
                                   <clientOrderId>{1}</clientOrderId>
                                   <Order>
                                       <allOrNone>false</allOrNone>
                                       <priceType>{2}</priceType>  
                                       <orderTerm>{3}</orderTerm>   
                                       <marketSession>REGULAR</marketSession>
                                       <stopPrice></stopPrice>
                                       <limitPrice>{4}</limitPrice>
                                       <Instrument>
                                           <Product>
                                               <securityType>{5}</securityType>
                                               <symbol>{6}</symbol>
                                           </Product>
                                           <orderAction>{7}</orderAction> 
                                           <quantityType>QUANTITY</quantityType>
                                           <quantity>{8}</quantity>
                                       </Instrument>
                                   </Order>
                               </PreviewOrderRequest>"""

                    options_select = int(options_select)
                    prev_orders[options_select - 1]["client_order_id"] = str(random.randint(1000000000, 9999999999))
                    payload = payload.format(prev_orders[options_select - 1]["order_type"],
                                             prev_orders[options_select - 1]["client_order_id"],
                                             prev_orders[options_select - 1]["priceType"],
                                             prev_orders[options_select - 1]["orderTerm"],
                                             prev_orders[options_select - 1]["limitPrice"],
                                             prev_orders[options_select - 1]["security_type"],
                                             prev_orders[options_select - 1]["symbol"],
                                             prev_orders[options_select - 1]["orderAction"],
                                             prev_orders[options_select - 1]["quantity"])

                    # Make API call for POST request
                    response = session.post(url, header_auth=True, headers=headers, data=payload)
                    logger.debug("Request Header: %s", response.request.headers)
                    logger.debug("Request payload: %s", payload)

                    # Handle and parse response
                    if response is not None and response.status_code == 200:
                        parsed = json.loads(response.text)
                        logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
                        data = response.json()
                        print("\nPreview Order: ")
                        if data is not None and "PreviewOrderResponse" in data and "PreviewIds" in data["PreviewOrderResponse"]:
                            for previewids in data["PreviewOrderResponse"]["PreviewIds"]:
                                print("Preview ID: " + str(previewids["previewId"]))
                        else:
                            # Handle errors
                            data = response.json()
                            if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                                print("Error: " + data["Error"]["message"])
                            else:
                                print("Error: Preview Order API service error")

                        if data is not None and "PreviewOrderResponse" in data and "Order" in data[
                            "PreviewOrderResponse"]:
                            for orders in data["PreviewOrderResponse"]["Order"]:
                                prev_orders[options_select - 1]["limitPrice"] = orders["limitPrice"]

                                if orders is not None and "Instrument" in orders:
                                    for instruments in orders["Instrument"]:
                                        if instruments is not None and "orderAction" in instruments:
                                            print("Action: " + instruments["orderAction"])
                                        if instruments is not None and "quantity" in instruments:
                                            print("Quantity: " + str(instruments["quantity"]))
                                        if instruments is not None and "Product" in instruments \
                                                and "symbol" in instruments["Product"]:
                                            print("Symbol: " + instruments["Product"]["symbol"])
                                        if instruments is not None and "symbolDescription" in instruments:
                                            print("Description: " + str(instruments["symbolDescription"]))

                            if orders is not None and "priceType" in orders and "limitPrice" in orders:
                                print("Price Type: " + orders["priceType"])
                                if orders["priceType"] == "MARKET":
                                    print("Price: MKT")
                                else:
                                    print("Price: " + str(orders["limitPrice"]))
                            if orders is not None and "orderTerm" in orders:
                                print("Duration: " + orders["orderTerm"])
                            if orders is not None and "estimatedCommission" in orders:
                                print("Estimated Commission: " + str(orders["estimatedCommission"]))
                            if orders is not None and "estimatedTotalAmount" in orders:
                                print("Estimated Total Cost: " + str(orders["estimatedTotalAmount"]))
                        else:
                            # Handle errors
                            data = response.json()
                            if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                                print("Error: " + data["Error"]["message"])
                            else:
                                print("Error: Preview Order API service error")
                    else:
                        # Handle errors
                        data = response.json()
                        if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                            print("Error: " + data["Error"]["message"])
                        else:
                            print("Error: Preview Order API service error")
                    break
                elif options_select.isdigit() and int(options_select) == len(prev_orders) + 1:
                    break
                else:
                    print("Unknown Option Selected!")

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

    @staticmethod
    def options_selection(options):
        """
        Formats and displays different options in a menu

        :param options: List of options to display
        :return the number user selected
        """
        while True:
            print("")
            for num, priceType in enumerate(options, start=1):
                print("{})\t{}".format(num, priceType))
            options_select = input("Please select an option: ")
            if options_select.isdigit() and 0 < int(options_select) < len(options) + 1:
                return options_select
            else:
                print("Unknown Option Selected!")

    def user_select_order(self):
        """
            Provides users options to select to preview orders
            :param self test
            :return user's order selections
            """
        order = {"priceType": "",
                 "orderTerm": "",
                 "symbol": "",
                 "orderAction": "",
                 "limitPrice":"",
                 "quantity": ""}

        priceType_options = ["MARKET", "LIMIT"]
        orderTerm_options = ["GOOD_FOR_DAY", "IMMEDIATE_OR_CANCEL", "FILL_OR_KILL"]
        orderAction_options = ["BUY", "SELL", "BUY_TO_COVER", "SELL_SHORT"]

        print("\nPrice Type:")
        order["priceType"] = priceType_options[int(self.options_selection(priceType_options)) - 1]

        if order["priceType"] == "MARKET":
            order["orderTerm"] = "GOOD_FOR_DAY"
        else:
            print("\nOrder Term:")
            order["orderTerm"] = orderTerm_options[int(self.options_selection(orderTerm_options)) - 1]

        order["limitPrice"] = None
        if order["priceType"] == "LIMIT":
            while order["limitPrice"] is None or not order["limitPrice"].isdigit() \
                    and not re.match(r'\d+(?:[.]\d{2})?$', order["limitPrice"]):
                order["limitPrice"] = input("\nPlease input limit price: ")

        order["client_order_id"] = random.randint(1000000000, 9999999999)

        while order["symbol"] == "":
            order["symbol"] = input("\nPlease enter a stock symbol :")

        print("\nOrder Action Type:")
        order["orderAction"] = orderAction_options[int(self.options_selection(orderAction_options)) - 1]

        while not order["quantity"].isdigit():
            order["quantity"] = input("\nPlease type quantity:")

        return order

    def preview_order_menu(self, session, account, prev_orders):
        """
        Provides the different options for preview orders: select new order or select from previous order

        :param session: authenticated session
        :param account: information on selected account
        :param prev_orders: list of instruments from previous orders
        """
        menu_list = {"1": "Select New Order",
                     "2": "Select From Previous Orders",
                     "3": "Go Back"}

        while True:
            print("")
            options = menu_list.keys()
            for entry in options:
                print(entry + ")\t" + menu_list[entry])

            selection = input("Please select an option: ")
            if selection == "1":
                print("\nPreview Order: ")
                self.preview_order()
                break
            elif selection == "2":
                self.previous_order(session, account, prev_orders)
                break
            elif selection == "3":
                break
            else:
                print("Unknown Option Selected!")

    def cancel_order(self):
        """
        Calls cancel order API to cancel an existing order
        :param self: Pass parameter with authenticated session and information on selected account
        """
        while True:
            # Display a list of Open Orders
            # URL for the API endpoint
            url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders.json"

            # Add parameters and header information
            params_open = {"status": "OPEN"}
            headers = {"consumerKey": self.consumer_key}

            # Make API call for GET request
            response_open = self.session.get(url, header_auth=True, params=params_open, headers=headers)

            logger.debug("Request Header: %s", response_open.request.headers)
            logger.debug("Response Body: %s", response_open.text)

            print("\nOpen Orders: ")
            # Handle and parse response
            if response_open.status_code == 204:
                logger.debug(response_open)
                print("None")
                menu_items = {"1": "Go Back"}
                while True:
                    print("")
                    options = menu_items.keys()
                    for entry in options:
                        print(entry + ")\t" + menu_items[entry])

                    selection = input("Please select an option: ")
                    if selection == "1":
                        break
                    else:
                        print("Unknown Option Selected!")
                break
            elif response_open.status_code == 200:
                parsed = json.loads(response_open.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_open.json()

                order_list = []
                count = 1
                if data is not None and "OrdersResponse" in data and "Order" in data["OrdersResponse"]:
                    for order in data["OrdersResponse"]["Order"]:
                        if order is not None and "OrderDetail" in order:
                            for details in order["OrderDetail"]:
                                if details is not None and "Instrument" in details:
                                    for instrument in details["Instrument"]:
                                        order_str = ""
                                        order_obj = {"priceType": None,
                                                     "orderTerm": None,
                                                     "executedTime": None,
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
                                        if instrument is not None and 'Product' in instrument and 'securityType' \
                                                in instrument["Product"]:
                                            order_str += "Type: " + instrument["Product"]["securityType"] + " | "
                                            order_obj["security_type"] = instrument["Product"]["securityType"]

                                        if instrument is not None and 'orderAction' in instrument:
                                            order_str += "Order Type: " + instrument["orderAction"] + " | "
                                            order_obj["orderAction"] = instrument["orderAction"]

                                        if instrument is not None and 'orderedQuantity' in instrument:
                                            order_str += "Quantity(Exec/Entered): " + str(
                                                "{:,}".format(instrument["orderedQuantity"])) + " | "
                                            order_obj["quantity"] = instrument["orderedQuantity"]

                                        if instrument is not None and 'Product' in instrument and 'symbol' \
                                                in instrument["Product"]:
                                            order_str += "Symbol: " + instrument["Product"]["symbol"] + " | "
                                            order_obj["symbol"] = instrument["Product"]["symbol"]

                                        if details is not None and 'priceType' in details:
                                            order_str += "Price Type: " + details["priceType"] + " | "
                                            order_obj["priceType"] = details["priceType"]

                                        if details is not None and 'orderTerm' in details:
                                            order_str += "Term: " + details["orderTerm"] + " | "
                                            order_obj["orderTerm"] = details["orderTerm"]

                                        if details is not None and 'limitPrice' in details:
                                            order_str += "Price: " + str(
                                                '${:,.2f}'.format(details["limitPrice"])) + " | "
                                            order_obj["limitPrice"] = details["limitPrice"]

                                        if instrument is not None and 'filledQuantity' in instrument:
                                            order_str += "Quantity Executed: " \
                                                         + str("{:,}".format(instrument["filledQuantity"])) + " | "
                                            order_obj["quantity"] = instrument["filledQuantity"]

                                        if instrument is not None and "averageExecutionPrice" in instrument:
                                            order_str += "Price Executed: " + str(
                                                '${:,.2f}'.format(instrument["averageExecutionPrice"])) + " | "

                                        if details is not None and 'status' in details:
                                            order_str += "Status: " + details["status"]

                                        print(str(count) + ")\t" + order_str)
                                        count = 1 + count
                                        order_list.append(order["orderId"])

                    print(str(count) + ")\tGo Back")
                    selection = input("Please select an option: ")
                    if selection.isdigit() and 0 < int(selection) < len(order_list) + 1:
                        # URL for the API endpoint
                        url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders/cancel.json"

                        # Add parameters and header information
                        headers = {"Content-Type": "application/xml", "consumerKey": self.consumer_key}

                        # Add payload for POST Request
                        payload = """<CancelOrderRequest>
                                        <orderId>{0}</orderId>
                                    </CancelOrderRequest>
                                   """
                        payload = payload.format(order_list[int(selection) - 1])

                        # Add payload for PUT Request
                        response = self.session.put(url, header_auth=True, headers=headers, data=payload)
                        logger.debug("Request Header: %s", response.request.headers)
                        logger.debug("Request payload: %s", payload)

                        # Handle and parse response
                        if response is not None and response.status_code == 200:
                            parsed = json.loads(response.text)
                            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
                            data = response.json()
                            if data is not None and "CancelOrderResponse" in data \
                                    and "orderId" in data["CancelOrderResponse"]:
                                print("\nOrder number #" + str(
                                    data["CancelOrderResponse"]["orderId"]) + " successfully Cancelled.")
                            else:
                                # Handle errors
                                logger.debug("Response Headers: %s", response.headers)
                                logger.debug("Response Body: %s", response.text)
                                data = response.json()
                                if 'Error' in data and 'message' in data["Error"] \
                                        and data["Error"]["message"] is not None:
                                    print("Error: " + data["Error"]["message"])
                                else:
                                    print("Error: Cancel Order API service error")
                        else:
                            # Handle errors
                            logger.debug("Response Headers: %s", response.headers)
                            logger.debug("Response Body: %s", response.text)
                            data = response.json()
                            if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                                print("Error: " + data["Error"]["message"])
                            else:
                                print("Error: Cancel Order API service error")
                        break

                    elif selection.isdigit() and int(selection) == len(order_list) + 1:
                        break
                    else:
                        print("Unknown Option Selected!")
                else:
                    # Handle errors
                    logger.debug("Response Body: %s", response_open.text)
                    if response_open is not None and response_open.headers['Content-Type'] == 'application/json' \
                            and "Error" in response_open.json() and "message" in response_open.json()["Error"] \
                            and response_open.json()["Error"]["message"] is not None:
                        print("Error: " + response_open.json()["Error"]["message"])
                    else:
                        print("Error: Balance API service error")
                    break
            else:
                # Handle errors
                logger.debug("Response Body: %s", response_open.text)
                if response_open is not None and response_open.headers['Content-Type'] == 'application/json' \
                        and "Error" in response_open.json() and "message" in response_open.json()["Error"] \
                        and response_open.json()["Error"]["message"] is not None:
                    print("Error: " + response_open.json()["Error"]["message"])
                else:
                    print("Error: Balance API service error")
                break

    def view_orders(self):
        """
        Calls orders API to provide the details for the orders

        :param self: Pass in authenticated session and information on selected account
        """
        while True:
            # URL for the API endpoint
            url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders.json"

            # Add parameters and header information
            headers = {"consumerKey": self.consumer_key}
            params_open = {"status": "OPEN"}
            params_executed = {"status": "EXECUTED","symbol":"ENPH"}
            # params_executed = {"status": "INDIVIDUAL_FILLS"}
            params_indiv_fills = {"status": "INDIVIDUAL_FILLS"}
            params_cancelled = {"status": "CANCELLED"}
            params_rejected = {"status": "REJECTED"}
            params_expired = {"status": "EXPIRED"}

            # Make API call for GET request
            response_open = self.session.get(url, header_auth=True, params=params_open, headers=headers)
            response_executed = self.session.get(url, header_auth=True, params=params_executed, headers=headers)
            response_indiv_fills = self.session.get(url, header_auth=True, params=params_indiv_fills, headers=headers)
            response_cancelled = self.session.get(url, header_auth=True, params=params_cancelled, headers=headers)
            response_rejected = self.session.get(url, header_auth=True, params=params_rejected, headers=headers)
            response_expired = self.session.get(url, header_auth=True, params=params_expired, headers=headers)

            prev_orders = []

            # Open orders
            logger.debug("Request Header: %s", response_open.request.headers)
            logger.debug("Response Body: %s", response_open.text)

            print("\nOpen Orders:")
            # Handle and parse response
            if response_open.status_code == 204:
                logger.debug(response_open)
                print("None")
            elif response_open.status_code == 200:
                parsed = json.loads(response_open.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_open.json()

                # Display list of open orders
                prev_orders.extend(self.print_orders(data, "open"))

            # Executed orders
            logger.debug("Request Header: %s", response_executed.request.headers)
            logger.debug("Response Body: %s", response_executed.text)
            logger.debug(response_executed.text)

            print("\nExecuted Orders:")
            # Handle and parse response
            if response_executed.status_code == 204:
                logger.debug(response_executed)
                print("None")
            elif response_executed.status_code == 200:
                parsed = json.loads(response_executed.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_executed.json()

                print(data)

                # Display list of executed orders
                prev_orders.extend(self.print_orders(data, "executed"))

            # Individual fills orders
            logger.debug("Request Header: %s", response_indiv_fills.request.headers)
            logger.debug("Response Body: %s", response_indiv_fills.text)

            print("\nIndividual Fills Orders:")
            # Handle and parse response
            if response_indiv_fills.status_code == 204:
                logger.debug("Response Body: %s", response_indiv_fills)
                print("None")
            elif response_indiv_fills.status_code == 200:
                parsed = json.loads(response_indiv_fills.text)
                logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
                data = response_indiv_fills.json()

                # Display list of individual fills orders
                prev_orders.extend(self.print_orders(data, "indiv_fills"))

            # Cancelled orders
            logger.debug("Request Header: %s", response_cancelled.request.headers)
            logger.debug("Response Body: %s", response_cancelled.text)

            print("\nCancelled Orders:")
            # Handle and parse response
            if response_cancelled.status_code == 204:
                logger.debug(response_cancelled)
                print("None")
            elif response_cancelled.status_code == 200:
                parsed = json.loads(response_cancelled.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_cancelled.json()

                # Display list of open orders
                prev_orders.extend(self.print_orders(data, "cancelled"))

            # Rejected orders
            logger.debug("Request Header: %s", response_rejected.request.headers)
            logger.debug("Response Body: %s", response_rejected.text)

            print("\nRejected Orders:")
            # Handle and parse response
            if response_rejected.status_code == 204:
                logger.debug(response_rejected)
                print("None")
            elif response_rejected.status_code == 200:
                parsed = json.loads(response_rejected.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_rejected.json()

                # Display list of open orders
                prev_orders.extend(self.print_orders(data, "rejected"))

            # Expired orders
            print("\nExpired Orders:")
            # Handle and parse response
            if response_expired.status_code == 204:
                logger.debug(response_executed)
                print("None")
            elif response_expired.status_code == 200:
                parsed = json.loads(response_expired.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_expired.json()

                # Display list of open orders
                prev_orders.extend(self.print_orders(data, "expired"))

            menu_list = {"1": "Preview Order",
                         "2": "Cancel Order",
                         "3": "Filter Order",
                         "4": "Go Back"}

            print("")
            options = menu_list.keys()
            for entry in options:
                print(entry + ")\t" + menu_list[entry])

            selection = input("Please select an option: ")
            if selection == "1":
                self.preview_order_menu(self.session, self.account, prev_orders)
            elif selection == "2":
                self.cancel_order()
            elif selection == "3":
                self.filter_order()
            elif selection == "4":
                break
            else:
                print("Unknown Option Selected!")

    def filter_order(self):
        """
        Calls orders API to provide the details for the orders

        :param self: Pass in authenticated session and information on selected account
        """

        n=int(input("Enter the number of filter conditions: "))
        filter={}

        for i in range(n):
            keys = input()
            values = input()
            filter[keys] = values
        print(filter)

        while True:
            # URL for the API endpoint
            url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders.json"

            # Add parameters and header information
            headers = {"consumerKey": self.consumer_key}

            # Make API call for GET request
            response_filter = self.session.get(url, header_auth=True, params=filter, headers=headers)

            prev_orders = []

            # Executed orders
            logger.debug("Request Header: %s", response_filter.request.headers)
            logger.debug("Response Body: %s", response_filter.text)
            logger.debug(response_filter.text)

            print("\nExecuted Orders:")
            # Handle and parse response
            if response_filter.status_code == 204:
                logger.debug(response_filter)
                print("None")
            elif response_filter.status_code == 200:
                parsed = json.loads(response_filter.text)
                logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
                data = response_filter.json()
                
                print(data)

                # Display list of executed orders
                prev_orders.extend(self.print_orders_customized(data,"executed"))

            menu_list = {"1": "Preview Order",
                         "2": "Cancel Order",
                         "3": "Filter Order",
                         "4": "Go Back"}

            print("")
            options = menu_list.keys()
            for entry in options:
                print(entry + ")\t" + menu_list[entry])

            selection = input("Please select an option: ")
            if selection == "1":
                self.preview_order_menu(self.session, self.account, prev_orders)
            elif selection == "2":
                self.cancel_order()
            elif selection == "3":
                self.filter_order()
            elif selection == "4":
                break
            else:
                print("Unknown Option Selected!")

    def get_open_orders(self) -> list:
        """
        Get all open orders for the current account.
        :return: List of open orders, each represented as a dictionary
        """
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        headers = {"consumerKey": self.consumer_key}
        params = {"status": "OPEN"}

        response = self.session.get(url, header_auth=True, params=params, headers=headers)
        logger.debug("Request Header: %s", response.request.headers)
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
        logger.debug("Request Header: %s", response.request.headers)
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
        logger.debug("Request Header: %s", response.request.headers)
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
            
            # Convert expiration date to the year, month, day required by the API
            try:
                expiry_date_obj = datetime.strptime(expiration_date, "%Y-%m-%d")
            except ValueError:
                print("Invalid expiration date format. Use 'YYYY-MM-DD'.")
                return None

            # Define the API endpoint for placing an option order
            # url = f"{self.base_url}/v1/accounts/{self.account_id}/orders/placeOptionOrder.json"
            url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/orders.json"

            # Construct the order payload
            order_payload = {
                "orderType": "OPTION",
                "clientOrderId": f"{ticker_symbol}_OptionOrder_{datetime.now().strftime('%Y%m%d%H%M%S')}",  # Unique client order ID
                "orderStrategyType": "SINGLE",
                "orderAction": order_action,
                "priceType": price_type,
                "limitPrice": limit_price if price_type == "LIMIT" else None,
                "orderTerm": order_term,
                "Instrument": [
                    {
                        "symbol": ticker_symbol,
                        "orderAction": order_action,
                        "quantity": quantity,
                        "expiryYear": expiry_date_obj.year,
                        "expiryMonth": expiry_date_obj.month,
                        "expiryDay": expiry_date_obj.day,
                        "callPut": call_put.upper(),
                        "strikePrice": strike_price,
                    }
                ]
            }

            # Make the API request to place the order
            response = self.session.post(url, json=order_payload)

            # Check if the request was successful
            if response.status_code != 200:
                print("Failed to place option order:", response.status_code, response.text)
                return None

            # Return the response JSON
            order_response = response.json()
            print("Option order placed successfully:", order_response)
            return order_response

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
        # URL for the API endpoint to get all open orders
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
        
        # Add parameters and header information
        params_open = {"status": "OPEN"}
        headers = {"consumerKey": self.consumer_key}
        
        # Make API call for GET request
        response_open = self.session.get(url, header_auth=True, params=params_open, headers=headers)
        logger.debug("Request Header: %s", response_open.request.headers)
        logger.debug("Response Body: %s", response_open.text)
        
        print("\nOpen Orders:")
        
        # Handle and parse response
        if response_open.status_code == 204:
            # No content - no open orders
            logger.debug(response_open)
            print("No open orders found.")
            return
        
        elif response_open.status_code == 200:
            parsed = json.loads(response_open.text)
            logger.debug(json.dumps(parsed, indent=4, sort_keys=True))
            data = response_open.json()
            
            order_list = []
            if data is not None and "OrdersResponse" in data and "Order" in data["OrdersResponse"]:
                # Display all open orders
                for order in data["OrdersResponse"]["Order"]:
                    if order is not None and "OrderDetail" in order:
                        for details in order["OrderDetail"]:
                            if details is not None and "Instrument" in details:
                                for instrument in details["Instrument"]:
                                    order_str = ""
                                    if order is not None and 'orderId' in order:
                                        order_str += "Order #" + str(order["orderId"]) + " : "
                                        order_list.append(order["orderId"])
                                    
                                    if instrument is not None and 'Product' in instrument and 'securityType' in instrument["Product"]:
                                        order_str += "Type: " + instrument["Product"]["securityType"] + " | "
                                    
                                    if instrument is not None and 'orderAction' in instrument:
                                        order_str += "Order Type: " + instrument["orderAction"] + " | "
                                    
                                    if instrument is not None and 'orderedQuantity' in instrument:
                                        order_str += "Quantity: " + str("{:,}".format(instrument["orderedQuantity"])) + " | "
                                    
                                    if instrument is not None and 'Product' in instrument and 'symbol' in instrument["Product"]:
                                        order_str += "Symbol: " + instrument["Product"]["symbol"] + " | "
                                    
                                    if details is not None and 'priceType' in details:
                                        order_str += "Price Type: " + details["priceType"] + " | "
                                    
                                    if details is not None and 'limitPrice' in details:
                                        order_str += "Price: " + str('${:,.2f}'.format(details["limitPrice"])) + " | "
                                    
                                    if details is not None and 'status' in details:
                                        order_str += "Status: " + details["status"]
                                    
                                    print(order_str)
                
                # Ask for confirmation
                if order_list:
                    print(f"\nFound {len(order_list)} open order(s).")
                    confirmation = input("Do you want to cancel ALL open orders? (yes/no): ").strip().lower()
                    
                    if confirmation == "yes":
                        # URL for the API endpoint to cancel orders
                        cancel_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/cancel.json"
                        
                        # Add parameters and header information for cancel request
                        cancel_headers = {"Content-Type": "application/xml", "consumerKey": self.consumer_key}
                        
                        # Counter for successful cancellations
                        cancelled_count = 0
                        
                        # Loop through all orders and cancel each one
                        for order_id in order_list:
                            # Add payload for the cancel request
                            payload = """<CancelOrderRequest>
                                            <orderId>{0}</orderId>
                                        </CancelOrderRequest>
                                    """
                            payload = payload.format(order_id)
                            
                            # Make API call for PUT request
                            response = self.session.put(cancel_url, header_auth=True, headers=cancel_headers, data=payload)
                            logger.debug("Request Header: %s", response.request.headers)
                            logger.debug("Request payload: %s", payload)
                            
                            # Handle and parse response
                            if response is not None and response.status_code == 200:
                                data = response.json()
                                if data is not None and "CancelOrderResponse" in data and "orderId" in data["CancelOrderResponse"]:
                                    print(f"Order #{data['CancelOrderResponse']['orderId']} successfully cancelled.")
                                    cancelled_count += 1
                                else:
                                    # Handle errors
                                    if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                                        print(f"Error cancelling order #{order_id}: {data['Error']['message']}")
                                    else:
                                        print(f"Error cancelling order #{order_id}: Cancel Order API service error")
                            else:
                                # Handle errors
                                if response:
                                    try:
                                        data = response.json()
                                        if 'Error' in data and 'message' in data["Error"] and data["Error"]["message"] is not None:
                                            print(f"Error cancelling order #{order_id}: {data['Error']['message']}")
                                        else:
                                            print(f"Error cancelling order #{order_id}: Cancel Order API service error")
                                    except ValueError:
                                        print(f"Error cancelling order #{order_id}: Invalid response format")
                                else:
                                    print(f"Error cancelling order #{order_id}: No response from API")
                        
                        print(f"\nCancellation summary: {cancelled_count} out of {len(order_list)} orders cancelled.")
                    else:
                        print("Operation cancelled. No orders were cancelled.")
                else:
                    print("No open orders found to cancel.")
            else:
                print("No open orders found.")
        else:
            # Handle errors
            logger.debug("Response Body: %s", response_open.text)
            if response_open is not None and response_open.headers['Content-Type'] == 'application/json':
                try:
                    error_data = response_open.json()
                    if 'Error' in error_data and 'message' in error_data["Error"] and error_data["Error"]["message"] is not None:
                        print(f"Error: {error_data['Error']['message']}")
                    else:
                        print("Error: Orders API service error")
                except ValueError:
                    print(f"Error: Invalid response format (Status code: {response_open.status_code})")
            else:
                print(f"Error: Orders API service error (Status code: {response_open.status_code})")

    def refresh_order_limit_old(self):
        """
        Refreshes the limit price of all open spread orders by adjusting NET_CREDIT orders down by 0.01
        and NET_DEBIT orders up by 0.01 every 10 seconds until no open orders remain.
        Uses the E*TRADE API v1 change order endpoint.
        """
        while True:
            # Query all open orders
            url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
            params = {"status": "OPEN"}
            headers = {"consumerKey": self.consumer_key}
            
            response = self.session.get(url, header_auth=True, params=params, headers=headers)

            if response.status_code == 204:
                # No content - no open orders
                print("No open orders found.")
                break

            if response.status_code != 200:
                print("Failed to fetch open orders. Status Code: %s, Response: %s", 
                            response.status_code, response.text)
                return
            
            data = response.json()
            open_orders = data.get("OrdersResponse", {}).get("Order", [])
            
            # Process each open order
            for order in open_orders:
                order_id = order['orderId']
                for detail in order.get("OrderDetail", []):
                    print(f"details: {detail}")
                    price_type = detail.get("priceType")
                    
                    # Only process NET_CREDIT or NET_DEBIT spread orders
                    if price_type not in ["NET_CREDIT", "NET_DEBIT"]:
                        continue
                    
                    limit_price = detail.get("limitPrice")
                    if limit_price is None:
                        logger.warning("No limit price found for order %s", order_id)
                        continue
                    
                    instruments = detail.get("Instrument", [])
                    if len(instruments) <= 1:
                        continue  # Skip if not a spread order (single leg)
                    
                    # Calculate new limit price
                    if price_type == "NET_CREDIT":
                        new_limit_price = limit_price - 0.01
                    elif price_type == "NET_DEBIT":
                        new_limit_price = limit_price + 0.01
                    
                    # Generate a new unique clientOrderId
                    new_client_order_id = str(random.randint(1000000000, 9999999999))
                    
                    # Construct instruments XML for all legs
                    instruments_xml = ""
                    for instrument in instruments:
                        product = instrument['Product']
                        leg_xml = f"""
                        <Instrument>
                            <Product>
                                <securityType>{product['securityType']}</securityType>
                                <symbol>{product['symbol']}</symbol>
                                <callPut>{product.get('callPut', '')}</callPut>
                                <expiryYear>{product.get('expiryYear', '')}</expiryYear>
                                <expiryMonth>{product.get('expiryMonth', '')}</expiryMonth>
                                <expiryDay>{product.get('expiryDay', '')}</expiryDay>
                                <strikePrice>{product.get('strikePrice', '')}</strikePrice>
                            </Product>
                            <orderAction>{instrument['orderAction']}</orderAction>
                            <quantityType>QUANTITY</quantityType>
                            <quantity>{instrument['orderedQuantity']}</quantity>
                        </Instrument>
                        """
                        instruments_xml += leg_xml.strip()
                    
                    # Get order term, default to GOOD_FOR_DAY if not present
                    order_term = detail.get("orderTerm", "GOOD_FOR_DAY")
                    
                    # Construct ChangeOrderRequest payload
                    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
                    <ChangeOrderRequest>
                        <orderType>OPTN</orderType>
                        <clientOrderId>{new_client_order_id}</clientOrderId>
                        <Order>
                            <allOrNone>false</allOrNone>
                            <priceType>{price_type}</priceType>
                            <limitPrice>{new_limit_price:.2f}</limitPrice>
                            <orderTerm>{order_term}</orderTerm>
                            <marketSession>REGULAR</marketSession>
                            {instruments_xml}
                        </Order>
                    </ChangeOrderRequest>
                    """
                    
                    # Send PUT request to change order
                    change_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/{order_id}/change/preview"
                    change_headers = {
                        "Content-Type": "application/xml",
                        "consumerKey": self.consumer_key,
                    }
                    
                    change_response = self.session.put(change_url, headers=change_headers, data=payload)
                    
                    if change_response.status_code == 200:
                        print("Successfully updated order %s to new limit price %.2f", 
                                order_id, new_limit_price)
                    else:
                        print("Failed to update order %s: Status Code: %s, Response: %s", 
                                    order_id, change_response.status_code, change_response.text)
                        print(change_url)
                        print(payload)
            # Wait 10 seconds before next iteration
            time.sleep(10)



    def refresh_order_limit(self):
        """
        Refreshes the limit prices of open spread orders by adjusting them slightly.
        Incorporates the preview step before placing the order change as per E*TRADE API v1.
        """
        while True:
            # Query all open orders
            url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders.json"
            params = {"status": "OPEN"}
            headers = {"consumerKey": self.consumer_key}
            
            response = self.session.get(url, header_auth=True, params=params, headers=headers)

            if response.status_code != 200:
                print("Failed to fetch open orders: %s", response.text)
                time.sleep(10)
                continue

            orders = response.json().get("OrdersResponse", {}).get("Order", [])
            if not orders:
                print("No open orders found. Exiting refresh loop.")
                break

            for order in orders:
                order_id = order['orderId']
                for detail in order.get("OrderDetail", []):
                    order_id = order["orderId"]
                    price_type = detail.get("priceType")
                    instruments = detail.get("Instrument", [])
                    order_type = order.get("orderType", "SPREADS")
                    print(order)
                    print(f"prcie type: {price_type}")
                    if price_type not in ["NET_CREDIT", "NET_DEBIT"]:
                        print("Skipping non-spread order %s", order_id)
                        continue  # Skip non-spread orders

                    # Extract current limit price and calculate new limit price
                    limit_price = float(detail["limitPrice"])
                    new_limit_price = limit_price - 0.01 if price_type == "NET_CREDIT" else limit_price + 0.01

                    # Generate a unique clientOrderId
                    new_client_order_id = str(random.randint(1000000000, 9999999999))
                    instruments_xml = ""
                    for instrument in instruments:
                        product = instrument['Product']
                        leg_xml = f"""
                        <Instrument>
                            <Product>
                                <securityType>{product['securityType']}</securityType>
                                <symbol>{product['symbol']}</symbol>
                                <callPut>{product.get('callPut', '')}</callPut>
                                <expiryYear>{product.get('expiryYear', '')}</expiryYear>
                                <expiryMonth>{product.get('expiryMonth', '')}</expiryMonth>
                                <expiryDay>{product.get('expiryDay', '')}</expiryDay>
                                <strikePrice>{product.get('strikePrice', '')}</strikePrice>
                            </Product>
                            <orderAction>{instrument['orderAction']}</orderAction>
                            <quantityType>QUANTITY</quantityType>
                            <quantity>{instrument['orderedQuantity']}</quantity>
                        </Instrument>"""
                        instruments_xml += leg_xml

                    order_term = detail.get("orderTerm", "GOOD_FOR_DAY")
                    # Step 1: Preview the order change

                    preview_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
                    <PreviewOrderRequest>
                        <orderType>{order_type}</orderType>
                        <clientOrderId>{new_client_order_id}</clientOrderId>
                        <Order>
                            <allOrNone>false</allOrNone>
                            <priceType>{price_type}</priceType>
                            <limitPrice>{new_limit_price:.2f}</limitPrice>
                            <orderTerm>{order_term}</orderTerm>
                            <marketSession>REGULAR</marketSession>{instruments_xml}
                        </Order>
                    </PreviewOrderRequest>"""

                    # preview_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/{order_id}/change/preview"
                    preview_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/preview"
                    preview_headers = {
                        "Content-Type": "application/xml",
                        "consumerKey": self.consumer_key,
                    }

                    preview_response = self.session.post(preview_url, headers=preview_headers, data=preview_payload)
                    if preview_response.status_code != 200:
                        print("Failed to preview change for order %s: %s", order_id, preview_response.text)
                        print(preview_payload)
                        print(preview_url)
                        return
                        continue

                    # Parse preview response to extract previewId
                    try:
                        preview_root = ET.fromstring(preview_response.text)
                        preview_id_elem = preview_root.find(".//previewId")
                        if preview_id_elem is None:
                            print("No previewId found in preview response for order %s", order_id)
                            print(preview_response.text)
                            continue
                        preview_id = preview_id_elem.text
                    except ET.ParseError:
                        print("Failed to parse preview response for order %s: %s", order_id, preview_response.text)
                        continue

                    time.sleep(5)
                    # Step 2: Place the order change with previewId
                    place_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
                    <PlaceOrderRequest>
                        <orderType>{order_type}</orderType>
                        <clientOrderId>{new_client_order_id}</clientOrderId>
                        <PreviewIds>
                            <previewId>{preview_id}</previewId>
                        </PreviewIds>
                        <Order>
                            <allOrNone>false</allOrNone>
                            <priceType>{price_type}</priceType>
                            <limitPrice>{new_limit_price:.2f}</limitPrice>
                            <orderTerm>{order_term}</orderTerm>
                            <marketSession>REGULAR</marketSession>
                            {instruments_xml}
                        </Order>
                    </PlaceOrderRequest>
                    """

                    place_url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/orders/{order_id}/change/place"
                    place_headers = {
                        "Content-Type": "application/xml",
                        "consumerKey": self.consumer_key,
                    }

                    place_response = self.session.put(place_url, headers=place_headers, data=place_payload)
                    if place_response.status_code == 200:
                        print("Successfully updated order %s to new limit price %.2f", order_id, new_limit_price)
                    else:
                        print("Failed to update order %s: %s", order_id, place_response.text)

            # Wait 10 seconds before the next iteration
            time.sleep(10)
