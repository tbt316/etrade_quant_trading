import json
import logging
from logging.handlers import RotatingFileHandler
import xmltodict
import xml.etree.ElementTree as ET
import configparser

# logger settings
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("python_client.log", maxBytes=5 * 1024 * 1024, backupCount=3)
FORMAT = "%(asctime)-15s %(message)s"
fmt = logging.Formatter(FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)

# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

class Market:
    def __init__(self, session, base_url,use_sandbox=False):
        self.session = session
        self.base_url = base_url
        self.use_sandbox = use_sandbox
        if self.use_sandbox:
            self.consumer_key = config["DEFAULT"]["SANDBOX_CONSUMER_KEY"]
        else: 
            self.consumer_key = config["DEFAULT"]["PROD_CONSUMER_KEY"]

    def lookup(self, symbol):
        """
        Calls the market lookup API to search for securities matching a symbol or name.

        Parameters:
        - symbol (str): The partial or complete ticker symbol or security name to search for.

        Returns:
        - list: A list of dictionaries containing symbol, description, and type for each match.
        """
        # URL for the API endpoint
        url = f"{self.base_url}/v1/market/lookup/{symbol}"
        
        # Parameters for the API call
        # params = {"search": symbol}
        response = self.session.get(url, auth=self.session.auth)
        logger.debug("Request Header: %s", response.request.headers)

        # Make the API call
        # response = self.session.get(url, params=params)

        if response.status_code == 200:
            try:
                # Parse the XML response
                root = ET.fromstring(response.content)
                results = []
                for data in root.findall("Data"):
                    symbol = data.find("symbol").text
                    description = data.find("description").text
                    security_type = data.find("type").text
                    results.append({
                        "symbol": symbol,
                        "description": description,
                        "type": security_type,
                    })

                # Print and return the results
                if results:
                    print(f"Found {len(results)} result(s) for '{symbol}':")
                    for result in results:
                        print(f"Symbol: {result['symbol']}, Description: {result['description']}, Type: {result['type']}")
                else:
                    print(f"No matches found for '{symbol}'.")
                return results
            except ET.ParseError as e:
                print("Error parsing XML response:", e)
                return []
        else:
            # Handle API errors
            print(f"Error: Lookup API returned status code {response.status_code}.")
            try:
                error_message = response.text
                print("Error response:", error_message)
            except Exception:
                print("Failed to parse error response.")
            return []

    def quotes(self,symbols):
        """
        Calls quotes API to provide quote details for equities, options, and mutual funds

        :param self: Passes authenticated session in parameter
        """
        # symbols = input("\nPlease enter Stock Symbol: ")

        # URL for the API endpoint
        # url = self.base_url + "/v1/market/quote/" + symbols + ".json"

        #update by Bo
        url = "%s%s%s" % (self.base_url, "/v1/market/quote/", ",".join(symbols[:25]))
        url += ".json"

        # Make API call for GET request
        response = self.session.get(url)

        logger.debug("Request Header: %s", response.request.headers)

        if response is not None and response.status_code == 200:

            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))

            # Handle and parse response
            print("")
            data = response.json()
            if data is not None and "QuoteResponse" in data and "QuoteData" in data["QuoteResponse"]:
                for quote in data["QuoteResponse"]["QuoteData"]:
                    if quote is not None and "dateTime" in quote:
                        print("Date Time: " + quote["dateTime"])
                    if quote is not None and "Product" in quote and "symbol" in quote["Product"]:
                        print("Symbol: " + quote["Product"]["symbol"])
                    if quote is not None and "Product" in quote and "securityType" in quote["Product"]:
                        print("Security Type: " + quote["Product"]["securityType"])
                    if quote is not None and "All" in quote and "lastTrade" in quote["All"]:
                        print("Last Price: " + str(quote["All"]["lastTrade"]))
                    if quote is not None and "All" in quote and "changeClose" in quote["All"] \
                        and "changeClosePercentage" in quote["All"]:
                        print("Today's Change: " + str('{:,.3f}'.format(quote["All"]["changeClose"])) + " (" +
                              str(quote["All"]["changeClosePercentage"]) + "%)")
                    if quote is not None and "All" in quote and "lastTrade" in quote["All"]:
                        print("Open: " + str('{:,.2f}'.format(quote["All"]["lastTrade"])))
                    if quote is not None and "All" in quote and "previousClose" in quote["All"]:
                        print("Previous Close: " + str('{:,.2f}'.format(quote["All"]["previousClose"])))
                    if quote is not None and "All" in quote and "bid" in quote["All"] and "bidSize" in quote["All"]:
                        print("Bid (Size): " + str('{:,.2f}'.format(quote["All"]["bid"])) + "x" + str(
                            quote["All"]["bidSize"]))
                    if quote is not None and "All" in quote and "ask" in quote["All"] and "askSize" in quote["All"]:
                        print("Ask (Size): " + str('{:,.2f}'.format(quote["All"]["ask"])) + "x" + str(
                            quote["All"]["askSize"]))
                    if quote is not None and "All" in quote and "low" in quote["All"] and "high" in quote["All"]:
                        print("Day's Range: " + str(quote["All"]["low"]) + "-" + str(quote["All"]["high"]))
                    if quote is not None and "All" in quote and "totalVolume" in quote["All"]:
                        print("Volume: " + str('{:,}'.format(quote["All"]["totalVolume"])))
            else:
                # Handle errors
                if data is not None and 'QuoteResponse' in data and 'Messages' in data["QuoteResponse"] \
                        and 'Message' in data["QuoteResponse"]["Messages"] \
                        and data["QuoteResponse"]["Messages"]["Message"] is not None:
                    for error_message in data["QuoteResponse"]["Messages"]["Message"]:
                        print("Error: " + error_message["description"])
                else:
                    print("Error: Quote API service error")
        else:
            logger.debug("Response Body: %s", response)
            print("Error: Quote API service error")

    def get_quote(
        self,
        symbols: list,
        detail_flag: str = None,
        require_earnings_date: str = None,
        skip_mini_options_check: str = None,
        resp_format="xml",
    ) -> dict:
        """:description: Get quote data on symbols provided in the list args.

           :param symbols: Symbols in list args format. Limit 25.
           :type symbols: list[], required
           :param detail_flag: Market fields returned from a quote request, defaults to None
           :type detail_flag: str, optional
           :param require_earnings_date: Provides Earnings date if True, defaults to None
           :type require_earnings_date: str, optional
           :param skip_mini_options_check: Skips mini options check if True, defaults to None
           :type skip_mini_options_check: str, optional
           :param resp_format: Desired Response format, defaults to xml
           :type  resp_format: str, optional
           :return: Returns quote data on symbols provided
           :rtype: xml or json based on ``resp_format``
           :symbols values:
               * Limited to 25. If exceeded, first 25 will be processed with warnings
               * Equities format - ``symbol`` name sufficient, e.g. GOOGL.
               * Options format - ``underlier:year:month:day:optionType:strikePrice``
           :detailflag values:
               * fundamental - Instrument fundamentals and latest price
               * intraday - Performance for the current of most recent trading day
               * options - Information on a given option offering
               * week_52 - 52-week high and low (highest high and lowest low)
               * mf_detail - MutualFund structure gets displayed
               * all (default) - All of the above information and more
               * None - Defaults to all.
           :skipMiniOptionsCheck values:
               * True - Call is NOT made to check whether the symbol has mini options
               * False - Call is made to check whether the symbol has mini options
               * None - Call is made to check whether the symbol has mini options (default)
           :EtradeRef: https://apisb.etrade.com/docs/api/market/api-quote-v1.html

            """

        if detail_flag is not None:
            detail_flag = detail_flag.lower()

        assert detail_flag in (
            "fundamental",
            "intraday",
            "options",
            "week_52",
            "all",
            "mf_detail",
            None,
        )

        assert require_earnings_date in (True, False, None)
        assert skip_mini_options_check in (True, False, None)
        assert isinstance(symbols, list or tuple)

        if len(symbols) > 25:
            logger.warning("get_quote asked for %d requests; only first 25 returned" % len(symbols))

        args = list()
        if detail_flag is not None:
            args.append("detailflag=%s" % detail_flag.upper())
        if require_earnings_date:
            args.append("requireEarningsDate=true")
        if skip_mini_options_check is not None:
            args.append("skipMiniOptionsCheck=%s" % str(skip_mini_options_check))

        api_url = "%s%s%s" % (self.base_url, "/v1/market/quote/", ",".join(symbols[:25]))

        if resp_format.lower() == "json":
            api_url += ".json"
        if len(args):
            api_url += "?" + "&".join(args)
        logger.debug(api_url)

        req = self.session.get(api_url)
        req.raise_for_status()
        logger.debug(req.text)

        return xmltodict.parse(req.text) if resp_format.lower() == "xml" else req.json()
