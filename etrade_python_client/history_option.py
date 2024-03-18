import yfinance as yf
import datetime


def get_option_data(stock_symbol, expiration_date, option_type, strike):
    stock = yf.Ticker(stock_symbol)
    option_chain = stock.option_chain(expiration_date)
    options = getattr(option_chain, "calls" if option_type.startswith("call") else "puts")
    option_data = options[options["strike"] == strike]
    return option_data


def get_option_history_data(contract_symbol, days_before_expiration=60):
    option = yf.Ticker(contract_symbol)
    option_info = option.info
    option_expiration_date = datetime.datetime.fromtimestamp(option_info["expireDate"])

    start_date = option_expiration_date - datetime.timedelta(days=days_before_expiration)
    option_history = option.history(start=start_date)
    return option_history


def main(*args):
    # Example:
    stock_symbol = "AAPL"
    expiration_date = "2024-03-22"
    # expiration_date = None
    option_type = "call"
    strike = 190.0

    option_data = get_option_data(stock_symbol, expiration_date, option_type, strike)
    print(expiration_date)
    print(option_data)
    for i, od in option_data.iterrows():
        print(od)
        contract_symbol = od["contractSymbol"]
        option_history = get_option_history_data(contract_symbol)
        print(option_history)
        first_option_history = option_history.iloc[0]
        first_option_history_date = option_history.index[0]
        first_option_history_close = first_option_history["Close"]
        print("For {}, the closing price was ${:.2f} on {}.".format(
            contract_symbol,
            first_option_history_close,
            first_option_history_date
        ))


if __name__ == "__main__":
    main()