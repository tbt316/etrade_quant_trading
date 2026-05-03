
import configparser
import os
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.append('/Users/btian/EtradePythonClient/etrade_python_client')

from accounts.accounts_bo import Accounts
from order.order_bo import Order

def get_today_orders():
    config = configparser.ConfigParser()
    config.read('/Users/btian/EtradePythonClient/etrade_python_client/config.ini')
    
    accounts = Accounts(config, 'live')
    # We need a session, which is usually handled in etrade_cover_call_new.py
    # For a quick check, let's see if we can load the session from the file
    
    session_file = '/Users/btian/EtradePythonClient/etrade_python_client/.etrade_oauth'
    if not os.path.exists(session_file):
        print("OAuth file not found.")
        return

    # Actually etrade_cover_call_new uses pyetrade and handles its own session
    # Let's try to mimic the initialization
    from rauth import OAuth1Service
    import pyetrade

    consumer_key = config['DEFAULT'].get('PROD_CONSUMER_KEY')
    consumer_secret = config['DEFAULT'].get('PROD_CONSUMER_SECRET')
    
    with open(session_file, 'r') as f:
        oauth_data = json.load(f)
        resource_owner_key = oauth_data['access_token']
        resource_owner_secret = oauth_data['access_token_secret']

    base_url = "https://api.etrade.com"
    service = OAuth1Service(
        name="etrade",
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        request_token_url=f"{base_url}/oauth/request_token",
        access_token_url=f"{base_url}/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etradetoken?key={}&token={}",
        base_url=base_url
    )
    
    session = service.get_session((resource_owner_key, resource_owner_secret))
    
    # Get account
    acct = pyetrade.ETradeAccounts(consumer_key, consumer_secret, resource_owner_key, resource_owner_secret, dev=False)
    account_list = acct.list_accounts(resp_format='json')
    account_id_key = account_list['AccountListResponse']['Accounts']['Account'][0]['accountIdKey']
    
    # Get orders
    orders_api = pyetrade.ETradeOrder(consumer_key, consumer_secret, resource_owner_key, resource_owner_secret, dev=False)
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching orders for {today}...")
    
    # List orders with different statuses
    for status in ['OPEN', 'EXECUTED', 'CANCELLED', 'EXPIRED']:
        print(f"\n--- Status: {status} ---")
        resp = orders_api.list_orders(account_id_key, status=status, resp_format='json')
        
        if 'OrdersResponse' in resp and 'Order' in resp['OrdersResponse']:
            for order in resp['OrdersResponse']['Order']:
                for detail in order.get('OrderDetail', []):
                    placed_time = detail.get('placedTime')
                    dt_str = "N/A"
                    if placed_time:
                        dt = datetime.fromtimestamp(placed_time / 1000)
                        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"Order ID: {order.get('orderId')} | Status: {detail.get('status')} | Placed: {dt_str} | Limit: {detail.get('limitPrice')}")
                    if detail.get('status') == 'EXECUTED':
                         # Print fill price
                         for inst in detail.get('Instrument', []):
                             print(f"  Avg Fill: {inst.get('averageExecutionPrice')}")
        else:
            print(f"No {status} orders found.")

if __name__ == "__main__":
    get_today_orders()
