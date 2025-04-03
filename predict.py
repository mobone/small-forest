import logging
import pickle
import yfinance as yf

# Configure logging
logging.basicConfig(
    filename='predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from configparser import ConfigParser
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide
from alpaca.trading.enums import OrderType
from alpaca.trading.enums import OrderClass
from alpaca.trading.enums import TimeInForce

import datetime
import time
now = datetime.datetime.now()

config = ConfigParser()
config.read('./config.ini')

api_key = config.get('alpaca', 'api_key')
secret_key = config.get('alpaca', 'api_secret')
trading_client = TradingClient(
    api_key=api_key,
    secret_key=secret_key,
    paper=True
)

account = trading_client.get_account()
available_cash = account.cash

logging.info("Starting script")


stock = yf.Ticker("TQQQ")
data = stock.history(period='2d')
print(data)

if now.hour < 12:
    # read model from file
    model_filename = 'Naive_Bayes_model.sav'

    clf = pickle.load(open('./models/'+model_filename, 'rb'))

    entry_price = data["Open"].iloc[1]

    close_yesterday = data["Close"].iloc[0]
    open_today = data["Open"].iloc[1]
    overnight_percent_change = (open_today - close_yesterday) / close_yesterday


    # get yesterdays volume
    yesterdays_volume = data["Volume"].iloc[0]

    # predict
    prediction = clf.predict([[overnight_percent_change, yesterdays_volume]])[0]    
    logging.info(f"Prediction: {prediction}")

    quantity = round((float(available_cash)/float(entry_price)),0)-3
    logging.info(f"Number of shares: {quantity}")
    logging.info(f"Entry Price: {entry_price}")

    if prediction == 1:
        order_type = OrderSide.BUY
        with open('order_type.txt', 'w') as f:
            f.write(str('BUY'))
        logging.info(f"Order Type: BUY")
    else:
        order_type = OrderSide.SELL
        with open('order_type.txt', 'w') as f:
            f.write(str('SELL'))
        logging.info(f"Order Type: SELL")

    # insert prediction into sqlite3 table
    import sqlite3
    conn = sqlite3.connect('predictions.db')
    this_row = data.tail(1)
    this_row['Prediction'] = prediction
    # add todays date to the row
    this_row['Date'] = now.strftime("%Y-%m-%d")
    this_row.to_sql('predictions', conn, if_exists='append', index=False)    

    logging.info("Placing order")
    order_request = OrderRequest(
        symbol="TQQQ",
        qty=quantity,
        side=order_type,
        type=OrderType.MARKET,
        order_class=OrderClass.SIMPLE,
        time_in_force=TimeInForce.DAY,
        extended_hours=False
    )
    try:
        order_submission_response = trading_client.submit_order(order_data=order_request)
        logging.info(order_submission_response)
    except Exception as e:
        logging.error(f"Error while submitting order: {e}")
        exit(1)
    exit()

# check if code is being run after noon
# if so, sell all shares

if now.hour > 12:
    # read order type from file
    with open('order_type.txt', 'r') as f:
        read_order_type = f.read()

    if read_order_type == "SELL":
        order_type = OrderSide.BUY
        logging.info("Order Type: SELL detected, buying to exit.")
    else:
        order_type = OrderSide.SELL
        logging.info("Order Type: BUY detected, selling to exit.")

    # determine how many shares we hold
    positions = trading_client.get_all_positions()
    for position in positions:
        if position.symbol == "TQQQ":
            quantity = position.qty
            logging.info(f"Current position quantity: {quantity}")

    logging.info("Placing order")
    order_request = OrderRequest(
        symbol="TQQQ",
        qty=quantity,
        side=order_type,
        type=OrderType.MARKET,
        order_class=OrderClass.SIMPLE,
        time_in_force=TimeInForce.DAY,
        extended_hours=True
    )

    try:
        order_submission_response = trading_client.submit_order(order_data=order_request)

        # sleep for 5 seconds
        time.sleep(5)

        # Check the order status
        order_status = trading_client.get_order(order_id=order_submission_response.id)
        logging.info(f"Order Status: {order_status.status}")
        if order_status.status == "filled":
            logging.info("Order filled successfully.")
        else:
            logging.warning(f"Order not filled, current status: {order_status.status}")

        
    except Exception as e:
        logging.error(f"Error while submitting order: {e}")
        exit(1)


