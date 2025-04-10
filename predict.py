import logging
import pickle
import yfinance as yf
import sqlite3

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
import math 

logging.info("Program starting")

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

try:
    account = trading_client.get_account()
    available_cash = account.cash
except Exception as e:
    logging.error(f"Error fetching account information: {e}")
    exit(1)



try:
    stock = yf.Ticker("TQQQ")
    data = stock.history(period="252d")
except Exception as e:
    logging.error(f"Error fetching stock data: {e}")
    exit(1)

logging.info("Fetched stock data for TQQQ")

try:
    
    
    data["Percent Change"] = (data["Close"] - data["Open"]) / data["Open"]
    data["Overnight Percent Change"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    # volume the day before
    #data["Volume"] = data["Volume"].shift(1)
    
    # shift all columns that arent open high low close 
    for col in data.columns:
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Percent Change', 'Overnight Percent Change']:
            data[col] = data[col].shift(1)
except Exception as e:
    logging.error(f"Error processing stock data: {e}")
    exit(1)




#data = data.tail)  # Get the last two rows (yesterday and today)



import ta

try:
    data_with_ta = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    del data_with_ta['others_dr']
    del data_with_ta['others_dlr']
except Exception as e:
    logging.error(f"Error adding technical indicators: {e}")
    exit(1)



# read the top features from disk into a list
with open('top_features.txt', 'r') as f:
    top_features = f.read().splitlines()


#print(data_with_ta[top_features])




if now.hour < 12:
    logging.info("Starting opening script")
    # read model from file
    model_filename = 'Logistic_Regression_model.sav'

    clf = pickle.load(open('./models/'+model_filename, 'rb'))

    entry_price = data["Open"].iloc[-1]


    '''
    try:
        # make entry price be the most recent open price
        entry_price = data["Open"].iloc[-1]
        # make the close yesterday be the last close price before today
        close_yesterday = data["Close"].iloc[-2]
        # make the open today be the most recent open price
        open_today = data["Open"].iloc[-1]
    except Exception as e:
        logging.error(f"Error fetching today's open price: {e}")
        logging.info("Exiting script due to missing open price data.")
        exit(1)
    overnight_percent_change = (open_today - close_yesterday) / close_yesterday
    data['Overnight Percent Change'] = overnight_percent_change
    
    # get yesterdays volume
    #yesterdays_volume = data["Volume"].iloc[0]
    '''

    # predict
    prediction = clf.predict(data_with_ta[top_features].tail(1))[0]
    print(top_features)
    print("prediction", prediction)
    logging.info(f"Top features: {top_features}")
    logging.info(f"Prediction: {prediction}")

    quantity = math.floor( (float(available_cash) / float(entry_price) )) - 4

    logging.info(f"Available Cash: {available_cash}")
    logging.info(f"Entry Price: {entry_price}")
    logging.info(f"Number of shares: {quantity}")

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
        
    this_row = data.tail(1)
    this_row['Prediction'] = prediction
    # add todays date to the row
    this_row['Date'] = now.strftime("%Y-%m-%d")
    

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
        
        
        '''
        # sleep for 10 seconds
        time.sleep(10)

        # Check the order status
        order_status = trading_client.get_order(order_id=order_submission_response.id)
        logging.info(f"Order Status: {order_status.status}")
        if order_status.status == "filled":
            logging.info("Order filled successfully.")
        else:
            logging.warning(f"Order not filled, current status: {order_status.status}")

        '''
    except Exception as e:
        logging.error(f"Error while submitting order: {e}")
    
    try:
        conn = sqlite3.connect('predictions.db')
        this_row.to_sql('predictions', conn, if_exists='append', index=False)    
    except Exception as e:
        logging.error(f"Error inserting prediction into database: {e}")
        
    exit()

# check if code is being run after noon
# if so, sell all shares

if now.hour > 12:
    logging.info("Starting closing script")
    # read order type from file
    with open('order_type.txt', 'r') as f:
        read_order_type = f.read()

    if read_order_type == "SELL":
        order_type = OrderSide.BUY
        logging.info("Order Type: SELL detected, buying to exit.")
    else:
        order_type = OrderSide.SELL
        logging.info("Order Type: BUY detected, selling to exit.")
    
    quantity = 0
    # determine how many shares we hold
    positions = trading_client.get_all_positions()
    logging.info("Current positions:")
    logging.info(str(positions))
    # if no positions exit
    if not positions:
        logging.info("No current positions. Exiting script.")
        exit(0)

    for position in positions:
        if position.symbol == "TQQQ":
            quantity = position.qty
            logging.info(f"Current position quantity: {quantity}")
    if quantity == 0:
        logging.info("No shares to exit. Exiting script.")
        exit(0)
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

        # sleep for 10 seconds
        time.sleep(10)

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


