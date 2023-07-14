"""
authors - Alessandro Pesare, Fabio Letizia
stock utils for preparing training data.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import requests

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime

def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)

def get_data(symbol, start_date=None, end_date=None, n=10):
    print("Start download")
    data = yf.download(symbol, start=start_date, end=end_date)
    print("End download")
    data.reset_index(inplace=True)

    data['date'] = pd.to_datetime(data['Date'], unit = 'ms')

    data['normalized_value'] = data.apply(lambda x: normalized_values(x.High, x.Low, x.Close), axis=1)

    data['loc_min'] = data.iloc[argrelextrema(data.Close.values, np.less_equal, order=n)[0]]['Close']
    data['loc_max'] = data.iloc[argrelextrema(data.Close.values, np.greater_equal, order=n)[0]]['Close']

    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]

    return data, idx_with_mins, idx_with_maxs

def get_stock_price(stock, date):
    start_date = date - timedelta(days=10)
    end_date = date
    
    stock_data = yf.download(stock, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    
    try:
        # Get the closing price of the latest available date
        latest_close_price = stock_data['Close'].iloc[-1]
        return latest_close_price
    except IndexError:
        pass

# Function to calculate normalized values
def normalized_values(high, low, close):
    epsilon = 10e-10
    high = high - low
    close = close - low
    return close / (high + epsilon)

# Function to perform linear regression
def linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.coef_[0][0]

# Function to perform n-day regression
def n_day_regression(n, df, idxs):
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            y = df['Close'][idx - n:idx].to_numpy()
            x = np.arange(0, n)
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef

    return df

# Function to create training data
def create_train_data(stocks, start_date=None, end_date=None, n=10):
    train_data = pd.DataFrame()

    for stock in stocks:
        data, _, _ = get_data(stock, start_date, end_date)
        data = n_day_regression(3, data, range(len(data)))
        data = n_day_regression(5, data, range(len(data)))
        data = n_day_regression(10, data, range(len(data)))
        data = n_day_regression(20, data, range(len(data)))

        data['normalized_value'] = normalized_values(data['High'], data['Low'], data['Close'])
        data['loc_min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order=n)[0]]['Close']
        data['loc_max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order=n)[0]]['Close']

        idx_with_mins = np.where(data['loc_min'] > 0)[0]
        idx_with_maxs = np.where(data['loc_max'] > 0)[0]

        data = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop=True)
        data['target'] = [1 if x > 0 else 0 for x in data.loc_max]

        cols_of_interest = ['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
        data = data[cols_of_interest]

        train_data = pd.concat([train_data, data], ignore_index=True)

    return train_data.dropna(axis=0)

# Function to create test data
def create_test_data(stocks, start_date=None, end_date=None, n=10):
    test_data = pd.DataFrame()

    for stock in stocks:
        data, _, _ = get_data(stock, start_date, end_date)
        data = n_day_regression(3, data, range(len(data)))
        data = n_day_regression(5, data, range(len(data)))
        data = n_day_regression(10, data, range(len(data)))
        data = n_day_regression(20, data, range(len(data)))

        data['normalized_value'] = normalized_values(data['High'], data['Low'], data['Close'])

        cols = ['Close', 'Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
        data = data[cols]

        test_data = pd.concat([test_data, data], ignore_index=True)

    return test_data.dropna(axis=0)

# List of Dow 30 stocks
dow30_stocks = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP',
                'HD', 'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT', 'TRV', 'JNJ',
                'CVX', 'MCD', 'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD']

# List of 20 important S&P 500 stocks (for demonstration purposes)
sp500_stocks = ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'MA', 'BAC', 'NFLX', 'ADBE',
                'DIS', 'PYPL', 'CMCSA', 'COST', 'PEP', 'INTU', 'CSCO', 'AVGO', 'TXN', 'CHTR']

# Training and validation data
train_data = create_train_data(dow30_stocks + sp500_stocks, start_date='2007-01-01', end_date='2020-12-31')

# Test data
test_data = create_test_data(dow30_stocks + sp500_stocks, start_date='2021-01-01', end_date='2021-12-31')

# Print the training data
print("Training Data:")
print(train_data.head())

# Print the test data
print("Test Data:")
print(test_data.head())
