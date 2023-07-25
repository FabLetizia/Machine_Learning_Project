"""
stock backtester to test the model given a dataset. 
author - Alessandro Pesare, Fabio Letizia
"""

import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import logistic_regression_inference
from datetime import datetime
from datetime import timedelta
import pandas as pd
from models.logistic_regression_inference import LR_v1_predict, LR_v1_sell
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm

class backtester(simulator):

    def __init__(self, stocks_list, model, capital, start_date, end_date, threshold, sell_perc, hold_till,\
         stop_perc):
        
        super().__init__(capital) #initialize simulator

        self.stocks = stocks_list
        self.model = model
        self.start_date = start_date
        self.day = start_date
        self.end_date = end_date  
        self.status = 'buy' #the status says if the backtester is in buy mode or sell mode
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc

        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, 'results')
        folder_name = f'{str(self.model.__name__)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, folder_name)
        if not os.path.exists(self.folder_dir):
            #create a new folder
            os.makedirs(self.folder_dir)
      
    def backtest(self):
        """
        start backtesting
        """
        delta = timedelta(days = 1)
        
        #progress bar to track progress
        total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Progress', total = total_days)

        while self.day <= self.end_date:
            
            #daily scanner dict
            self.daily_scanner = {}  
            if self.status == 'buy':
                #scan stocks for the day
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommended_stock = list(self.daily_scanner.keys())[0]
                    recommended_price = list(self.daily_scanner.values())[0][2]
                    self.buy(recommended_stock, recommended_price, self.day) #buy stock
                    # print(f'Bought {recommended_stock} for {recommended_price} on the {self.day}')
                    self.status = 'sell' #change the status to sell
                else:
                    # print('No recommendations')
                    pass
            else: #if the status is sell
                #get stock price on the day
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)

                    if np.random.choice([0, 1]) == 1: #randomly sell
                        # print(f'Sold {s} for {current_price} on {self.day}')
                        self.sell(s, current_price, self.buy_orders[s][1], self.day)
                        self.status = 'buy'              
            #go to next day
            self.day += delta
            d += 1
            pbar.update(1)
        pbar.close()
        #sell the final stock and print final capital also print stock history 
        self.print_bag()
        self.print_summary()   
        return

    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries to td database and get data of a particular stock on a given day back to certain amount of days
        (default is 30). 
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)        
        # prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, threshold = 0.5)
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.threshold)
        return prediction[0], prediction_thresholded, close_price

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        stock = np.random.choice(self.stocks)
        prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
        self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)

if __name__ == "__main__":
    #stocks list
    dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',\
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',\
        'CRM', 'VZ', 'V']

    other = ['AMD', 'MU', 'ABT', 'AAL', 'UAL', 'DAL', 'ANTM', 'ATVI', 'BAC', 'PNC', 'C', 'EBAY', 'AMZN', 'GOOG', 'FB', 'SNAP', 'TWTR'\
        'FDX', 'MCD', 'PEP', ]
    
    stocks = list(np.unique(dow + other))
    # threshold soglia di confidenza, trattiene l'azione fino a una % di guadagno(sell_perc) o a una certa % di perdita(stop_perc)
    # o entro un certo numero di giorni(hold_till).
    back = backtester(stocks, LR_v1_predict, 3000, datetime(2019, 1, 1), datetime(2019, 1, 10), threshold = 1, sell_perc = 0.04, hold_till = 10,\
        stop_perc = 0.005)
    back.backtest()

    


    