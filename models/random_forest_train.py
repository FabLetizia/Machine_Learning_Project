import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
plt.style.use('grayscale')

from scipy import linalg
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
import os
import sys
import pickle
from stock_utils.stock_utils import create_train_data
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

class RF_training:

    def __init__(self, model_version, threshold=0.98, start_date=None, end_date=None):
        self.model_version = model_version
        self.threshold = threshold
        self.main_df = pd.DataFrame(columns=['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target'])

        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']
        sp500 = pd.read_csv('companies.csv')
        sp = list(sp500['Ticker'])
        stocks = dow + sp[:20]
        self.stocks = list(np.unique(stocks))

        self.scaler = MinMaxScaler()
        self.rf = RandomForestClassifier()

        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.save_model()

    def fetch_data(self):
        for stock in self.stocks:
            try: 
                df = create_train_data(stock, n=10)
                self.main_df = pd.concat([self.main_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error fetching data for {stock}: {str(e)}")
        
    def create_train_test(self):
        self.main_df = self.main_df.sample(frac=1, random_state=3).reset_index(drop=True)
        self.main_df['target'] = self.main_df['target'].astype('category')
        
        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scaler.fit_transform(self.main_df)
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.05, random_state=50, shuffle=True)

    def fit_model(self):
        self.rf.fit(self.train_x, self.train_y)
        self.predictions = self.rf.predict(self.test_x)
        self.score = self.rf.score(self.test_x, self.test_y)
        print(f'Random Forest model score: {self.score}')
        self.predictions_proba = self.rf.predict_proba(self.test_x)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)

    def _threshold(self, probs, threshold):
        print(f"Probs shape: {probs.shape}")
        print(f"Probs type: {type(probs)}")
        prob_thresholded = np.where(probs[:, 1] > threshold, 1, 0)
        return prob_thresholded

    def save_model(self):
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'rf_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.rf, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

if __name__ == "__main__":
    run_rf = RF_training('v1')

