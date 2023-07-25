"""
LR training 
authors: Alessandro Pesare, Fabio Letizia
"""
"""
Import di diverse librerie tra cui td.client per l'accesso a dati finanziari tramite TD Ameritrade API, 
requests per effettuare richieste HTTP, matplotlib.pyplot per la visualizzazione dei dati,
pandas per la manipolazione dei dati tabulari, pickle per la serializzazione degli oggetti Python, 
numpy per operazioni matematiche avanzate e sklearn per l'apprendimento automatico.
"""
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
from stock_utils import create_train_data
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

"""
La classe LR_training rappresenta il processo di addestramento del modello di regressione logistica.
Nel suo metodo __init__, la classe prende in input diversi parametri:
model_version: una stringa che rappresenta la versione del modello.
threshold: un valore di soglia opzionale (predefinito a 0.98) utilizzato per la classificazione binaria
basata sulle probabilità previste dal modello.
start_date e end_date: date di inizio e fine opzionali per l'intervallo di dati storici da considerare.
"""
class LR_training:

    def __init__(self, model_version, threshold = 0.98, start_date = None, end_date = None):

        self.model_version = model_version
        self.threshold = threshold
        #main dataframe
        self.main_df = pd.DataFrame(columns=['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target'])

        
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        #get stock ticker symbols
        print("Start inizialization")

        dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',\
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',\
        'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']

        sp500 = pd.read_csv('companies.csv')

        sp = list(sp500['Ticker'])

        stocks = dow + sp[:20]
        
        self.stocks = list(np.unique(stocks))

        #init models
        self.scaler = MinMaxScaler()
        self.lr = LogisticRegression()

        #run logistic regresion
        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.save_model()
        """
    fetch_data: recupera i dati di addestramento richiamando la funzione create_train_data del modulo stock_utils
    per ogni simbolo ticker.I dati vengono aggiunti al dataframe principale main_df.
    """
    def fetch_data(self):
        print("stat fetch data")
        """
        get train and test data
        """ 
        for stock in self.stocks:
            try: 
                print("try")
                df = create_train_data(stock, n = 10)
                print("Dati creati")
                print(df)
                self.main_df = pd.concat([self.main_df, df], ignore_index=True)
                print(f"Size of self.main_df after fetching data for {stock}: {len(self.main_df)}")
                print(f"Fetched data for {stock}. Total samples: {len(df)}")
            except Exception as e:
                print(f"Error fetching data for {stock}: {str(e)}")
        print(f'{len(self.main_df)} samples were fetched from the database..')
    """
    create_train_test: crea i dati di addestramento e test suddividendo il dataframe principale main_df in input (x) e output (y),
    utilizzando train_test_split da sklearn.model_selection.
    """
    def create_train_test(self):
        """
        create train and test data
        """
        self.main_df = self.main_df.sample(frac = 1, random_state = 3). reset_index(drop = True)
        self.main_df['target'] = self.main_df['target'].astype('category')

        # Check if the main_df is empty
        if self.main_df.empty:
            print("Error: The main DataFrame is empty. No data to split.")
            return
        
        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scaler.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.05, random_state = 50, shuffle = True)

        print('Created test and train data...')
    """
    fit_model: addestra il modello di regressione logistica utilizzando i dati di addestramento e calcola le previsioni
    sui dati di test. Vengono anche calcolate le previsioni soglia basate sul parametro threshold.
    """
    def fit_model(self):

        print('Training model...')
        self.lr.fit(self.train_x, self.train_y)
        
        #predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        print(f'Logistic regression model score: {self.score}')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)
    """
    _threshold  prende in input un array di probabilità previste e applica una soglia per ottenere le previsioni
    soglia corrispondenti. Restituisce un array numpy costituito di 0 o 1 a secondo della soglia.
    """
    def _threshold(self, predictions, threshold):

        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        return np.array(prob_thresholded)

#save_model: salva il modello addestrato e lo scaler utilizzati in file specifici.

    def save_model(self):

        #save models
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

import argparse

if __name__ == "__main__":
    run_lr = LR_training('v2')
