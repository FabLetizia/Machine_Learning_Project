"""
authors - Alessandro Pesare, Fabio Letizia
stock utils for preparing training data.
"""

# Il modulo stock_utils contiene diverse funzioni utili per preparare i dati di addestramento per il tuo modello

from td.client import TDClient
import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
#from datetime import datetime

#TD API - 
TD_API = 'XXXXX' ### your TD ameritrade api key

"""
Questa funzione converte una data dt in un timestamp UNIX espresso in millisecondi.
La funzione timestamp(dt) prende un argomento dt, un oggetto di tipo datetime che rappresenta una data e un'ora specifiche.
La variabile epoch viene inizializzata con il valore di datetime.utcfromtimestamp(0).
Questa chiamata restituisce un oggetto datetime che rappresenta la data e l'ora corrispondenti all'inizio dell'epoca Unix,
ovvero 1 gennaio 1970 00:00:00 UTC.
Successivamente, la funzione calcola la differenza tra l'argomento dt e l'epoca (in millisecondi) 
"""
def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)

"""
Questa funzione esegue una regressione lineare sui dati di input x e y e restituisce il coefficiente di regressione.
Nel caso della funzione linear_regression(x, y), si sta eseguendo una regressione lineare con una sola variabile di input x
e una sola variabile di output y. Pertanto, coef_ è un array bidimensionale con una sola riga e una sola colonna.
"""
def linear_regression(x, y):
    #fit linear regression
    lr = LinearRegression()
    lr.fit(x, y)
    
    return lr.coef_[0][0]
"""
La funzione n_day_regression(n, df, idxs) esegue una regressione lineare per un determinato numero di giorni (n)
utilizzando una finestra mobile di dati. Calcola il coefficiente di regressione per ogni finestra e lo assegna al dataframe df.
Aggiunge i valori di regressione come una nuova colonna nel DataFrame df e restituisce il DataFrame modificato.
"""
def n_day_regression(n, df, idxs):
    #_varname_ è una variabile che viene creata all'interno della funzione utilizzando una f-string (format string) di Python. 
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan # np.nan assegna un valore NaN (Not a Number) a ogni cella della nuova colonna _varname_

    for idx in idxs:
        if idx > n:
            # La riga y = df['close'][idx - n: idx].to_numpy() seleziona i valori di chiusura (close) dell'azione per 
            # gli ultimi n giorni fino all'indice corrente idx e li converte in un array NumPy utilizzando il metodo to_numpy().
            # Questi valori di chiusura verranno utilizzati come variabili di output y per il calcolo della regressione lineare.
            y = df['close'][idx - n: idx].to_numpy()  
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

"""
normalize the price between 0 and 1.
"""
def normalized_values(high, low, close):

    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    high = high - low
    close = close - low
    return close/(high + epsilon)

"""
returns the stock price given a date
"""
def get_stock_price(stock, date):

    start_date = date - timedelta(days = 10)
    end_date = date
    
    #enter url of database
    url = f'https://api.tdameritrade.com/v1/marketdata/{stock}/pricehistory'

    query = {'apikey': str(TD_API), 'startDate': timestamp(start_date), \
            'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
            'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}

    #request
    results = requests.get(url, params = query)
    data = results.json()
    
    try:
        #converte i dati dal formato JSON in un DataFrame di Pandas, il dizionario data restituito dalla richiesta
        #contiene una chiave "candles" che rappresenta una lista di dizionari, ognuno dei quali rappresenta un singolo
        #giorno di dati dei prezzi dell'azione
        data = pd.DataFrame(data['candles'])
        #converte la colonna 'datetime' del DataFrame data in un formato di data/ora leggibile da Pandas.
        data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')
        #restituisce il prezzo di chiusura ('close') dell'ultimo giorno dei dati dei prezzi dell'azione,
        #selezionando la colonna 'close' del DataFrame data e utilizzando l'attributo values per accedere ai
        #valori del DataFrame come un array NumPy, quindi selezionando l'ultimo valore dell'array con l'indice -1.
        #In questo modo, la funzione restituisce il prezzo di chiusura dell'azione più recente disponibile nei dati
        #storici ottenuti dalla richiesta all'API di TD Ameritrade.
        return data['close'].values[-1]
    except:
        pass

"""
La funzione get_data(sym, start_date=None, end_date=None, n=10) è responsabile di ottenere
i dati storici di prezzo per l'azione specificata utilizzando l'API di TD Ameritrade.
Dopo aver ottenuto i dati, la funzione get_data esegue ulteriori operazioni sui dati, ad esempio calcola 
i valori normalizzati dei prezzi e identifica i minimi e i massimi locali.
"""
def get_data(sym, start_date = None, end_date = None, n = 10):

    #enter url
    url = f'https://api.tdameritrade.com/v1/marketdata/{sym}/pricehistory'
    
    if start_date:
        payload = {'apikey': str(TD_API), 'startDate': timestamp(start_date), \
            'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
            'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}
    else:
        payload = {'apikey': str(TD_API), 'startDate': timestamp(datetime(2007, 1, 1)), \
            'endDate': timestamp(datetime(2020, 12, 31)), 'periodType': 'year', 'frequencyType': \
            'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}
            
    #request
    results = requests.get(url, params = payload)
    data = results.json()
    
    #change the data from ms to datetime format
    data = pd.DataFrame(data['candles'])
    data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')

    #add the noramlzied value function and create a new column
    #la funzione apply() viene utilizzata per applicare una funzione personalizzata normalized_values() a ogni riga 
    #del DataFrame data. La funzione normalized_values() prende in input i valori high, low e close di ogni riga e
    #restituisce il valore normalizzato calcolato tramite la formula (close - low) / (high - low).
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    #La funzione argrelextrema() restituisce gli indici degli elementi dell'array data che
    #corrispondono ai minimi e ai massimi locali.
    #In sostanza, il parametro order viene utilizzato per determinare la larghezza della finestra di ricerca
    #per i minimi e i massimi relativi dell'array 'close', consentendo di trovare i minimi e i massimi locali
    #dell'azione in base a una finestra di osservazione specifica.
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

    #idx with mins and max
    #a funzione np.where() di NumPy per trovare gli indici delle righe del DataFrame data che contengono
    #un valore positivo sia nella colonna 'loc_min' che nella colonna 'loc_max'
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    #la funzione restituisce il DataFrame data con le colonne 'loc_min' e 'loc_max', che contengono i valori
    #del prezzo di chiusura dell'azione corrispondenti ai minimi e ai massimi locali, insieme agli indici delle
    #righe che contengono i minimi e i massimi locali.
    return data, idx_with_mins, idx_with_maxs

"""
Questa funzione prende un'azione come argomento e restituire un dataframe contenente i dati di addestramento
per quell'azione. I dati di addestramento vengono ottenuti chiamando la funzione get_data con
l'azione specifica.
"""
def create_train_data(stock, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, start_date, end_date, n)
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(5, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(10, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(20, data, list(idxs_with_mins) + list(idxs_with_maxs))

    #crea un nuovo DataFrame _data_ contenente solo le righe del DataFrame data che contengono minimi o massimi locali e 
    #poi reimposta gli indici del DataFrame in modo che siano numerati in ordine crescente a partire da 0.
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop = True) 
    
    #create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]
    
    #columns of interest
    cols_of_interest = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
    _data_ = _data_[cols_of_interest]
    
    return _data_.dropna(axis = 0) # elimino i Nan prima di restituire il risultato

"""
this function create test data sample for logistic regression model
"""
def create_test_data_lr(stock, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, _, _ = get_data(stock, start_date, end_date, n)
    idxs = np.arange(0, len(data))
    
    #create regressions for 3, 5 and 10 days (ogni n_day_regression introduce una nuova colonna nel df n_reg)
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
    
    cols = ['close', 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    data = data[cols]

    return data.dropna(axis = 0)

def predict_trend(stock, _model_, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, _, _ = get_data(stock, start_date, end_date, n)
    
    idxs = np.arange(0, len(data))
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
        
    #create a column for predicted value
    data['pred'] = np.nan

    #get data
    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data[cols]

    #scale the x data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    #x.shape[0] restituisce la lunghezza del primo asse dell'array
    for i in range(x.shape[0]):
        
        try:
            data['pred'][i] = _model_.predict(x[i, :])

        except:
            data['pred'][i] = np.nan

    return data
