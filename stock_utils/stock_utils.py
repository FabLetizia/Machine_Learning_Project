"""
authors - Alessandro Pesare, Fabio Letizia
stock utils for preparing training data.
"""

# Il modulo stock_utils contiene diverse funzioni utili per preparare i dati
#  di addestramento per il tuo modello

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
La funzione get_data(sym, start_date=None, end_date=None, n=10) è responsabile di ottenere
i dati storici di prezzo per l'azione specificata utilizzando Yahoo finance.
Dopo aver ottenuto i dati, la funzione get_data esegue ulteriori operazioni sui dati, ad esempio calcola 
i valori normalizzati dei prezzi e identifica i minimi e i massimi locali.
"""
def get_data(symbol, start_date=None, end_date=None, n=10):
    print("Start download")
    data = yf.download(symbol, start=start_date, end=end_date)
    print("End download")
    data.reset_index(inplace=True)

    data['date'] = pd.to_datetime(data['Date'], unit = 'ms')

#add the noramlzied value function and create a new column
    #la funzione apply() viene utilizzata per applicare una funzione personalizzata normalized_values() a ogni riga 
    #del DataFrame data. La funzione normalized_values() prende in input i valori high, low e close di ogni riga e
    #restituisce il valore normalizzato calcolato tramite la formula (close - low) / (high - low).
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.High, x.Low, x.Close), axis=1)

#column with local minima and maxima
    #La funzione argrelextrema() restituisce gli indici degli elementi dell'array data che
    #corrispondono ai minimi e ai massimi locali.
    #In sostanza, il parametro order viene utilizzato per determinare la larghezza della finestra di ricerca
    #per i minimi e i massimi relativi dell'array 'close', consentendo di trovare i minimi e i massimi locali
    #dell'azione in base a una finestra di osservazione specifica.
    data['loc_min'] = data.iloc[argrelextrema(data.Close.values, np.less_equal, order=n)[0]]['Close']
    data['loc_max'] = data.iloc[argrelextrema(data.Close.values, np.greater_equal, order=n)[0]]['Close']

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
returns the stock price given a date
"""
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

"""
normalize the price between 0 and 1.
"""
def normalized_values(high, low, close):
    epsilon = 10e-10  #epsilon to avoid deletion by 0
    high = high - low
    close = close - low
    return close / (high + epsilon)

"""
Questa funzione esegue una regressione lineare sui dati di input x e y e restituisce il coefficiente di regressione.
Nel caso della funzione linear_regression(x, y), si sta eseguendo una regressione lineare con una sola variabile di input x
e una sola variabile di output y. Pertanto, coef_ è un array bidimensionale con una sola riga e una sola colonna.
"""
def linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x, y)
    return lr.coef_[0][0]

"""
La funzione n_day_regression(n, df, idxs) esegue una regressione lineare per un determinato numero di giorni (n)
utilizzando una finestra mobile di dati. Calcola il coefficiente di regressione per ogni finestra e lo assegna al dataframe df.
Aggiunge i valori di regressione come una nuova colonna nel DataFrame df e restituisce il DataFrame modificato.
"""
def n_day_regression(n, df, idxs):
    #_varname_ è una variabile che viene creata all'interno della 
    # funzione utilizzando una f-string (format string) di Python.
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan # np.nan assegna un valore NaN (Not a Number) a ogni cella della nuova colonna _varname_

    for idx in idxs:
        if idx > n:
            # La riga y = df['close'][idx - n: idx].to_numpy() seleziona i valori di chiusura (close) dell'azione per 
            # gli ultimi n giorni fino all'indice corrente idx e li converte in un array NumPy utilizzando il metodo to_numpy().
            # Questi valori di chiusura verranno utilizzati come variabili di output y per il calcolo della regressione lineare.
            y = df['Close'][idx - n:idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value

    return df

"""
Questa funzione prende un'azione come argomento e restituire un dataframe contenente i dati di addestramento
per quell'azione. I dati di addestramento vengono ottenuti chiamando la funzione get_data con
l'azione specifica.
"""
def create_train_data(stocks, start_date=None, end_date=None, n=10):
    train_data = pd.DataFrame()

    for stock in stocks:
        #get data to a dataframe
        data, idx_with_mins, idx_with_maxs = get_data(stock, start_date, end_date)
        #create regressions for 3, 5 and 10 days
        data = n_day_regression(3, data, range(len(data)))
        data = n_day_regression(5, data, range(len(data)))
        data = n_day_regression(10, data, range(len(data)))
        data = n_day_regression(20, data, range(len(data)))

        data['normalized_value'] = normalized_values(data['High'], data['Low'], data['Close'])
        data['loc_min'] = data.iloc[argrelextrema(data['Close'].values, np.less_equal, order=n)[0]]['Close']
        data['loc_max'] = data.iloc[argrelextrema(data['Close'].values, np.greater_equal, order=n)[0]]['Close']

       # idx_with_mins = np.where(data['loc_min'] > 0)[0]
       # idx_with_maxs = np.where(data['loc_max'] > 0)[0]

#crea un nuovo DataFrame _data_ contenente solo le righe del DataFrame data che contengono minimi o massimi locali e 
    #poi reimposta gli indici del DataFrame in modo che siano numerati in ordine crescente a partire da 0.
        data = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop=True)
        #create a dummy variable for local_min (0) and max (1)
        data['target'] = [1 if x > 0 else 0 for x in data.loc_max]

        cols_of_interest = ['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
        data = data[cols_of_interest]

        train_data = pd.concat([train_data, data], ignore_index=True)

    return train_data.dropna(axis=0) # elimino i Nan prima di restituire il risultato

"""
this function create test data sample for logistic regression model
"""
def create_test_data(stocks, start_date=None, end_date=None, n=10):
    test_data = pd.DataFrame()

    for stock in stocks:
        data, _, _ = get_data(stock, start_date, end_date)
        #create regressions for 3, 5 and 10 days (ogni n_day_regression introduce una nuova colonna nel df n_reg)
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

'''
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
'''
