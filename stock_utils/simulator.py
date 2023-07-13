"""
stock simulator to trade in real time. 
authors Alessandro Pesare, Fabio Letizia
Questa classe consente agli utenti di simulare l'acquisto e la vendita di azioni in tempo reale.
"""

import numpy as np
import math
import pandas as pd

class simulator:
    """
    Costruttore che inizializza l'oggetto del simulatore. Prende il capitale iniziale come parametro e imposta
    vari attributi come il capitale, il capitale iniziale, il guadagno totale, gli ordini di acquisto
    e la cronologia delle transazioni.
    """
    def __init__(self, capital):
        self.capital = capital
        self.initial_capital = capital #keep a copy of the initial capital
        self.total_gain = 0
        self.buy_orders = {}
        self.history = []
        #create a pandas df to save history
        cols = ['stock', 'buy_price', 'n_shares', 'sell_price', 'net_gain', 'buy_date', 'sell_date']
        self.history_df = pd.DataFrame(columns = cols)
    """
    Questo metodo permette di acquistare azioni. Prende il nome del titolo, il prezzo di acquisto e la data di acquisto
    come parametri. Calcola la quantità di azioni da acquistare in base al prezzo di acquisto e al capitale disponibile.
    Sottrae il costo delle azioni acquistate dal capitale e tiene traccia dell'ordine di acquisto nel dizionario buy_orders.
    """
    def buy(self, stock, buy_price, buy_date):
        """
        function takes buy price and the number of shares and buy the stock
        """
        #calculate the procedure
        n_shares = self.buy_percentage(buy_price)
        self.capital = self.capital - buy_price * n_shares
        self.buy_orders[stock] = [buy_price, n_shares, buy_price * n_shares, buy_date]

    """
    Questo metodo permette di vendere azioni. Prende il nome del titolo, il prezzo di vendita, il numero di azioni da vendere
    e la data di vendita come parametri. Recupera il prezzo di acquisto, il numero di azioni e altre informazioni relative 
    all'ordine di acquisto dal dizionario buy_orders. Calcola il valore della vendita e aggiunge l'importo alla variabile capital.
    Se vengono vendute tutte le azioni, registra la transazione nella cronologia (history) e rimuove l'ordine di acquisto dal
    dizionario buy_orders. Se vengono vendute solo alcune azioni, aggiorna il numero di azioni e il valore dell'ordine di
    acquisto nel dizionario buy_orders.
    """
    def sell(self, stock, sell_price, n_shares_sell, sell_date):
        """
        function to sell stock given the stock name and number of shares
        """
        buy_price, n_shares, _, buy_date = self.buy_orders[stock]
        sell_amount = sell_price * (n_shares_sell)

        self.capital = self.capital + sell_amount

        if (n_shares - n_shares_sell) == 0: #if sold all
            self.history.append([stock, buy_price, n_shares, sell_price, buy_date, sell_date])
            del self.buy_orders[stock]
        else:
            n_shares = n_shares - n_shares_sell
            self.buy_orders[stock][1] = n_shares
            self.buy_orders[stock][2] = buy_price * n_shares

    # Questo metodo calcola la percentuale di capitale da investire in un'azione e restituisce il numero di azioni da acquistare
    def buy_percentage(self, buy_price, buy_perc = 1):
        """
        this function determines how much capital to spend on the stock and returns the number of shares
        """
        stock_expenditure = self.capital * buy_perc
        n_shares = math.floor(stock_expenditure / buy_price)
        return n_shares
    
    # Questo metodo stampa l'elenco delle azioni attualmente detenute, insieme ai relativi prezzi di acquisto,
    # il numero di azioni e il valore totale.
    def print_bag(self):
        """
        print current stocks holding
        """
        print ("{:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'TOTAL VALUE'))
        for key, value in self.buy_orders.items():
            print("{:<10} {:<10} {:<10} {:<10}".format(key, value[0], value[1], value[2]))
        print('\n')  

    # Questo metodo crea un riepilogo delle transazioni effettuate. Se il parametro print_results è impostato su True,
    # stampa anche i risultati. 
    def create_summary(self, print_results = False):
        """
        create summary
        """
        if print_results:
            print ("{:<10} {:<10} {:<10} {:<10} {:<10}".format('STOCK', 'BUY PRICE', 'SHARES', 'SELL PRICE', 'NET GAIN'))  
        
        for values in self.history:
            net_gain = (values[3] - values[1]) * values[2]
            self.total_gain += net_gain
            self.history_df = self.history_df.append({'stock': values[0], 'buy_price': values[1], 'n_shares': values[2], 'sell_price': values[3]\
                 ,'net_gain': net_gain, 'buy_date': values[4], 'sell_date': values[5]}, ignore_index = True)
                    
            if print_results:
                print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(values[0], values[1], values[2], values[3], np.round(net_gain, 2)))

    # Questo metodo stampa il riepilogo dei risultati. Chiama il metodo create_summary per generare i risultati e quindi
    # stampa il saldo iniziale, il saldo finale, il guadagno totale e il rapporto tra guadagno e capitale in percentuale.

    def print_summary(self):
        """
        prints the summary of results
        """
        self.create_summary(print_results = True)
        print('\n')
        print(f'Initial Balance: {self.initial_capital:.2f}')
        print(f'Final Balance: {(self.initial_capital + self.total_gain):.2f}')
        print(f'Total gain: {self.total_gain:.2f}')
        print(f'P/L : {(self.total_gain/self.initial_capital)*100:.2f} %')
        print('\n')