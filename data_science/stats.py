import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import seaborn as sns

# progettata per calcolare e salvare le statistiche relative ai risultati di un modello di previsione finanziaria
class create_stats:

    #vengono inizializzati gli attributi dell'oggetto e vengono caricati i dati necessari per calcolare e salvare le statistiche. 
    def __init__(self, model_name, threshold, hold_till):
        
        self.model = model_name
        self.threshold = threshold
        self.hold_till = hold_till
        
        results_dir = ''' add results directory (la directory in cui si trovano i risultati) '''

        #Viene creato il nome della cartella folder_name utilizzando una stringa formattata che combina il nome del modello, la soglia e il periodo di detenzione.
        self.folder_name = f'{str(self.model)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, self.folder_name)

        #os.path.join per creare il path completo
        history_df_path = os.path.join(self.folder_dir, 'history_df.csv')

        self.history_df = pd.read_csv(history_df_path)

        #le colonne 'buy_date' e 'sell_date' nel DataFrame vengono convertite in oggetti datetime utilizzando pd.to_datetime
        self.history_df['buy_date'] = pd.to_datetime(self.history_df['buy_date'])
        self.history_df['sell_date'] = pd.to_datetime(self.history_df['sell_date'])
        
        params_path = os.path.join(self.folder_dir, 'params')
        #Viene aperto il file "params" in modalità binaria utilizzando open con il parametro 'rb', e il suo contenuto 
        # viene caricato come oggetto pickle utilizzando pickle.load. Il contenuto viene assegnato all'attributo self.params.
        with open(params_path, 'rb') as fp:
            self.params = pickle.load(fp)
        
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        with open(results_summary_path, 'rb') as fp:
            self.results_summary = pickle.load(fp)
        

        #get params from stored files
        #Alcuni parametri vengono estratti dai file caricati e assegnati come attributi dell'oggetto create_stats.
        # self.initial_capital viene impostato come il primo elemento di self.results_summary, self.total_gain come il secondo elemento,
        # self.start_date come il quarto elemento di self.params, e self.end_date come il quinto elemento.
        self.initial_capital = self.results_summary[0]
        self.total_gain = self.results_summary[1]
        self.start_date = self.params[4]
        self.end_date = self.params[5]

        self.calculate_stats()
        self.save_stats()
    
    ''' aggrega le informazioni contenute nel DataFrame history_df per calcolare statistiche come la percentuale di guadagno totale,
    i guadagni totali, il guadagno massimo, le perdite totali e la perdita massima. Queste statistiche vengono quindi assegnate agli attributi corrispondenti dell'oggetto create_stats. '''
    def calculate_stats(self):

        #calcolo della percentuale totale di guadagno: totale guadagnato/capitale iniziale * 100, arrotondato a 2 cifre decimali
        self.total_percentage = np.round(self.total_gain/self.initial_capital * 100, 2)

        #calcolo dei guadagni totali: viene calcolato sommando tutti i valori della colonna "net_gain" (guadagno netto) nel DataFrame history_df per cui il valore è maggiore di 0.
        self.total_gains = np.round(self.history_df[self.history_df['net_gain'] > 0]['net_gain'].sum(), 2)

        #calcolo del massimo guadagno: viene calcolato come il valore massimo della colonna "net_gain" nel DataFrame history_df per cui il valore è maggiore di 0.
        self.maximum_gain = np.round(self.history_df[self.history_df['net_gain'] > 0]['net_gain'].max(), 2)

        #calcolo delle perdite totali: viene calcolato sommando tutti i valori della colonna "net_gain" (guadagno netto) nel DataFrame history_df per cui il valore è minore di 0.
        self.total_losses = np.round(self.history_df[self.history_df['net_gain'] < 0]['net_gain'].sum(), 2)

        #calcolo della massima perdita: viene calcolato come il valore minimo della colonna "net_gain" nel DataFrame history_df per cui il valore è minore di 0.
        self.maximum_loss = np.round(self.history_df[self.history_df['net_gain'] < 0]['net_gain'].min())
    
    # il metodo save_stats legge un file CSV esistente, aggiunge una nuova riga contenente le statistiche calcolate e salva il DataFrame aggiornato nel file CSV. 
    # In questo modo, il file CSV conterrà una raccolta di statistiche per diversi modelli, soglie e periodi di detenzione
    def save_stats(self):

        # probabilmente model_result_summary.csv è da creare vuoto
        df = pd.read_csv('''add results/model_result_summary.csv path''')

        # viene creato un dizionario chiamato results_dict che contiene le statistiche calcolate e altre informazioni da salvare. 
        # Le chiavi del dizionario rappresentano i nomi delle colonne nel file CSV.
        results_dict = {'Model': f'{self.model}_{self.threshold}_{self.hold_till}',\
            'Gains': self.total_gains,
            'Losses': self.total_losses,
            'Profit': np.round(self.total_gain, 2),
            'Profit Percentage': self.total_percentage,
            'Maximum Gain': self.maximum_gain,
            'Maximum Loss': self.maximum_loss}
        #Il dizionario results_dict viene aggiunto come una nuova riga nel DataFrame df utilizzando il metodo append. 
        # L'argomento ignore_index=True garantisce che l'indice delle righe venga ricalcolato in modo incrementale.
        df = df.append(results_dict, ignore_index = True)
        #Il DataFrame aggiornato viene salvato nel file CSV specificato nel percorso 'add path results/model_result_summary.csv'. 
        # Questo viene fatto utilizzando il metodo to_csv` di pandas, passando il percorso del file CSV come argomento
        df.to_csv(''' add path results/model_result_summary.csv''')        

if __name__ == "__main__":
    cs = create_stats('LR_v1_predict', 1, 1)