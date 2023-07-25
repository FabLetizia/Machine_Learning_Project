# Importiamo le librerie necessarie
import numpy as np
import pickle
from datetime import timedelta
import time
from stock_utils.stock_utils import timestamp, create_train_data, get_data, create_test_data, get_stock_price
from sklearn.ensemble import RandomForestClassifier

    
def load_RF(model_path):
        file = '/Users/alessandropesare/Desktop/ML_Project/saved_models/rf_v2.sav'
        loaded_model = pickle.load(open(file, 'rb'))
        return loaded_model

def load_scaler(scaler_path):
        file = '/Users/alessandropesare/Desktop/ML_Project/saved_models/scaler_v2.sav'
        loaded_model = pickle.load(open(file, 'rb'))
        return loaded_model

def predict(stock, start_date, end_date, threshold):
        print(f"Stock: {stock}")
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        print(f"Threshold: {threshold}")

        scaler = load_scaler('v2')
        rf = load_RF('v2')
        print(rf)
        data = create_test_data(stock, start_date, end_date)
        close_price = data['Close'].values[-1]

        input_data = data[['Volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']]
        input_data = input_data.to_numpy()[-1].reshape(1, -1)

        input_data_scaled = scaler.transform(input_data)
        prediction = rf.predict_proba(input_data_scaled)
        prediction_thresholded = _threshold(prediction, threshold)

        print(f"Prediction: {prediction}")
        print(f"Prediction Thresholded: {prediction_thresholded}")
        print(f"Close Price: {close_price}")

        return prediction[:, 0], prediction_thresholded[0], close_price

def _threshold(probs, threshold):
        prob_thresholded = [0 if x > threshold else 1 for x in probs]
        return np.array(prob_thresholded)

def sell(stock, buy_date, buy_price, todays_date, sell_perc, hold_till, stop_perc):
        print(f"Stock: {stock}")
        print(f"Buy date: {buy_date}")
        print(f"Buy price: {buy_price}")
        print(f"Today's date: {todays_date}")
        print(f"Sell percentage: {sell_perc}")
        print(f"Hold till: {hold_till}")
        print(f"Stop percentage: {stop_perc}")

        current_price = get_stock_price(stock, todays_date)
        sell_price = buy_price + buy_price * sell_perc
        stop_price = buy_price - buy_price * stop_perc
        sell_date = buy_date + timedelta(days=hold_till)

        time.sleep(1)

        if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
            return "SELL", current_price
        else:
            return "HOLD", current_price
