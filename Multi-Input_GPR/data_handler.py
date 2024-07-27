import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os
import tensorflow as tf
from OrdinalEntroPy.OrdinalEntroPy import PE, WPE, RPE, DE, RDE, RWDE

class DataHandler:
    def __init__(self, train_start_date, train_end_date):
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    
    def fetch_and_save_data(self, ticker, period):
        load_dotenv()
        api_token = os.getenv('API_TOKEN')
        url = f'https://eodhd.com/api/eod/{ticker}.US?period={period}&api_token={api_token}&fmt=json&from={self.train_start_date}&to={self.train_end_date}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        csv_file_path = f'../Stocks/{ticker}/{ticker}_us_{period}.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        df.to_csv(csv_file_path, index=False)
    
    # Process single file data, return normalized X and Y, X as day_of_year, Y as return
    def process_data(self, file_type, ticker, period, predict_Y='close', normalize=True, isFetch=False):
        if isFetch:
            self.fetch_and_save_data(ticker, period)
            print(f'{ticker} data fetched and saved')
        
        file_path = f'../{file_type}/{ticker}/{ticker}_us_{period}.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        df = df[(df['date'] >= self.train_start_date) & (df['date'] <= self.train_end_date)]
        df['day_of_year'] = df['date'].apply(self.convert_to_day_of_year)
        
        
        df['return'] = df['close'].pct_change()
        first_return = df['return'].iloc[1]
        df.fillna({'return': first_return}, inplace=True)
        df['intraday_return'] = (df['close'] - df['open']) / df['open']
        
        return self.normalize_and_reshape(df, column=predict_Y)
    
    def process_2D_X(self, ticker, start_date, end_date, predict_Y='close'):
        
        file_path = f'../Stocks/{ticker}/{ticker}.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
        df['day_of_year'] = df['date'].apply(self.convert_to_day_of_year)
        
        df['return'] = df['close'].pct_change()
        first_return = df['return'].iloc[1]
        df.fillna({'return': first_return}, inplace=True)
        df['intraday_return'] = (df['close'] - df['open']) / df['open']

        X_tf, Y_tf, df['date'], mean, std = self.normalize_and_reshape(df, column=predict_Y)

        # # Convert TensorFlow tensors to numpy arrays
        # X_np = X_tf.numpy()
        # Y_np = Y_tf.numpy()
        
        # # Ensure Y is 2D
        # if Y_np.ndim == 1:
        #     Y_np = Y_np.reshape(-1, 1)
        
        # X = np.column_stack((X_np, Y_np))
        # print(X_np.shape)  # (n, 1)
        # print(Y_np.shape)   # (n, 1)
        # print(X.shape)  # (n, 2)
            
        return X_tf, Y_tf, df['date'], mean, std
    
    # Combine the arrays to vectorize the input
    # X is a list of tensors, e.g. [X_tf, X_tf_2, Y_tf_2]
    def concatenate_X(self, X):
        if not isinstance(X, (list, tuple)):
            raise ValueError("Input X should be a list or tuple of tensors")

        # Ensure there's at least one array
        if len(X) < 1:
            raise ValueError("Input X should contain at least one tensor array")

        # Convert all inputs to numpy arrays if they aren't already
        if not all(x.shape == X[0].shape for x in X):
            raise ValueError("All input tensors should have the same shape")

        X_arrays = [x.numpy().reshape(-1, 1) for x in X]

        # Concatenate along the second axis (column-wise)
        X_combined = np.concatenate(X_arrays, axis=1)

        # for i in range(len(X)):
        #     if i == 0:
        #         # Convert TensorFlow tensors to numpy arrays
        #         X_combined = X[i].numpy().reshape(-1)
        #     else:
        #         X_combined = np.concatenate([X_combined, X[i].numpy().reshape(-1)])

        print(X_combined.shape)
        return X_combined

    def convert_to_day_of_year(self, date):
        start_date = pd.Timestamp(self.train_start_date)
        return (date - start_date).days

    def normalize_and_reshape(self, df, column='close'):
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std
        Y = df[column].values
        X = df['day_of_year'].values
        Y_reshaped = Y.reshape(-1, 1)
        X_reshaped = X.reshape(-1, 1)
        X_tf = tf.convert_to_tensor(X_reshaped, dtype=tf.float64)
        Y_tf = tf.convert_to_tensor(Y_reshaped, dtype=tf.float64)
        return X_tf, Y_tf, df['date'], mean, std
    
    def generate_future_dates(self, ticker, period='d', total_days=90):
        file_path = f'../Commodities/{ticker}/{ticker}.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()

        if period == 'd':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=total_days, freq='D')
        elif period == 'w':
            num_weeks = total_days // 7
            future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1), periods=num_weeks, freq='W')
        elif period == 'm':
            num_months = total_days // 30
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=num_months, freq='ME')
        else:
            raise ValueError("Period must be 'd', 'w', or 'm'")

        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_year'] = future_df['date'].apply(self.convert_to_day_of_year)
        X_pred = future_df['day_of_year'].values
        X_pred_reshaped = X_pred.reshape(-1, 1)
        X_pred_tf = tf.convert_to_tensor(X_pred_reshaped, dtype=tf.float64)

        return X_pred_tf
    