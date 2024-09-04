import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

train_start_date = '2024-02-10'
train_end_date = '2024-05-10'
test_start_date = '2024-05-13'
test_end_date = '2024-05-17'
file_path = f'../Stocks/AAPL/AAPL_us_d.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

df = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
# df['day_of_year'] = df['date'].apply(self.convert_to_day_of_year)
