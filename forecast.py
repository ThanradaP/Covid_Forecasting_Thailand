import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras import activations
import plotly
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
tf.random.set_seed(123)
np.random.seed(123)

def lstm_model(df, features, target):  #preprocess(df['new_cases'],df['rolling7']])
    win_length = 7 
    batch_size = 32
    num_features = len(features)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features.values.reshape(-1,1))
    target_scaled = scaler.fit_transform(target.values.reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, 
                                                        test_size=0.1, 
                                                        random_state=123, 
                                                        shuffle=False)
    train_generator = TimeseriesGenerator(X_train, y_train, 
                                          length=win_length, 
                                          sampling_rate=1, 
                                          batch_size=batch_size)
    test_generator = TimeseriesGenerator(X_test, y_test, 
                                         length=win_length, 
                                         sampling_rate=1, 
                                         batch_size=batch_size)
    DROPOUT = 0.2 
    WINDOW_SIZE = 7
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, num_features)))
    model.add(Dropout(rate=DROPOUT))
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences = True)))
    model.add(Dropout(rate=DROPOUT))
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation('relu'))
    BATCH_SIZE = 64
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_generator, epochs=50, batch_size=BATCH_SIZE, shuffle=False, validation_data=test_generator)
    predicts_inverse = model.predict(test_generator) 
    predicts = scaler.inverse_transform(predicts_inverse)
    predicts= predicts.astype(int)
    actuals = df['rolling7'][-len(predicts):].astype(int)
    
    return history,predicts,actuals

def generate_forecast(df_, unimulti, model):
  if unimulti == 'Univariate':
    df = df_[['new_cases','rolling7']].copy()
    print(df)
    if model == 'Arima':
      print('Univariate Arima')
    elif model == 'Facebook Prophet':
      print('Univariate Facebook Prophet')
    elif model == 'LSTM':
        history,predicts,actuals = lstm_model(df,df['new_cases'],df['rolling7'])
        res = pd.DataFrame(predicts,columns=['predicts'],index=actuals.index)
        res['actuals'] = actuals
        print(res)
    elif model == 'GRU':
      print('Univariate GRU')
    
  else:
    df = df_.copy()
    print(df)
    if model == 'Arima':
      print('Multivariate Arima')
    elif model == 'Facebook Prophet':
      print('Multivariate Facebook Prophet')
    elif model == 'LSTM':
      print('Multivariate LSTM')
    elif model == 'GRU':
      print('Multivariate GRU')
  
  return df,res     

# def calculate_smape(df_, regressor, forecast_horizon, window_length):
#     df = df_.copy()
#     df.fillna(method = 'ffill', inplace = True)
#     y = df.iloc[:,0].reset_index(drop=True)
#     y_train, y_test = temporal_train_test_split(y, test_size = forecast_horizon)
#     fh = np.arange(y_test.shape[0]) + 1
#     regressor = select_regressor(regressor)
#     forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
#                                              strategy='recursive')
#     forecaster.fit(y_train, fh=fh)
#     y_pred = forecaster.predict(fh)  
#     return smape_loss(y_pred, y_test)


# def timeseries_evaluation_metrics_func(y_true, y_pred):
#   def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#   print('Evaluation metric results:-')
#   print(f'MSE is : {mean_squared_error(y_true, y_pred)}')
#   print(f'MAE is : {mean_absolute_error(y_true, y_pred)}')
#   print(f'RMSE is : {np.sqrt(mean_squared_error(y_true, y_pred))}')
#   print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
  # print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
