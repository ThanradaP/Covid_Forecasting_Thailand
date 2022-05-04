# -*- coding: utf-8 -*-
"""
Source code

https://towardsdatascience.com/deploying-a-prophet-forecasting-model-with-streamlit-to-heroku-caf1729bd917
"""
from cProfile import label
from tkinter import image_names
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras import activations
import plotly
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
tf.random.set_seed(123)
np.random.seed(123)
import warnings
warnings.filterwarnings('ignore')
import json
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import itertools
from fbprophet.diagnostics import cross_validation
from fbprophet.serialize import model_to_json, \
model_from_json
with open('Univariate_prophet_with_rolling_sum_model.json', 'r') as file_in:
    uni_fb_model = model_from_json(json.load(file_in))

from my_useful_func import*


#Add sidebar to the app
st.sidebar.markdown("### My first Awesome App")
st.sidebar.markdown("Welcome to my first awesome app. This app is built using Streamlit and uses data source from redfin housing market data. I hope you enjoy!")

#Add title and subtitle to the main interface of the app
st.title("Covid forecasting in Thailand")
st.markdown("This web app was created to predict the trend of the spread of COVID-19 through the daily number of new cases in Thailand.")

#Import cleaned dataset
df = pd.read_csv('covid_data_cleaned.csv')

prophet_df = (df[['date','rolling7','new_cases','new_deaths','full_vac','resident_gg']])
prophet_df.columns = ['ds','y','add1','add2','add3','add4']
#st.write(prophet_df)

ALL = 'No model'
arima = "Autoregressive integrated moving average"
arimax = "Autoregressive integrated moving average with exogenous"
prophet_uni = "Univariate Facebook Prophet"
prophet_multi = "Multivariate Facebook Prophet"
lstm_uni = "Univariate lstm"
lstm_multi = "Multivariate lstm"


# loaded_model =  tf.keras.models.load_model(r'C:/Users/acer/OneDrive - King Mongkut’s University of Technology Thonburi (KMUTT)/Desktop/streamlit/my_model1.h5')

# scaler= MinMaxScaler()
# new_cases = df.new_cases.values.reshape(-1,1)
# scaled_newcases = scaler.fit_transform(new_cases)
# scaled_newcases = scaled_newcases[~np.isnan(scaled_newcases)]
# scaled_newcases = scaled_newcases.reshape(-1, 1) 

with st.sidebar:
    #Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
        st.subheader("Showing raw data---->>>")
        st.write(df.tail(6))
        st.info('Dataset is updated every day')
        
    selected_series = st.selectbox("Select a model:", (None,arima,arimax,prophet_uni,prophet_multi,lstm_uni,lstm_multi))
    periods_input = st.number_input('How many periods would you like to forecast into the future?',min_value = 1, max_value = 365)

if selected_series == arima:
    trainArima = df[:int(0.8*(len(df)))]
    trainArima.index=pd.to_datetime(trainArima['date'])
    testArima = df[int(0.8*(len(df))):]
    testArima.index=pd.to_datetime(testArima['date'])
    #rolling forward on testset
    history = [x for x in trainArima['rolling7']]
    pred_on_test = []
    test = testArima['rolling7']
    for t in range(len(testArima)):
        model = sm.tsa.arima.ARIMA(history, order=(1,1,4)) 
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        pred_on_test.append(yhat)
        obs = test[t]
        history.append(obs)
    pred_on_test = pd.DataFrame(pred_on_test,index=pd.to_datetime(testArima.index),columns=['pred'])
    timeseries_evaluation_metrics_func(testArima['rolling7'], pred_on_test['pred'])
    #Predict beyond test set
    arima_future = sm.tsa.statespace.SARIMAX(df['rolling7'], order=(1,1,4), seasonal=False).fit()
    forecast_future = arima_future.get_forecast(steps=periods_input, dynamic=False) 
    pred_df = forecast_future.conf_int()
    pred_df['pred'] = forecast_future .predicted_mean
    pred_df.columns = ['lower', 'upper', 'pred']
    pred_df.index = pd.date_range(start=testArima.index[-1]+timedelta(1), periods=periods_input)
    fig2=arima_plot(trainArima,testArima,pred_on_test,pred_df)
    st.write(fig2)

if selected_series == arimax:
    trainArima = df[:int(0.8*(len(df)))]
    trainArima.index=pd.to_datetime(trainArima['date'])
    testArima = df[int(0.8*(len(df))):]
    testArima.index=pd.to_datetime(testArima['date'])
    st.write(type(trainArima.index))
    st.write('train shape',trainArima.shape)
    st.write('test shape',testArima.shape)
    #rolling forward on testset
    history = [x for x in trainArima['rolling7']]
    history_exog = [x for x in trainArima.new_cases]
    pred_on_test = []
    test = testArima['rolling7']
    test_exog = [x for x in testArima.new_cases]
    for t in range(len(testArima)):
        model = SARIMAX(endog=history, exog=history_exog, order=(1,1,5)) 
        model_fit = model.fit()
        output = model_fit.forecast(1, exog=test_exog[t])
        yhat = output[0]
        pred_on_test.append(yhat)
        obs = test[t]
        obs_exog = test_exog[t]
        history.append(obs)
        history_exog.append(obs_exog)
    pred_on_test = pd.DataFrame(pred_on_test,index=pd.to_datetime(testArima.index),columns=['pred'])
    timeseries_evaluation_metrics_func(testArima['rolling7'], pred_on_test['pred'])
    #Predict beyond test set
    

if selected_series == prophet_uni:
    uni_prophet_df = prophet_df[['ds','y']].copy()
    uni_prophet_df.set_index(pd.to_datetime(uni_prophet_df.ds))
    trainfb = uni_prophet_df[:int(0.8*(len(uni_prophet_df)))]
    testfb = uni_prophet_df[int(0.8*(len(uni_prophet_df))):]
    # walk validation on test set
    start = pd.to_datetime(testfb['ds'].values[0]) - timedelta(1) # create cutoff by each testfb index
    cutoffs = pd.date_range(start, periods=len(testfb))
    cutoffs = pd.to_datetime(cutoffs)
    m = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10.0)
    m.fit(uni_prophet_df)
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='1 days')
    df_cv.yhat[df_cv.yhat < 0] = 0 #deal with negative value becases cases cannot be negative
    df_p = performance_metrics(df_cv)
    timeseries_evaluation_metrics_func(df_cv['y'],df_cv['yhat'])

    #Predict beyond test set
    uni_fb_tuned_future = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10.0)
    uni_fb_tuned_future.fit(uni_prophet_df)
    future2 = uni_fb_tuned_future.make_future_dataframe(periods=periods_input, freq='D', include_history=False)
    forecast_future = uni_fb_tuned_future.predict(future2)
    forecast_future.yhat[forecast_future.yhat < 0] = 0
    fig = prophet_plot(trainfb,testfb,df_cv,forecast_future)
    st.write(fig)

if selected_series == prophet_multi:
    trainProphet = prophet_df[:int(0.8*(len(prophet_df)))]
    testProphet = prophet_df[int(0.8*(len(prophet_df))):]
    multi_fb_model = Prophet(changepoint_prior_scale=0.005, changepoint_range=0.8, seasonality_prior_scale=0.1, holidays_prior_scale=10.0)
    multi_fb_model.add_regressor('add1')
    multi_fb_model.add_regressor('add2')
    multi_fb_model.add_regressor('add3')
    multi_fb_model.add_regressor('add4')
    multi_fb_model = multi_fb_model.fit(trainProphet)
    future = multi_fb_model.make_future_dataframe(periods= len(testProphet),freq='D',include_history=False)
    future['add1'] = testProphet['add1'].values
    future['add2'] = testProphet['add2'].values
    future['add3'] = testProphet['add3'].values
    future['add4'] = testProphet['add4'].values
    forecast = multi_fb_model.predict(future)
    st.write(prophet_df)
    st.write(forecast)
    timeseries_evaluation_metrics_func(testProphet['y'],forecast['yhat'])
##########################################################################
    m = Prophet(changepoint_prior_scale= 0.7,n_changepoints=5,changepoint_range=0.9, seasonality_prior_scale=1)
    m.add_regressor('add1')
    m.add_regressor('add2')
    m.add_regressor('add3')
    m.add_regressor('add4')
    m.fit(prophet_df)
    future2 = m.make_future_dataframe(periods=periods_input, freq='D')
    forecast_future = m.predict(future2)
    st.write(forecast_future['yhat'][-periods_input:])
############################################################################

    
if selected_series == lstm_uni:
    # loaded_model.summary()
    # train_generator,test_generator = train_test_generator(df)
    # st.write(train_generator[0])
    # st.write(test_generator[0])
    # predictions=loaded_model.predict(test_generator)
    # loaded_model.evaluate(test_generator, verbose=0)
    
    history,predicts,actuals = lstm_uni_model(df['new_cases'],df['rolling7'])
    res = pd.DataFrame(predicts,columns=['predicts'],index=actuals.index)
    res['actuals'] = actuals
    timeseries_evaluation_metrics_func(res['actuals'], res['predicts'])
    fig = st.line_chart(res)
    st.write(fig)

if selected_series == lstm_multi:
    history,predicts,actuals = lstm_multi_model(df[['new_cases','new_vac']],df['rolling7'])
    res = pd.DataFrame(predicts,columns=['predicts'],index=actuals.index)
    res['actuals'] = actuals
    timeseries_evaluation_metrics_func(res['actuals'], res['predicts'])
    fig = st.line_chart(res)
    st.write(fig)

if selected_series == None:
    newcase_delta = '+'+str(df['new_cases'].iloc[-1])
    death_delta = '+'+str(df['new_deaths'].iloc[-1])
    st.metric(label="Last updated date",value=datetime.today().strftime("%d/%m/%Y"))
    col1,col2 = st.columns(2)
    col1.metric(label='Confirmed',value=sum(df['new_cases']), delta= newcase_delta)
    col2.metric(label='Deaths',value=sum(df['new_deaths']), delta = death_delta)
    cases_deaths_plot(df)
    exog_plot(df)
    
    
    
