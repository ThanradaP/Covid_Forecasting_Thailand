# -*- coding: utf-8 -*-

from nbformat import write
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
from st_btn_select import st_btn_select
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
from my_useful_func import*


#Add sidebar to the app
st.sidebar.markdown("### Covid forecasting in Thailand")

#Add title and subtitle to the main interface of the app
st.title("Covid forecasting in Thailand")
st.write("The app is part of the Senior Project on forecasting the cumulative number of infections in Thailand for the next 7 days with statistical models and machine learning models.")
st.write("The variable we predicted was the cumulative number of new infections in the next 7 days.")
st.write("Each of the data for the 7-day cumulative number of infections predicted means that in the next 7 days there will be X more infections.")
#Import cleaned dataset
url1 = 'https://drive.google.com/file/d/1AHFFtoz9tWoNDnbGc-Ssa4JGC-7LKDYD/view?usp=sharing'
url2='https://drive.google.com/uc?id=' + url1.split('/')[-2]
df = pd.read_csv(url2,index_col=[0])


prophet_df = (df[['date','new_cases_rolling7','new_deaths_rolling7','full_vac_rolling7','resident_gg']])
prophet_df.columns = ['ds','y','add1','add2','add3']

ALL = 'No model'
arima = "Arima"

prophet = "Facebook Prophet"

lstm = 'LSTM'

gru = 'GRU'


with st.sidebar:
    if st.checkbox('Show Raw Data'):
        st.subheader("Example of raw data---------->>>")
        st.write(df.tail(5))
        st.info('Dataset is updated every day \n See all data click below')
        st.write(url1)

    genre = st.radio(
        "Select mode:",
        ('Normal', 'Expert'))

    if genre == 'Normal':
        newcase_delta = '+'+str(df['new_cases'].iloc[-1])
        death_delta = '+'+str(df['new_deaths'].iloc[-1])
        st.metric(label="Today's(Last updated)",value=datetime.today().strftime("%d/%m/%Y"))
        col1,col2 = st.columns(2)
        col1.metric(label='Confirmed',value=sum(df['new_cases']), delta= newcase_delta)
        col2.metric(label='Deaths',value=sum(df['new_deaths']), delta = death_delta)

        selected_series = arima
        option= 'Univariate'
        periods_input = st.number_input('How many periods would you like to forecast into the future(days)?',min_value = 1, max_value = 365)
    else:
        selected_series = st.selectbox("Select a model:", (None,arima,prophet,lstm,gru))
        if selected_series == None:
            selected_series = None 
        else:
            option = st_btn_select(('Univariate', 'Multivariate'))
            periods_input = st.number_input('How many periods would you like to forecast into the future(days)?',min_value = 1, max_value = 365)

if selected_series == arima:
    if option == 'Univariate':
        df.index = pd.to_datetime(df['date'])
        predictions = []
        upper_ci = []
        lower_ci = []
        history = [x for x in df['new_cases_rolling7']]

        for t in range(periods_input): 
            model = sm.tsa.arima.ARIMA(history, order=(2,2,1)) 
            model_fit = model.fit()
            output = model_fit.get_forecast(7)
            pred_df = output.conf_int()
            yhat = output.predicted_mean
            predictions.append(yhat[6])
            upper_ci.append(pred_df[6][0])
            lower_ci.append(pred_df[6][1])
            history.append(yhat[0])
        pred_df= pd.DataFrame(predictions,columns =['pred'], index=pd.date_range(start=df.index[-1]+timedelta(1), periods=periods_input, freq="D"))
        pred_df['upper'] = upper_ci
        pred_df['lower'] = lower_ci
        fig2=arima_plot(df,pred_df)
        st.write(fig2)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(pred_df)
    


    if option == 'Multivariate':
        df.index = pd.to_datetime(df['date'])
        exog = ['new_deaths_rolling7', 'full_vac_rolling7']
        endog = ['new_cases_rolling7']
        #compute future of exogenous
        exog_future = pd.DataFrame(index= pd.date_range(df.index[-1]+timedelta(1), periods = periods_input+7))
        for column in df[exog]:
                fitted = find_orders(df[column],periods_input)
                exog_future[column]=fitted
      
        predictions = []
        upper_ci = []
        lower_ci = []
        history = [x for x in df['new_cases_rolling7']]
        history_exog = df[exog].fillna(0).values.tolist()
        future_exog =exog_future[exog].values.tolist()

        for t in range(periods_input): 
            model = SARIMAX(endog=history,exog=history_exog, order=(2,1,0),seasonal=False) 
            model_fit = model.fit()
            output = model_fit.get_forecast(7,exog=future_exog[:7])
            pred_df = output.conf_int()
            yhat = output.predicted_mean
            predictions.append(yhat[6])
            upper_ci.append(pred_df[6][0])
            lower_ci.append(pred_df[6][1])
            history.append(yhat.tolist()[0])
            history_exog.append(future_exog[t])
            future_exog[:1]

        pred_df= pd.DataFrame(predictions,columns =['pred'], index=pd.date_range(start=df.index[-1]+timedelta(1), periods=periods_input, freq="D"))
        pred_df['upper'] = upper_ci
        pred_df['lower'] = lower_ci
        fig2=arima_plot(df,pred_df)
        st.write(fig2)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(pred_df)
    

if selected_series == prophet:
    if option == 'Univariate':
        uni_prophet_df = prophet_df[['ds','y']].copy()
        uni_prophet_df.set_index(pd.to_datetime(uni_prophet_df.ds))
        trainfb = uni_prophet_df[:int(0.8*(len(uni_prophet_df)))]
        testfb = uni_prophet_df[int(0.8*(len(uni_prophet_df))):]

        start = pd.to_datetime(pd.to_datetime(uni_prophet_df['ds'].values[-1])+ timedelta(1))
        multi_date = pd.date_range(start=start,periods=periods_input)

        data_to_fit=uni_prophet_df.copy()
        pred_future = []
        pred_future_upper = []
        pred_future_lower=[]
        for i in range(periods_input):
            data_to_fit = data_to_fit[: (len(uni_prophet_df) + i)]
            prophet_model = Prophet(changepoint_prior_scale=0.5,
                                    changepoint_range= 0.9, 
                                    seasonality_mode='additive',
                                    seasonality_prior_scale=10.0,interval_width=0.95)
            prophet_model.fit(data_to_fit)

            # do one step prediction
            prophet_forecast = prophet_model.make_future_dataframe(periods=7, freq="d", include_history=False)
            prophet_forecast = prophet_model.predict(prophet_forecast)
            print(prophet_forecast[['ds','yhat']])

            #update data to forecast future
            data_to_fit = data_to_fit.append({'ds':prophet_forecast['ds'].iloc[0],
                                        'y':prophet_forecast["yhat"].iloc[0]}, ignore_index=True)
            
            #collect result of future forecasting
            pred_future.append(prophet_forecast["yhat"].iloc[-1])
            pred_future_upper.append(prophet_forecast["yhat_upper"].iloc[-1])
            pred_future_lower.append(prophet_forecast["yhat_lower"].iloc[-1])

        pred_df = pd.DataFrame(index=multi_date)
        pred_df['yhat'] = pred_future
        pred_df['yhat_upper'] = pred_future_upper
        pred_df['yhat_lower'] = pred_future_lower

        fig = prophet_plot(uni_prophet_df,pred_df)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(pred_df)

    if option == 'Multivariate':
        multi_prophet_df = prophet_df.copy()
        multi_prophet_df.set_index(pd.to_datetime(multi_prophet_df.ds))
        #define dataframe for each exogenous variables
        add1_df = multi_prophet_df[['ds','add1']].rename({'add1' : 'y'}, axis=1)
        add2_df = multi_prophet_df[['ds','add2']].rename({'add2' : 'y'}, axis=1)
        add3_df = multi_prophet_df[['ds','add3']].rename({'add3' : 'y'}, axis=1)

        #create model to forecast future value of each exogenous variables
        model_add1 = Prophet()
        model_add1.fit(add1_df)
        future_add1 = model_add1.make_future_dataframe(periods=periods_input+7,freq='D',include_history=False)
        forecast_add1 = model_add1.predict(future_add1)

        model_add2 = Prophet()
        model_add2.fit(add2_df)
        future_add2 = model_add2.make_future_dataframe(periods=periods_input+7,freq='D',include_history=False)
        forecast_add2 = model_add2.predict(future_add2)
        
        model_add3 = Prophet()
        model_add3.fit(add3_df)
        future_add3 = model_add3.make_future_dataframe(periods=periods_input+7,freq='D',include_history=False)
        forecast_add3 = model_add3.predict(future_add3)

        #combine all future of each exogenous variables
        future_regressor = pd.concat([forecast_add1['ds'],forecast_add1['yhat'], forecast_add2['yhat'], forecast_add3['yhat']], axis=1)
        future_regressor.columns= ['ds','add1','add2','add3']
        #create date of future predictions
        start = pd.to_datetime(pd.to_datetime(multi_prophet_df['ds'].values[-1])+ timedelta(1))
        multi_date = pd.date_range(start=start,periods=periods_input)
        #fit model to data 
        data_to_fit=multi_prophet_df
        predict_future = pd.DataFrame(columns = ['ds','yhat','yhat_upper','yhat_lower'])
        for i in range(periods_input):
            data_to_fit = data_to_fit[: (len(multi_prophet_df) + i)]
            prophet_model = Prophet(changepoint_prior_scale= 5,
                                    changepoint_range=0.9,
                                    n_changepoints=20, 
                                    seasonality_mode= 'multiplicative',
                                    seasonality_prior_scale=0.1,
                                    interval_width=0.95)
            prophet_model.add_regressor('add1')
            prophet_model.add_regressor('add2')
            prophet_model.add_regressor('add3')
            prophet_model.fit(data_to_fit)
            # do one step prediction
            prophet_forecast = prophet_model.make_future_dataframe(periods=7, freq="d", include_history=False)
            prophet_forecast['add1'] = future_regressor['add1'][future_regressor.index[:7]]
            prophet_forecast['add2'] = future_regressor['add2'][future_regressor.index[:7]]
            prophet_forecast['add3'] = future_regressor['add3'][future_regressor.index[:7]]
            prophet_forecast = prophet_model.predict(prophet_forecast)

            #update data to forecast future
            data_to_fit = data_to_fit.append({'ds':prophet_forecast['ds'].iloc[0],
                                        'add1': future_regressor['add1'][future_regressor.index[i]],
                                        'add2': future_regressor['add2'][future_regressor.index[i]],
                                        'add3': future_regressor['add3'][future_regressor.index[i]],
                                        'y':prophet_forecast["yhat"].iloc[0]}, ignore_index=True)
            
            predict_future = predict_future.append({'ds':multi_date[i],
                                            'yhat': prophet_forecast['yhat'].iloc[-1],
                                            'yhat_upper':prophet_forecast["yhat_upper"].iloc[-1],
                                            'yhat_lower':prophet_forecast["yhat_lower"].iloc[-1]},
                                            ignore_index=True)
        predict_future.index = predict_future.ds
        fig = prophet_plot(multi_prophet_df,predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)
    
if selected_series == lstm:
    if option == 'Univariate':
        lst_output= lstm_uni_model(df['new_cases'],n_periods=periods_input,model_type=simple_lstm)
        train,test = split(df)
        train.index=pd.to_datetime(train['date'])
        test.index=pd.to_datetime(test['date'])
        predict_future = pd.DataFrame(lst_output,columns =['pred'], index=pd.date_range(start=test.index[-1]+timedelta(1), periods=periods_input, freq="D"))
        fig= lstm_plot(df, predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)

    if option == 'Multivariate':
        index_future = pd.date_range(pd.to_datetime(df.date.iloc[-1])+timedelta(1), periods=periods_input+1)
        future_new_cases = future_exog(col='new_cases',df=df,n_periods=periods_input)
        future_new_deaths = future_exog(col='new_deaths',df=df,n_periods=periods_input)
        future_full_vac = future_exog(col='full_vac',df=df,n_periods=periods_input)
        future_resident_gg = future_exog(col='resident_gg',df=df,n_periods=periods_input)
        future_df = pd.DataFrame(index=index_future) 
        future_df['new_deaths'] = future_new_deaths
        future_df['full_vac'] = future_full_vac
        future_df['resident_gg'] = future_resident_gg
        future_df['new_cases'] = future_new_cases

        lst_output_inv= lstm_multi_model(df,periods_input,future_df)
        new_cases_sum7 = []
        for i in range(periods_input):
            temp = lst_output_inv[i].sum()
            new_cases_sum7.append(temp)
        predict_future = pd.DataFrame(index=index_future[:-1])
        predict_future['pred'] = new_cases_sum7 
        fig= lstm_plot(df, predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)
        
    
if selected_series == gru:
    if option == 'Univariate':
        lst_output= lstm_uni_model(df['new_cases'],n_periods=periods_input,model_type=simple_gru)
        train,test = split(df)
        train.index=pd.to_datetime(train['date'])
        test.index=pd.to_datetime(test['date'])
        predict_future = pd.DataFrame(lst_output,columns =['pred'], index=pd.date_range(start=test.index[-1]+timedelta(1), periods=periods_input, freq="D"))
        fig= lstm_plot(df, predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)
    if option == 'Multivariate':
        index_future = pd.date_range(pd.to_datetime(df.date.iloc[-1])+timedelta(1), periods=periods_input+1)
        future_new_cases = future_exog(col='new_cases',df=df,n_periods=periods_input)
        future_new_deaths = future_exog(col='new_deaths',df=df,n_periods=periods_input)
        future_full_vac = future_exog(col='full_vac',df=df,n_periods=periods_input)
        future_resident_gg = future_exog(col='resident_gg',df=df,n_periods=periods_input)
        future_df = pd.DataFrame(index=index_future) 
        future_df['new_deaths'] = future_new_deaths
        future_df['full_vac'] = future_full_vac
        future_df['resident_gg'] = future_resident_gg
        future_df['new_cases'] = future_new_cases

        lst_output_inv= gru_multi_model(df,periods_input,future_df)
        new_cases_sum7 = []
        for i in range(periods_input):
            temp = lst_output_inv[i].sum()
            new_cases_sum7.append(temp)
        predict_future = pd.DataFrame(index=index_future[:-1])
        predict_future['pred'] = new_cases_sum7 
        fig= lstm_plot(df, predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)


if selected_series == None:
    newcase_delta = '+'+str(df['new_cases'].iloc[-1])
    death_delta = '+'+str(df['new_deaths'].iloc[-1])
    st.metric(label="Today's(Last updated)",value=datetime.today().strftime("%d/%m/%Y"))
    col1,col2 = st.columns(2)
    col1.metric(label='Confirmed',value=sum(df['new_cases']), delta= newcase_delta)
    col2.metric(label='Deaths',value=sum(df['new_deaths']), delta = death_delta)
    cases_deaths_plot(df)
    exog_plot(df)
    
    
    
