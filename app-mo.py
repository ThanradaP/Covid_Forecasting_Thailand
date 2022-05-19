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


prophet_df = (df[['date','new_cases_rolling7','new_deaths_rolling7','full_vac_rolling7']])
prophet_df.columns = ['ds','y','add1','add2']
#st.write(prophet_df)

ALL = 'No model'
arima = "Arima"

prophet = "Facebook Prophet"

lstm = 'LSTM'

gru = 'GRU'



# loaded_model =  tf.keras.models.load_model(r'C:/Users/acer/OneDrive - King Mongkut’s University of Technology Thonburi (KMUTT)/Desktop/streamlit/my_model1.h5')

# scaler= MinMaxScaler()
# new_cases = df.new_cases.values.reshape(-1,1)
# scaled_newcases = scaler.fit_transform(new_cases)
# scaled_newcases = scaled_newcases[~np.isnan(scaled_newcases)]
# scaled_newcases = scaled_newcases.reshape(-1, 1) 

with st.sidebar:
    #Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Raw Data'):
        st.subheader("Example of raw data---------->>>")
        st.write(df.tail(5))
        st.info('Dataset is updated every day \n See all data click below')
        st.write(url1)

    genre = st.radio(
        "If you are:",
        ('The general public', 'The expert'))

    if genre == 'The general public':
        newcase_delta = '+'+str(df['new_cases'].iloc[-1])
        death_delta = '+'+str(df['new_deaths'].iloc[-1])
        st.metric(label="Today's(Last updated)",value=datetime.today().strftime("%d/%m/%Y"))
        col1,col2 = st.columns(2)
        col1.metric(label='Confirmed',value=sum(df['new_cases']), delta= newcase_delta)
        col2.metric(label='Deaths',value=sum(df['new_deaths']), delta = death_delta)

        selected_series = arima
        option= 'Univariate'
        # selected_series = arima
        periods_input = st.number_input('How many periods would you like to forecast into the future(days)?',min_value = 1, max_value = 365)
    else:
        selected_series = st.selectbox("Select a model:", (None,arima,prophet,lstm,gru))
        if selected_series == None:
            selected_series = None 
        else:
            option = st_btn_select(('Univariate', 'Multivariate'))
            # st.write(f'Selected option: {option}')
            periods_input = st.number_input('How many periods would you like to forecast into the future(days)?',min_value = 1, max_value = 365)


# with st.sidebar:
#     option = st_btn_select(('Arima', 'Arimax'))
#     st.write(f'Selected option: {option}')


if selected_series == arima:
    if option == 'Univariate':
        df.index = pd.to_datetime(df['date'])
        # trainArima = df[:int(0.8*(len(df)))]
        # trainArima.index=pd.to_datetime(trainArima['date'])
        # testArima = df[int(0.8*(len(df))):]
        # testArima.index=pd.to_datetime(testArima['date'])
        #rolling forward on testset
        # history = [x for x in trainArima['rolling7']]
        # pred_on_test = []
        # test = testArima['rolling7']
        # for t in range(len(testArima)):
        #     model = sm.tsa.arima.ARIMA(history, order=(0,2,1)) 
        #     model_fit = model.fit()
        #     output = model_fit.forecast()
        #     yhat = output[0]
        #     pred_on_test.append(yhat)
        #     obs = test[t]
        #     history.append(obs)
        # pred_on_test = pd.DataFrame(pred_on_test,index=pd.to_datetime(testArima.index),columns=['pred'])
        # timeseries_evaluation_metrics_func(testArima['rolling7'], pred_on_test['pred'])
        #Predict beyond test set
        # arima_future = sm.tsa.statespace.SARIMAX(df['rolling7'], order=(0,2,1), seasonal=False).fit()
        # forecast_future = arima_future.get_forecast(steps=periods_input, dynamic=False) 
        # pred_df = forecast_future.conf_int()
        # pred_df['pred'] = forecast_future .predicted_mean
        # pred_df.columns = ['lower', 'upper', 'pred']
        # pred_df.index = pd.date_range(start=testArima.index[-1]+timedelta(1), periods=periods_input)
        # fig2=arima_plot(trainArima,testArima,pred_df)
        # st.write(fig2)
        
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
        # trainArima = df[:int(0.8*(len(df)))]
        # trainArima.index=pd.to_datetime(trainArima['date'])
        # testArima = df[int(0.8*(len(df))):]
        # testArima.index=pd.to_datetime(testArima['date'])
        
        exog = ['new_deaths_rolling7', 'full_vac_rolling7']
        endog = ['new_cases_rolling7']
        #compute future of exogenous
        exog_future = pd.DataFrame(index= pd.date_range(df.index[-1]+timedelta(1), periods = periods_input+7))
        for column in df[exog]:
                fitted = find_orders(df[column],periods_input)
                exog_future[column]=fitted
        
        #rolling forward on testset
        # history = [x for x in trainArima['rolling7']]
        # history_exog = trainArima[exog].values.tolist()
        # pred_on_test = []
        # test = testArima['rolling7']
        # test_exog = testArima[exog].values
        # for t in range(len(testArima)):
        #     model = SARIMAX(endog=history, exog=history_exog, order=(2,0,1)) 
        #     model_fit = model.fit()
        #     output = model_fit.forecast(1, exog=test_exog[t])
        #     yhat = output[0]
        #     pred_on_test.append(yhat)
        #     obs = test[t]
        #     obs_exog = test_exog[t]
        #     history.append(obs)
        #     history_exog.append(obs_exog)
        # pred_on_test = pd.DataFrame(pred_on_test,index=pd.to_datetime(testArima.index),columns=['pred'])
        # timeseries_evaluation_metrics_func(testArima['rolling7'], pred_on_test['pred'])
        #Predict beyond test set
        # arimax_future = SARIMAX(endog=df[endog],exog=df[exog], order=(2,1,0)).fit()
        # forecast_future = arimax_future.get_forecast(steps = periods_input, exog=exog_future)
        # pred_df = forecast_future.conf_int(alpha=0.05)
        # pred_df['pred'] = forecast_future.predicted_mean
        # pred_df.columns = ['lower', 'upper', 'pred']
        # pred_df.index = pd.date_range(start=testArima.index[-1]+timedelta(1), periods=periods_input)
        # get_arima_plot(trainArima['rolling7'],testArima['rolling7'], predictions_arima, pred_df['pred'], pred_df['upper'],pred_df['lower'])
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
        # trainProphet = prophet_df[:int(0.8*(len(prophet_df)))]
        # testProphet = prophet_df[int(0.8*(len(prophet_df))):]
            
        #define dataframe for each exogenous variables
        add1_df = multi_prophet_df[['ds','add1']].rename({'add1' : 'y'}, axis=1)
        add2_df = multi_prophet_df[['ds','add2']].rename({'add2' : 'y'}, axis=1)

        #create model to forecast future value of each exogenous variables
        model_add1 = Prophet()
        model_add1.fit(add1_df)
        future_add1 = model_add1.make_future_dataframe(periods=periods_input+7,freq='D',include_history=False)
        forecast_add1 = model_add1.predict(future_add1)

        model_add2 = Prophet()
        model_add2.fit(add2_df)
        future_add2 = model_add2.make_future_dataframe(periods=periods_input+7,freq='D',include_history=False)
        forecast_add2 = model_add2.predict(future_add2)

        #combine all future of each exogenous variables
        future_regressor = pd.concat([forecast_add1['ds'],forecast_add1['yhat'], forecast_add2['yhat']], axis=1)
        future_regressor.columns= ['ds','add1','add2']
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
            # prophet_model.add_regressor('add3')
            prophet_model.fit(data_to_fit)
            # do one step prediction
            prophet_forecast = prophet_model.make_future_dataframe(periods=7, freq="d", include_history=False)
            prophet_forecast['add1'] = future_regressor['add1'][future_regressor.index[:7]]
            prophet_forecast['add2'] = future_regressor['add2'][future_regressor.index[:7]]
            # prophet_forecast['add3'] = future_regressor['add3'][future_regressor.index[:7]]
            prophet_forecast = prophet_model.predict(prophet_forecast)

            #update data to forecast future
            data_to_fit = data_to_fit.append({'ds':prophet_forecast['ds'].iloc[0],
                                        'add1': future_regressor['add1'][future_regressor.index[i]],
                                        'add2': future_regressor['add2'][future_regressor.index[i]],
                                        # 'add3': future_regressor['add3'][future_regressor.index[i]],
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
        future_index = pd.date_range(start=test.index[-1]+timedelta(1), periods=periods_input)
        predict_future = pd.DataFrame(index=future_index)
        predict_future['pred'] = lst_output 
        st.write('df', df)
        st.write('predict_future', predict_future)
        fig= lstm_plot(df, predict_future)
        st.write(fig)
        if st.checkbox('Show result in table'):
            st.markdown("Table: The cumulative number of new infections in the next 7 days")
            st.table(predict_future)

    if option == 'Multivariate':
        index_future = pd.date_range(pd.to_datetime(df.date.iloc[-1])+timedelta(1), periods=periods_input)
        future_rolling7 = future_exog('rolling7',df,periods_input)
        future_new_cases = future_exog('new_cases',df,periods_input)
        future_new_deaths = future_exog('new_deaths',df,periods_input)
        future_full_vac = future_exog('full_vac',df,periods_input)
        future_resident_gg = future_exog('resident_gg',df,periods_input)
        # future_date = pd.date_range(pd.to_datetime(df.date.iloc[-1])+timedelta(1), periods=periods_input)
        future_df = pd.DataFrame() #index = future_date
        future_df.index=index_future
        future_df['new_cases'] = future_new_cases
        future_df['new_deaths'] = future_new_deaths
        future_df['full_vac'] = future_full_vac
        future_df['resident_gg'] = future_resident_gg
        future_df['rolling7'] = future_rolling7
        st.write('future_df',future_df)
        
        y_pred_inv,lst_output= lstm_multi_model(df[['new_cases','new_deaths','full_vac','resident_gg','rolling7']],periods_input,future_df)
        # st.write('y_pred_inv',y_pred_inv)
        # st.write('lst_output',lst_output)
        train,test = split(df)
        train.index=pd.to_datetime(train['date'])
        test.index=pd.to_datetime(test['date'])
        # validate on test df
        predict_on_test = pd.DataFrame(index=pd.to_datetime(test.date.iloc[-(len(y_pred_inv)):])) 
        predict_on_test['pred'] = y_pred_inv
        st.write('predict_on_test', predict_on_test)
        #future_df
        future_index = pd.date_range(start=predict_on_test.index[-1]+timedelta(1), periods=periods_input)
        predict_future = pd.DataFrame(index=future_index)
        predict_future['pred'] = lst_output 
        timeseries_evaluation_metrics_func(test['rolling7'][-len(predict_on_test):], predict_on_test['pred'])
        st.write('predict_future', predict_future)
        fig= lstm_plot(train, test, predict_on_test, predict_future)
        st.write(fig)
        
    
# if selected_series == gru_uni:
#     y_pred_inv,lst_output= lstm_uni_model(df['rolling7'],n_periods=periods_input,model_type=simple_gru)
#     train,test = split(df)
#     train.index=pd.to_datetime(train['date'])
#     test.index=pd.to_datetime(test['date'])

#     #validate on test df
#     predict_on_test = pd.DataFrame(index=pd.to_datetime(test.date.iloc[-(len(y_pred_inv)):])) 
#     predict_on_test['pred'] = y_pred_inv
#     # st.write('predict_on_test', predict_on_test)
#     #future_df
#     future_index = pd.date_range(start=predict_on_test.index[-1]+timedelta(1), periods=periods_input)
#     predict_future = pd.DataFrame(index=future_index)
#     predict_future['pred'] = lst_output 
#     timeseries_evaluation_metrics_func(test['rolling7'][-len(predict_on_test):], predict_on_test['pred'])
#     # st.write('predict_future', predict_future)
#     fig= lstm_plot(train, test, predict_on_test, predict_future)
#     st.write(fig)

if selected_series == None:
    newcase_delta = '+'+str(df['new_cases'].iloc[-1])
    death_delta = '+'+str(df['new_deaths'].iloc[-1])
    st.metric(label="Today's(Last updated)",value=datetime.today().strftime("%d/%m/%Y"))
    col1,col2 = st.columns(2)
    col1.metric(label='Confirmed',value=sum(df['new_cases']), delta= newcase_delta)
    col2.metric(label='Deaths',value=sum(df['new_deaths']), delta = death_delta)
    cases_deaths_plot(df)
    exog_plot(df)
    
    
    
