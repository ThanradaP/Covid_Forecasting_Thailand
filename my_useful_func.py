# from turtle import color, fillcolor
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Activation,GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import activations
import plotly
from pmdarima import auto_arima
from datetime import date
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from fbprophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots
from sklearn import metrics
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model
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

def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    with st.expander("Evaluation metric results"):
        st.write("""
        The evaluation metric are calculated from prediction on test data and actual test data.
        """)
        col1,col2 = st.columns(2)
        col1.metric(label='RMSE',value=round(rmse,2))
        col2.metric(label='MAPE',value=round(mape,2))
        
@st.cache(suppress_st_warning=True)      
def find_orders(ts,periods_input):
    stepwise_model = auto_arima(ts.dropna(), start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q 
                      # m=7, # daily data   
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True) # this works 
    fitted = stepwise_model.predict(n_periods=periods_input+7)

    return fitted

def cases_deaths_plot(df):
        fig_newcases = go.Figure()
        fig_newcases.add_trace(go.Scatter(
            x=df['date'], y=df['new_cases'],
            hoverinfo='x+y', mode='lines', fill='tozeroy',
            fillcolor='lightblue', line_color='lightblue'
        ))
        fig_newcases.update_layout(title_text="New cases in Thailand", title_x=0.5)
        st.plotly_chart(fig_newcases)
        
        fig_newdeaths = go.Figure()
        fig_newdeaths.add_trace(go.Scatter(
            x=df['date'], y=df['new_deaths'],
            hoverinfo='x+y', mode='lines', fill= 'tozeroy',
            fillcolor='goldenrod', line_color='goldenrod'
        ))
        fig_newdeaths.update_layout(title_text="New deaths in Thailand", title_x=0.5)
        st.plotly_chart(fig_newdeaths)

def exog_plot(df):
    exog = df[['new_cases','new_deaths','full_vac','resident_gg']].copy()
    fig = make_subplots(rows=2, subplot_titles=('New deaths','People fully vacinated'))
    # fig.add_scatter(x=exog.index, y=exog.new_cases, row=1, col=1, name='new_cases') 
    fig.add_scatter(x=exog.index, y=exog.new_deaths, row=1, col=1, name='new_deaths') 
    fig.add_scatter(x=exog.index, y=exog.full_vac, row=2, col=1,name='full_vac') 
    # fig.add_scatter(x=exog.index, y=exog.resident_gg, row=4, col=1,name='resident_gg') 
    fig.update_layout(height=800, width=800,title_text='Exogenous variable plot')

    with st.expander("Exogenous variables"):
        st.write("""
        The exogenous variable is a parallel time series that are not modeled directly but is used as a weighted input to the model. 
        This project has selected 2 external variables as follows: \n
            * Deaths: Daily death \n
            * Full vac: Number of people who have completed two doses of vaccination \n
        """)
        st.plotly_chart(fig)
    

def lstm_plot(df, predict_future):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.new_cases_rolling7,
                        mode='lines',
                        name='train'))
    fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['pred'],
                        mode='lines',
                        name='predict future'))
    fig.update_xaxes(range=[df.index[-30],predict_future.index[-1]],
        rangeselector_bgcolor= '#6C6665',
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])))
    return fig

def arima_plot(df,predict_future): #predict_on_test
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df.index, y=df['new_cases_rolling7'],
                      mode='lines',
                      name='Actual data'))
#   fig.add_trace(go.Scatter(x=train.index, y=train.new_cases_rolling7,
#                       mode='lines',
#                       name='train'))
#   fig.add_trace(go.Scatter(x=test.index, y=test.new_cases_rolling7,
#                       mode='lines',
#                       name='test'))
#   fig.add_trace(go.Scatter(x=test.index, y=predict_on_test['pred'],
#                       mode='lines',
#                       name='predict on test set'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['pred'],
                      mode='lines',
                      name='predict future'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['lower'],
                      mode='lines',
                      name='Lower bounds of the 95% confidence interval'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['upper'],
                      mode='lines',
                      name='Upper bounds of the 95% confidence interval'))
#   fig.add_vrect(
#       x0=test.index[0], x1=test.index[-1],
#       fillcolor="LightPink", opacity=0.5,
#       layer="below", line_width=0,
#       )
  fig.add_vrect(
      x0=predict_future.index[0], x1=predict_future.index[-1],
      fillcolor="Yellow", opacity=0.5,
      layer="below", line_width=0,
      )
  fig.update_xaxes(range=[df.index[-30],predict_future.index[-1]],
      rangeselector_bgcolor= '#6C6665',
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
  return fig

def prophet_plot(prophet_df,predict_future):
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=prophet_df.ds, y=prophet_df['y'],
                      mode='lines',
                      name='Actual data'))
#   fig.add_trace(go.Scatter(x=train.ds, y=train.y,
#                       mode='lines',
#                       name='train'))
#   fig.add_trace(go.Scatter(x=test.ds, y=test.y,
#                       mode='lines',
#                       name='test'))
#   fig.add_trace(go.Scatter(x=test.ds, y=predict_on_test['yhat'],
#                       mode='lines',
#                       name='predict on test set'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat'],
                      mode='lines',
                      name='Predict future'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat_lower'],
                      mode='lines',
                      name='Lower bounds of the 95% confidence interval'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat_upper'],
                      mode='lines',
                      name='Upper bounds of the 95% confidence interval'))
#   fig.add_vrect(
#       x0=test.ds.iloc[0], x1=test.ds.iloc[-1],
#       fillcolor="LightPink", opacity=0.5,
#       layer="below", line_width=0,
#       )
  fig.add_vrect(
      x0=predict_future.index[0], x1=predict_future.index[-1],
      fillcolor="Yellow", opacity=0.5,
      layer="below", line_width=0,
      )
  fig.update_xaxes(range=[prophet_df.ds.iloc[-30],predict_future.index[-1]],
      rangeselector_bgcolor= '#6C6665',
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
  return fig


# convert an array of values into a dataset matrix
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    i_range = len(data) - look_back - 1
    print(i_range)
    for i in range(0, i_range):
        dataX.append(data[i:(i+look_back)])    # index can move down to len(dataset)-1
        dataY.append(data[i + look_back])      # Y is the item that skips look_back number of items
    
    return np.array(dataX), np.array(dataY)


def split(df):
  train = df[:int(0.8*(len(df)))]
  test = df[int(0.8*(len(df))):]
  return train,test

def get_mape(var1, var2):
    return mean_absolute_percentage_error(var1, var2)



def simple_lstm(n_steps,n_features):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model

def simple_gru(n_steps,n_features):
    model = Sequential()
    model.add(GRU(128, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(64))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequence)):
    # find the end of this pattern
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    # check if we are beyond the sequence
    if out_end_ix > len(sequence):
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

@st.cache(suppress_st_warning=True)
def lstm_uni_model(data,n_periods,model_type):  #preprocess(df['new_cases'],df['rolling7']])
    n_features = 1
    n_epochs=50
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.dropna().values.reshape(-1,1))
    # choose a number of time steps
    n_steps_in, n_steps_out = 14, 1
    # split into samples
    X, y = split_sequence(data_scaled, n_steps_in, n_steps_out)
    X_train, X_test = split(X)
    y_train, y_test = split(y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    params = (n_steps_in,n_features)
    model = model_type(*params)
    model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=n_epochs,shuffle=False, verbose=0)
    x_input=X_test[-1].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    x_input=np.array(temp_input)
    lst_output=[]
    i=0
    while(i<n_periods):
        
        if(len(temp_input)>n_timesteps):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_timesteps, 1))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend([yhat[0][0]])
            temp_input=temp_input[1:] 
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_timesteps,1))
            print("{} day input {}".format(i,x_input))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend([yhat[0][0]])
            lst_output.extend(yhat.tolist())
            i=i+1
    lst_output_inv = scaler.inverse_transform(lst_output)
    pred_sum7 = []
    for i in range(7):
        temp = lst_output_inv[i].sum()
        pred_sum7.append(temp)
    return pred_sum7

@st.cache(suppress_st_warning=True)    
def future_exog(col,df,n_periods):
    n_steps=14
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled=scaler.fit_transform(np.array(df[col]).reshape(-1,1))
    train,test=split(scaled)
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    X_train, y_train = create_dataset(train,n_steps)
    X_test, y_test = create_dataset(test,n_steps)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    params = (n_steps,1)
    model = simple_lstm(*params)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32,verbose=0,shuffle=False)
    lookback = len(test) - 14
    x_input=test[lookback:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    x_input=np.array(temp_input)

    lst_output=[]
    i=0
    while(i<n_periods):
        
        if(len(temp_input)>14):
            temp_input = temp_input[1:]
            print('before',type(temp_input))
            x_input=np.array(temp_input)
            print('after',type(temp_input))
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            model = simple_lstm(n_steps,1)
            history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32,verbose=0,shuffle=False)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
            print('i=',i)
        else:
            x_input = x_input.reshape((1, n_steps,1))
            model = simple_lstm(n_steps,1)
            history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32,verbose=0,shuffle=False)
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
            print('i=',i)

    lst_output_scaled = scaler.inverse_transform(lst_output)

    return lst_output_scaled


@st.cache(suppress_st_warning=True)
def lstm_multi_model(data,n_periods,future_df):  #preprocess(df['new_cases'],df['rolling7']])
    n_features = 5
    n_epochs=50
    #prepare future df
    future_scaler=MinMaxScaler()
    future_df_scaled = future_scaler.fit_transform(future_df.values)
    future_input = np.array(future_df_scaled)
    #prepare main df
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)
    train,test = split(data_scaled)
    # st.write(train,test)
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    X_train, y_train = create_dataset(train,14)
    X_test, y_test = create_dataset(test,14)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
    st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    best_rmse = np.inf
    best_mape = np.inf
    best_lstm_model=None
    n_steps = 14
    params = (n_steps,n_features)
    model = simple_lstm(*params)
    model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=n_epochs,shuffle=False, verbose=0)
    #one step prediction on test set
    y_pred=[]
    for ix in range(y_test.shape[0]): 
        x_input = X_test[ix]
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        y_pred.extend(yhat[0])
        
    y_test_inv = scaler.inverse_transform(y_test)
    y_test_inv = y_test_inv[:,-1]
    x_test = X_test.reshape(X_test.shape[0],X_test.shape[2]*X_test.shape[1])
    a = np.concatenate((x_test[:,-4:],np.array(y_pred).reshape(-1,1)),axis=1)
    a = scaler.inverse_transform(a)
    y_pred_inv = a[:,-1]

    mape = get_mape(y_test_inv, y_pred_inv)
    st.write("Simple_lstm -  MAPE = {:4.4f}".format(mape))
    
    fig = plt.figure(figsize=(16,7))
    plt.plot(y_pred_inv,label='Pred')
    plt.plot(y_test_inv,label='Actual')
    plt.legend()
    st.write(fig)
        
    #forecast future
    lookback = len(test) - 14
    x_input=test[lookback:]
    temp_input=list(x_input)
    x_input=np.array(temp_input) 
    lst_output=[]
    n_steps=14
    i=0
    while(i<n_periods):
        
        if(len(temp_input)>14):
            #print(temp_input)
            temp_input = temp_input[1:]
            print('before',type(temp_input))
            x_input=np.array(temp_input)
            print('after',type(x_input))
            print("{} day input {}".format(i,x_input))
            print('x_input',x_input)
            # x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, n_features))
            #print(x_input)
            model = simple_lstm(n_steps,n_features)
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = 50, verbose=0, shuffle=False)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.append(future_input[i])
            # temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,n_features))
            model = simple_lstm(n_steps,n_features)
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs = 50, verbose=0, shuffle=False)
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(future_input[i])
            print('i',i)
            print('len(temp_input):',len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    #inverse future prediction to normal scale
    future_temp = np.hstack((np.delete(future_df_scaled,-1,axis=1),lst_output))
    future_temp = future_scaler.inverse_transform(future_temp)
    lst_output_inv = future_temp[:,-1]
        
    return y_pred_inv,lst_output_inv



