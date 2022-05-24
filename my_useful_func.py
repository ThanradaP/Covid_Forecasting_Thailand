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
tf.random.set_seed(123)
np.random.seed(123)
import warnings
warnings.filterwarnings('ignore')
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
    fig.add_scatter(x=exog.index, y=exog.new_deaths, row=1, col=1, name='new_deaths') 
    fig.add_scatter(x=exog.index, y=exog.full_vac, row=2, col=1,name='full_vac') 
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
    fig.add_trace(go.Scatter(x=pd.to_datetime(df.date), y=df.new_cases_rolling7,
                        mode='lines',
                        name='train'))
    fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['pred'],
                        mode='lines',
                        name='predict future'))
    fig.update_xaxes(range=[df.date.iloc[-30],predict_future.index[-1]],
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

def arima_plot(df,predict_future): 
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df.index, y=df['new_cases_rolling7'],
                      mode='lines',
                      name='Actual data'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['pred'],
                      mode='lines',
                      name='predict future'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['lower'],
                      mode='lines',
                      name='Lower bounds of the 95% confidence interval'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['upper'],
                      mode='lines',
                      name='Upper bounds of the 95% confidence interval'))
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
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat'],
                      mode='lines',
                      name='Predict future'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat_lower'],
                      mode='lines',
                      name='Lower bounds of the 95% confidence interval'))
  fig.add_trace(go.Scatter(x=predict_future.index, y=predict_future['yhat_upper'],
                      mode='lines',
                      name='Upper bounds of the 95% confidence interval'))
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
def create_dataset(data, look_back=14):
    dataX, dataY = [], []
    i_range = len(data) - look_back - 1
    print(i_range)
    for i in range(0, i_range):
        dataX.append(data[i:(i+look_back)])    # index can move down to len(dataset)-1
        dataY.append(data[i + look_back])      # Y is the item that skips look_back number of items
    
    return np.array(dataX), np.array(dataY)

def create_dataset2(dataset, time_step=14):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):  
		a = dataset[i:(i+time_step), 0]   
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def split(df):
  train = df[:int(0.8*(len(df)))]
  test = df[int(0.8*(len(df))):]
  return train,test

def get_mape(var1, var2):
    return mean_absolute_percentage_error(var1, var2)



def simple_lstm(n_steps,n_features):
    model = Sequential()
    model.add(LSTM(448, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model
def multi_simple_lstm(n_steps,n_features):
    model = Sequential()
    model.add(GRU(320, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model

def simple_gru(n_steps,n_features):
    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    return model

def multi_simple_gru(n_steps,n_features):
    model = Sequential()
    model.add(GRU(192, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.5))
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

def split_sequence2(sequence1,sequence2, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequence1)):
    # find the end of this pattern
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    # check if we are beyond the sequence
    if out_end_ix > len(sequence1):
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence1[i:end_ix], sequence2[end_ix:out_end_ix]
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
    n_steps_in, n_steps_out = 14, 7
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
    for i in range(n_periods):
        temp = lst_output_inv[i].sum()
        pred_sum7.append(temp)
    return pred_sum7

   
def future_exog(col,df,n_periods):
    n_steps=14
    scaler_exog=MinMaxScaler(feature_range=(0,1))
    scaled=scaler_exog.fit_transform(np.array(df[col]).reshape(-1,1))
    train,test=split(scaled)
    X, y = create_dataset2(scaled)
    X_train, X_test = split(X)
    y_train, y_test = split(y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(LSTM(128,activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(128))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32,verbose=0,shuffle=False)
    x_input=X_test[-1].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    x_input=np.array(temp_input)

    lst_output=[]
    i=0
    while(i<n_periods+1):
        
        if(len(temp_input)>n_steps):
            temp_input = temp_input[1:]
            x_input=np.array(temp_input)
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    lst_output_scaled = scaler_exog.inverse_transform(lst_output)

    return lst_output_scaled


def lstm_multi_model(df,n_periods,future_df): 
    n_epochs=50
    n_features = 4
    n_steps = 14
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    dataX = x_scaler.fit_transform(df[['new_deaths','full_vac', 'resident_gg','new_cases']])
    dataY = y_scaler.fit_transform(df[['new_cases']])
    future_df_scaled = x_scaler.transform(future_df.values)
    future_input = np.array(future_df_scaled)
    X,y =split_sequence2(dataX,dataY,14,7)
    X_train, X_test = split(X)
    y_train, y_test = split(y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
    model = multi_simple_lstm(n_steps,n_features)
    model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=n_epochs,shuffle=False, verbose=0)
    x_input = np.concatenate([X_test[-1][1:],[future_input[0]]])
    temp_input=list(x_input)
    x_input=np.array(temp_input)
    future_input = future_input[1:]
    lst_output=[]
    i=0
    while(i<n_periods):
        
        if(len(temp_input)>n_steps):
            temp_input = temp_input[1:]
            x_input=np.array(temp_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(future_input[i])
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,n_features))
            yhat = model.predict(x_input, verbose=0) 
            temp_input.append(future_input[i])
            lst_output.extend(yhat.tolist())
            i=i+1
            
    lst_output_inv = y_scaler.inverse_transform(lst_output)
    
    return lst_output_inv 

def gru_multi_model(df,n_periods,future_df): 
    n_epochs=50
    n_features = 4
    n_steps = 14
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    dataX = x_scaler.fit_transform(df[['new_deaths','full_vac', 'resident_gg','new_cases']])
    dataY = y_scaler.fit_transform(df[['new_cases']])
    future_df_scaled = x_scaler.transform(future_df.values)
    future_input = np.array(future_df_scaled)
    X,y =split_sequence2(dataX,dataY,14,7)
    X_train, X_test = split(X)
    y_train, y_test = split(y)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
    model = multi_simple_gru(n_steps,n_features)
    model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=n_epochs,shuffle=False, verbose=0)
    x_input = np.concatenate([X_test[-1][1:],[future_input[0]]])
    temp_input=list(x_input)
    x_input=np.array(temp_input)
    future_input = future_input[1:]
    lst_output=[]
    i=0
    while(i<n_periods):
        
        if(len(temp_input)>n_steps):
            temp_input = temp_input[1:]
            x_input=np.array(temp_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(future_input[i])
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,n_features))
            yhat = model.predict(x_input, verbose=0) 
            temp_input.append(future_input[i])
            lst_output.extend(yhat.tolist())
            i=i+1
            
    lst_output_inv = y_scaler.inverse_transform(lst_output)
    
    return lst_output_inv 



