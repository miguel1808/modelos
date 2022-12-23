import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_datareader as datas
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import keras
import os
import pandas as pd
import numpy as np
import math
import datetime as dtr
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
#from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def app():
    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))

    st.title('Predicción de precios de acciones usando GRU')

    user_input = st.text_input("Introducir cotización bursátil", 'AMZN')
    dfi = yf.download(user_input, start, end)
    dfi.index=dfi.index.strftime('%Y-%m-%d')
    dfi.reset_index(inplace=True)
    #escribir un poco acerca de la empresa introducida en user_input
    # con la libreria de pandas_datareader podemos obtener informacion de la empresa
    
    st.subheader('Acerca de la empresa')
    st.write(datas.get_quote_yahoo(user_input))
    st.subheader('Datos de la empresa')
    st.write(dfi)

    #separar los datos en train y test
    df= dfi.iloc[:int(dfi.shape[0]*0.8),:]
    test_df = dfi.iloc[int(dfi.shape[0]*0.8):,:]


    #ordenar por fecha

    df = df.sort_values('Date')
    st.subheader('Serie de tiempo de la cotización bursátil de '+user_input)
    fig = px.line(df, x="Date", y="Close", title='Precio de cierre de '+ user_input)
    st.plotly_chart(fig)

    #Comparación de tendencias entre precio de apertura de acciones, precio de cierre, precio alto, precio bajo

    st.subheader('Comparación de tendencias entre precio de apertura de acciones, precio de cierre, precio alto, precio bajo')
    names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])
    fig2 = px.line(dfi, x=dfi.Date, y=[dfi['Open'], dfi['Close'], dfi['High'], dfi['Low']],
            labels={'date': 'Date','value':'Stock value'})
    fig2.update_layout(title_text='Stock analysis chart', font_size=15,legend_title_text='Stock Parameters')
    fig2.for_each_trace(lambda t:  t.update(name = next(names)))
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2)


    #Hacer un marco de datos separado del precio de cierre
    closedf =dfi[['Date','Close']]
    fig3= px.line(closedf, x=closedf.Date, y=closedf.Close, labels={'date':'Date','close':'Close Stock'})
    fig3.update_traces(marker_line_width=2, opacity=0.8)
    fig3.update_layout(title_text='Stock Close Price', font_size=15,legend_title_text='Stock Parameters')
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)
    st.subheader("Trazado del gráfico de precios de cierre de acciones")
    st.plotly_chart(fig3)


    #Considere solo los datos del último año para la predicción
    closedf = closedf[closedf['Date'] > '2021-1-11']
    close_stock = closedf.copy()
    st.write("Período considerado para predecir el precio de cierre de las acciones")
    fig4 = px.line(closedf, x=closedf.Date, y=closedf.Close,labels={'date':'Date','close':'Close Stock'})
    fig4.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
    fig4.update_layout(title_text='Período considerado para predecir el precio de cierre de las acciones', font_size=15)
    fig4.update_xaxes(showgrid=False)
    fig4.update_yaxes(showgrid=False)
    st.plotly_chart(fig4)

    #Normalización del precio de cierre
    del closedf['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
    
    #Preparar datos para entrenar y probar
    training_size=int(len(closedf)*0.60)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    st.write("train_data: ", train_data.shape)
    st.write("test_data: ", test_data.shape)

    #Transformar la base de precios de cierre en el requisito de pronóstico de análisis de series temporales
    # convertir una matriz de valores en una matriz de conjunto de datos
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    
    # remodelar la entrada para que sea [muestras, pasos de tiempo, características] que se requiere para LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    #Construcción de modelos (GRU)
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(GRU(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(GRU(32,return_sequences=True))
    model.add(GRU(32))
    model.add(Dropout(0.20))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    #Entrenamiento del modelo
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    fig5=px.line(x=epochs, y=[loss,val_loss], labels={'x':'Epochs','y':'Loss'})
    fig5.update_layout(title_text='Loss,val_loss vs Epochs', font_size=15,legend_title_text='Loss,val_loss')
    fig5.update_xaxes(showgrid=False)
    fig5.update_yaxes(showgrid=False)
    st.subheader("Loss,val_loss vs Epochs")
    st.plotly_chart(fig5)

    ### Hagamos la predicción y verifiquemos las métricas de rendimiento
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape

    # Transformar de nuevo a la forma original

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    st.subheader(" Métricas de evaluación RMSE, MSE y MAE")
    st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    st.write("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    st.write("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    st.write("-------------------------------------------------------------------------------------")
    st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    st.write("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    st.write("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))

    #Puntuación de regresión de varianza explicada
    st.subheader("Puntuación de regresión de varianza explicada")
    st.write("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    st.write("The test data explained the variance regression score:", explained_variance_score(original_ytest, test_predict))

    #Puntuación R2 para regresión
    st.subheader("Puntuación R2 para regresión")
    st.write("Train data R2 score:", r2_score(original_ytrain, train_predict))
    st.write("Test data R2 score:", r2_score(original_ytest, test_predict))

    #Pérdida de regresión, Pérdida de regresión de desviación media gamma (MGD) y pérdida de regresión de desviación media de Poisson (MPD)
    st.subheader("Pérdida de regresión, Pérdida de regresión de desviación media gamma (MGD) y pérdida de regresión de desviación media de Poisson (MPD)")
    st.write("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
    st.write("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
    st.write("----------------------------------------------------------------------")
    st.write("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
    st.write("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))


    #Comparación del precio de cierre de la acción original y el precio de cierre previsto

    st.subheader("Comparación del precio de cierre de la acción original y el precio de cierre previsto")
    # predicciones de cambio de train para trazar

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    print("Train predicted data: ", trainPredictPlot.shape)

    # predicciones de prueba de cambio para trazar
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
    print("Test predicted data: ", testPredictPlot.shape)

    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


    plotdf = pd.DataFrame({'Date': close_stock['Date'],
                        'original_close': close_stock['Close'],
                        'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig6 = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                            plotdf['test_predicted_close']],
                labels={'value':'Stock price','date': 'Date'})
    fig6.update_layout(title_text='Comparación entre el precio de cierre original y el precio de cierre previsto',
                    plot_bgcolor='white', font_size=15, legend_title_text='Close Price')
    fig6.for_each_trace(lambda t:  t.update(name = next(names)))

    fig6.update_xaxes(showgrid=False)
    fig6.update_yaxes(showgrid=False)
    st.plotly_chart(fig6)

    st.subheader("Predicción de precios de cierre de acciones para los próximos 30 días")


    # Predicción de los próximos 30 días
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    from numpy import array

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 30
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1


    st.subheader("Trazado de los últimos 15 días del conjunto de datos y los próximos 30 días previstos")
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    # Trazado de los últimos 15 días del conjunto de datos y los próximos 30 días previstos
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle(['Precio de cierre de los últimos 15 días','Precio de cierre previsto para los próximos 30 días'])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                        new_pred_plot['next_predicted_days_value']],
                labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Compara los últimos 15 días con los próximos 30 días',
                    plot_bgcolor='white', font_size=15,legend_title_text='Close Price')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    st.subheader("Trazar todo el precio de cierre de las acciones con el próximo período de predicción de 30 días")

    lstmdf=closedf.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['Close price'])

    fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                    plot_bgcolor='white', font_size=15,legend_title_text='Stock')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)
