import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as datas
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import yfinance as yf


def app():

    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones usando LSTM')

    user_input = st.text_input('Introducir cotización bursátil' , 'NFLX')
    dfi = yf.download(user_input, start, end)
    #dfi.index=dfi.index.strftime('%Y-%m-%d')
    #dfi.reset_index(inplace=True)
    #escribir un poco acerca de la empresa introducida en user_input
    # con la libreria de pandas_datareader podemos obtener informacion de la empresa
    st.subheader('Acerca de la empresa')
    st.write(datas.get_quote_yahoo(user_input))
    st.subheader('Datos de la empresa')
    st.write(dfi)

    #separar el dataframe en train y test
    df = dfi.iloc[:int(dfi.shape[0]*0.8), :]
    test_df = dfi.iloc[int(dfi.shape[0]*0.8):, :]



    # ordenar por fecha
    df = df.sort_values('Date')
    test_df = test_df.sort_values('Date')

    # fix the date 
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    test_df.reset_index(inplace=True)
    test_df.set_index("Date", inplace=True)

    #hacemos un grafico de la serie de tiempo
    st.subheader('Serie de tiempo de la cotización bursátil de '+user_input)
    fig = px.line(dfi, x=dfi.index , y="Close", title='Precio de cierre de '+ user_input)
    st.plotly_chart(fig)

    # mostrar un grafico de la media movil de la serie de tiempo
    st.subheader('Media movil de la cotización bursátil de '+user_input)
    fig = px.line(dfi, x=dfi.index , y="Close", title='Precio de cierre de '+ user_input)
    fig.add_scatter(x=dfi.index, y=dfi.Close.rolling(window=30).mean(), mode='lines', name='Media movil')
    st.plotly_chart(fig)


    import matplotlib.dates as mdates

    # cambiar las fechas en enteros para el entrenamiento
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    # Almacene las fechas originales para trazar las predicciones
    org_dates = dates_df['Date']

    # convert to ints
    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

    # Crear un conjunto de datos de entrenamiento de precios de 'Adj Close':
    train_data = df.loc[:,'Adj Close'].to_numpy()


    # Aplique la normalización antes de alimentar a LSTM usando sklearn:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)

    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    '''Función para crear un conjunto de datos para alimentar un LSTM'''
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
        
        
    # Cree los datos para entrenar nuestro modelo en:
    time_steps = 36
    X_train, y_train = create_dataset(train_data, time_steps)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_train = np.reshape(X_train, (X_train.shape[0], 36, 1))



    import keras
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Construye el modelo
    model = keras.Sequential()

    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(units = 1))

    # Compilando el modelo
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Ajuste del modelo al conjunto de entrenamiento
    history = model.fit(X_train, y_train, epochs = 20, batch_size = 10, validation_split=.30)

    # Plot training & validation loss values with streamlit
    st.subheader('Gráfico de pérdida de entrenamiento y validación')
    fig = px.line(history.history, y=['loss', 'val_loss'], title='Pérdida de entrenamiento y validación')
    st.plotly_chart(fig)


    # Obtenga los precios de las acciones para 2019 para que nuestro modelo haga las predicciones
    test_data = test_df['Adj Close'].values
    test_data = test_data.reshape(-1,1)
    test_data = scaler.transform(test_data)

    # Cree los datos para probar nuestro modelo en:
    time_steps = 36
    X_test, y_test = create_dataset(test_data, time_steps)

    # almacenar los valores originales para trazar las predicciones
    y_test = y_test.reshape(-1,1)
    org_y = scaler.inverse_transform(y_test)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_test = np.reshape(X_test, (X_test.shape[0], 36, 1))

    # Predecir los precios con el modelo.
    predicted_y = model.predict(X_test)
    predicted_y = scaler.inverse_transform(predicted_y)

    # Graficar los resultados con streamlit
    st.subheader('Gráfico de predicción de precios de cierre de '+user_input)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=org_dates, y=org_y.reshape(-1), name='Real'))
    fig.add_trace(go.Scatter(x=org_dates[36:], y=predicted_y.reshape(-1), name='Predicción'))
    st.plotly_chart(fig)

    #mostrar en una tabla los valores reales y los predichos
    st.subheader('Tabla de valores reales y predichos')
    df = pd.DataFrame({'Real': org_y.reshape(-1), 'Predicho': predicted_y.reshape(-1)})
    st.write(df)

    #imprimir la evaluación del modelo
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score

    st.subheader('Evaluación del modelo')
    st.write('MSE: '+str(mean_squared_error(org_y, predicted_y)))
    st.write('MAE: '+str(mean_absolute_error(org_y, predicted_y)))
    st.write('R2: '+str(r2_score(org_y, predicted_y)))

    # Graficar en barras las metricas de evaluación con streamlit
    st.subheader('Gráfico de métricas de evaluación')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=['MSE', 'MAE', 'R2'], y=[mean_squared_error(org_y, predicted_y), mean_absolute_error(org_y, predicted_y), r2_score(org_y, predicted_y)], name='Metricas'))
    st.plotly_chart(fig)




