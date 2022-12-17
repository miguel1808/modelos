
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import math
import datetime as dt
import time
import arrow
import streamlit as st
from sklearn import preprocessing,svm # Preprocessing for scaling data,Accuracy,Processing speed ,cross validation for training and testing
from sklearn.linear_model import LinearRegression #
import matplotlib.pyplot as plt
from matplotlib import style
import yfinance as yf
import pandas_datareader as datas
style.use('ggplot')
from datetime import timedelta
import plotly.express as px
from sklearn.model_selection import train_test_split


def app(): 
    st.title('Predicción de tendencia de acciones usando SVM')
    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end = st.date_input('Fin' , value=pd.to_datetime('today'))


    user_input = st.text_input('Introducir cotización bursátil' , 'ADA-USD')
    df = yf.download(user_input, start, end)

    st.subheader('Acerca de la empresa')
    st.write(datas.get_quote_yahoo(user_input))
    st.subheader('Datos de la empresa')
    st.write(df)

    st.subheader("Definir las variables explicativas")
    st.write("Las variables explicativas son las que se utilizan para predecir el valor de la variable dependiente. En este caso, la variable dependiente es el precio de cierre de la acción y las variables explicativas son el precio de apertura, el precio más alto, el precio más bajo y el volumen de la acción.")
    
    df['HIGHLOW_PCT']=(df['High']-df['Close'])/(df['Close'])*100
    #Calculating new and old prices
    df['PCT_Change']=(df['Close']-df['Open'])/(df['Open'])*100
    # Extracting required data from file
    df=df[['Close','HIGHLOW_PCT','PCT_Change','Volume']]
    st.write(df.head())


    st.subheader("Creación de la columna 'label' en el dataframe para la predicción del precio de cierre")

    forecast_col='Close'
    #We have to replace to na data with negative 99999.It will be useful when we lacking with data
    df.fillna(-99999,inplace=True)
    # if the length of data frame is returning decimal point or float it will round up to integer
    # 0.1 means tomorrow data ,we can change accordingly
    forecast_out=int(math.ceil(0.1*len(df)))
    print (forecast_out)
    df['label']=df[forecast_col].shift(-forecast_out)
    st.write(df.head())


    X = np.array(df.drop(['label'],1)) #1 columna en vez de fila
    # Estandariza las características de un conjunto de datos
    X = preprocessing.scale(X)
    X = X[:-forecast_out] ## eliminando filas superiores

    X_lately=X[-forecast_out:] #de forecast_out en adelante

    df.dropna(inplace=True)
    y=np.array(df['label'])



    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) ## conjunto de pruebas, que en este caso es del 20% 
    split_percentage = 0.8
    split = int(split_percentage*len(df)) ## datos para entrenamiento 
    st.write("Datos para entrenamiento:"+ str(split) )

    clf=svm.SVR() # support vector regression con svm 
    clf.fit(X_train,y_train)
    accuracy=clf.score(X_test,y_test)
    st.write("Precisión del modelo:"+ str(accuracy) )

    forecast_set=clf.predict(X_lately)

    df['Forecast']=np.nan
    last_date=df.iloc[-1].name #ultimo dia de la fila 
    last_unix = arrow.get(last_date).timestamp() ## conversion a segundo desde 
    print(last_unix)
    one_day=86400 ## valor de segundos en un dia 
    next_unix=last_unix + one_day
    for i in forecast_set:
        next_date=dt.datetime.fromtimestamp(next_unix) #conversión a objeto de fecha y hora de Python
        next_unix +=one_day # siguiente dia 
        df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i] # Agregamos una fila al DataFrame con la fecha generada


    st.subheader("Gráfico de la predicción")
    tabla=df.reset_index()

    f, ax = plt.subplots(figsize=(10, 6))
    sA = "Close"
    sB = "Forecast"
    ax.plot(tabla[sA], label=sA)
    ax.plot(tabla[sB], label=sB)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Predicción de cierre de acciones")
    ax.legend(loc='upper left',prop={'size':8})
    plt.setp(ax.get_xticklabels(), rotation=70)
    st.pyplot(f)


    # Obtener la fecha y hora actuales
    #today = datetime.today()-timedelta(days=1)
    today = dt.datetime.today()

    # Crear un array vacío para almacenar las fechas posteriores
    dates = []

    # Iterar 15 veces para agregar 15 días a la fecha actual
    for i in range(187):
        # Agregar un día a la fecha actual utilizando timedelta()
        next_date = today + timedelta(days=i+1)
        # Añadir la fecha al array
        dates.append(next_date)

    # Imprimir el array de fechas
    df1 = pd.DataFrame(dates, columns=["Fecha"])
    st.subheader(" Crear un array para almacenar las fechas posteriores")
    st.write(df1)


    st.subheader("Concatenar los dos DataFrames")
    df2=tabla['Forecast'].dropna()
    df2=df2.reset_index(drop=True)
    
    # Concatena los dos DataFrames
    df_concat = pd.concat([df1, df2], axis=1)
    st.write(df_concat)

    st.subheader("Buscamos el valor minimo y maximo de la predicción")
    df_concat_min=df_concat['Forecast'].idxmin()
    st.write(" Obtenemos la fila donde se encuentra el valor mínimo")
    fila = df_concat.loc[df_concat_min]
    st.write(fila)

    df_concat_max=df_concat['Forecast'].idxmax()
    df_concat_max
    fila = df_concat.loc[df_concat_max]

    st.write(" Obtenemos la fila donde se encuentra el valor maximo")
    st.write(fila)

    st.subheader("Gráfica")
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_concat['Forecast'], color='red', label='Forecast')
    ax.set_xlabel("Date")
    st.pyplot(f)


    st.subheader("Gráfica scatter")
    fig=px.scatter(df_concat, x="Fecha", y="Forecast", title="Predicción de cierre de acciones")
    st.write(fig)

