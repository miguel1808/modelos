import streamlit as st
from matplotlib import style
import seaborn as sns
from scipy.stats import randint as sp_randint
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import pandas_datareader as datas
from scipy.cluster.hierarchy import ward, dendrogram, linkage,cophenet
import matplotlib.cm as cm
from scipy.spatial.distance import pdist
import pylab
import yfinance as yf
def app():    
    st.title('Modelo 12 - PCA and Hierarchical Portfolio Optimisation')
    start = st.date_input('Start' , value=pd.to_datetime('2000-01-01'))
    end = st.date_input('End' , value=pd.to_datetime('today'))


    user_input = st.text_input('Introducir cotización bursátil' , 'NFLX')

    df = yf.download(user_input, start, end)
    st.subheader('Datos del 2000 al 2022') 
    st.write(df.describe())
    #Visualizaciones 

    st.subheader('Gráfico Financiero') 
    candlestick = go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close']
                                )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
            width=800, height=600,
            title=user_input,
            yaxis_title='Precio'
        )
        
    st.plotly_chart(fig)

    st.write(df.describe())
    # Use PCA to reduce the number of features in the dataset
    st.subheader("PCA")
    returns = df
    returns["valor"]= returns.mean(axis=1)
    returns["valor"] = returns["valor"].pct_change(1)
    returns.dropna(how='any', inplace=True)
    st.dataframe(returns.head(10))
    # Perform PCA on the returns
    pca = PCA()
    pca_data =pca.fit(returns)
    # Plot the cumulative sum of the explained variance ratio in pyplot
    st.subheader("Explained Variance Ratio")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    st.pyplot(fig)

    # Compute the hierarchical clusters
    clusters = linkage(pca.components_, method="ward")


    # Plot the dendrogram of the hierarchical clusters

    row = returns.loc[:,"valor"]
    stddev = row.std()
    # calculate the Sharpe ratio
    sharpe = returns["valor"] / stddev 
    # plot the Sharpe ratio in a figure
    st.subheader("Sharpe Ratio")
    fig = plt.figure()
    ax = sharpe.plot()
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio for " + "Valor")
    st.pyplot(fig)

    st.subheader("Matriz de correlación")
    corr = returns.corr()
    size = 7
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr,cmap=cm.get_cmap('coolwarm'), vmin=0,vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical', fontsize=8);
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8);
    st.pyplot(fig)
    Z = linkage(corr, 'average')
    c, coph_dists = cophenet(Z, pdist(corr))
    plt.figure(figsize=(25, 10))
    labelsize=20
    ticksize=15
    plt.title('Hierarchical Clustering Dendrogram for '+"Valor", fontsize=labelsize)
    plt.xlabel('Caracteristic', fontsize=labelsize)
    plt.ylabel('distance', fontsize=labelsize)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = corr.columns
    )
    pylab.yticks(fontsize=ticksize)
    pylab.xticks(rotation=-90, fontsize=ticksize)
    plt.savefig('dendogram_'+'Valor'+'.png')
    st.subheader("Dendograma")
    #mostrar una imagen
    st.image('dendogram_'+'Valor'+'.png')
    #plot sample correlations
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    #high correlation
    sA = "High"
    sB = "valor"
    ax1.plot(returns[sA],label=sA)
    ax1.plot(returns[sB],label=sB)
    ax1.set_title('Correlación = %.3f'%corr[sA][sB])
    ax1.set_ylabel('Normalized Adj Close prices')
    ax1.legend(loc='upper left',prop={'size':8})
    plt.setp(ax1.get_xticklabels(), rotation=70);

    #low correlation
    sA = "Open"
    sB = "High"
    ax2.plot(returns[sA],label=sA)
    ax2.plot(returns[sB],label=sB)
    ax2.set_title('Correlación de caracteristicas = %.3f'%corr[sA][sB])
    ax2.legend(loc='upper left',prop={'size':8})
    plt.setp(ax2.get_xticklabels(), rotation=70)
    st.subheader("Correlación de caracteristicas")
    st.pyplot(f)