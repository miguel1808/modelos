from datetime import date
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
import streamlit as st
import plotly.express as px


def app():

    start = st.date_input('Inicio(Start)',value=pd.to_datetime('2010-01-01'))
    end= st.date_input('Fin(End)',value=pd.to_datetime('today'))
    st.title("Twitter Sentiment Analysis")
    st.subheader('Extrayendo los Twitters')

    # Crear una lista para ajuntar tweets
    tweets_list = []
    maxTweets = 1000
    # Utilizando TwitterSearchScraper para "scrapear" data y obtener tweets a la lista
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('crypto since:2020-01-01 until:{today}').get_items()):
        if i>maxTweets:
            break
        tweets_list.append([tweet.content])
        
    # Creando el dataframe de los tweets extraidos
    tweets_to_df = pd.DataFrame(tweets_list, columns=['Tweets'])
    st.write(tweets_to_df.head())

    st.subheader('Limpiando los Tweets')
    # Función para limpiar tweets
    def cleanTweets(text):
        text = re.sub('@[A-Za-z0-9_]+', '', text) # Removiendo menciones (@mencion)
        text = re.sub('#','',text) # Removiendo símbolo "#"
        text = re.sub('RT[\s]+','',text)
        text = re.sub('https?:\/\/\S+', '', text) 
        text = re.sub('\n',' ',text)
        return text
    tweets_to_df['cleanedTweets'] = tweets_to_df['Tweets'].apply(cleanTweets) # Aplicando función cleanTweets
    st.write(tweets_to_df.head()) # Comparando tweets originales con los limpiados

    tweets_to_df.to_csv('tweets_crypto.csv') # Escribiendo dataframe a un archivo csv
    savedTweets = pd.read_csv('tweets_crypto.csv',index_col=0) # Leyendo archivo csv

    st.subheader('Detectando Sentimientos')
    # Función para obtener la subejtividad de los tweets
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    # Función para obtener la polaridad de los tweets
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    savedTweets['Subjectivity'] = savedTweets['cleanedTweets'].apply(getSubjectivity)
    savedTweets['Polarity'] = savedTweets['cleanedTweets'].apply(getPolarity)
    st.write(savedTweets.drop('Tweets', axis=1).head()) # Mostrando subjetividad y polaridad de los tweets limpiados

    # Creando una función para clasificar como negativo, neutral y positivos
    def getAnalysis(score):
        if score<0:
            return 'Negative'
        elif score ==0:
            return 'Neutral'
        else:
            return 'Positive'
        
    savedTweets['Analysis'] = savedTweets['Polarity'].apply(getAnalysis)
    st.write("Sentimientos de los tweets")
    st.write(savedTweets['Analysis'].value_counts() )# La cantidad de tweets según su polaridad
    df=savedTweets['Analysis'].value_counts()
    positive=df.Positive
    negative=df.Negative
    neutral=df.Neutral
    # Visualizando los tweets y su polaridad
    st.subheader('Visualizando los Tweets y su Polaridad')
    datos={
        'polaridad':['positiva','negativa','neutral'],
        'cantidad':[positive,negative,neutral]
    }
    datos=pd.DataFrame(datos)
    
    fig=px.bar(
        datos,
        x='polaridad',
        y='cantidad',
        color='polaridad',
        title='Sentimientos de los tweets'
    )
    st.plotly_chart(fig)

    st.subheader('# Gráfica de distribucion de polaridad en forma de pie')
    fig2=px.pie(
        datos,
        values='cantidad',
        names='polaridad',
        title='Distribucion de polaridad en forma de pie',
        color='polaridad'
    )
    st.plotly_chart(fig2)


    st.subheader('Trazar la polaridad y la subjetividad en un diagrama de dispersión')
    # Visualizando la polaridad y la subjetividad
    fig3=px.scatter(
        savedTweets,
        x='Polarity',
        y='Subjectivity',
        hover_name='Analysis',

    )
    st.plotly_chart(fig3)

    st.subheader('Creando una nube de palabras para los tweets')
    st.code("""# Creando una nube de palabras para los tweets
    def create_wordcloud(text):    
        allWords = ' '.join([tweets for tweets in text])
        wordCloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(allWords)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wordCloud, interpolation="bilinear")
        ax.axis('off')
        st.pyplot(fig)""",language='python')
    
    # Creando una nube de palabras para los tweets
    def create_wordcloud(text):    
        allWords = ' '.join([tweets for tweets in text])
        wordCloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(allWords)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wordCloud, interpolation="bilinear")
        ax.axis('off')
        st.pyplot(fig)
    
    st.write('## Nube de palabras para tweets positivos')
    st.code("""posTweets = savedTweets.loc[savedTweets['Analysis']=='Positive', 'cleanedTweets']
    create_wordcloud(posTweets)""",language='python')

    posTweets = savedTweets.loc[savedTweets['Analysis']=='Positive', 'cleanedTweets']
    create_wordcloud(posTweets)

    st.write('## Nube de palabras para tweets negativos')
    # Nube de palabras para tweets negativos
    st.code("""negTweets = savedTweets.loc[savedTweets['Analysis']=='Negative', 'cleanedTweets']
    create_wordcloud(negTweets)""",language='python')


    negTweets = savedTweets.loc[savedTweets['Analysis']=='Negative', 'cleanedTweets']
    create_wordcloud(negTweets)

    st.subheader('Encontrando las palabras más populares en los tweets y su frecuencia')
    # Separando cada tweet en palabras
    sentences = []
    for word in savedTweets['cleanedTweets']:
        sentences.append(word)
    sentences
    lines = list()
    for line in sentences:
        words = line.split()
        for w in words:
            lines.append(w)



    st.write('Codigo:')
    st.code("""# Separando cada tweet en palabras
    sentences = []
    for word in savedTweets['cleanedTweets']:
        sentences.append(word)
    sentences
    lines = list()
    for line in sentences:
        words = line.split()
        for w in words:
            lines.append(w)""",language='python')
    
    st.write('Resultado:')
    st.write(lines[:10])

    # Derivar todas las palabras a su raíz
    st.write('## Derivar todas las palabras a su raíz')
    stemmer = SnowballStemmer(language='english')
    stem=[]
    for word in lines:
        stem.append(stemmer.stem(word))
    stem[:20]
    # Removiendo "stopwords"
    stem2 = []
    for word in stem:
        if word not in nlp.Defaults.stop_words:
            stem2.append(word)
    # Creando un nuevo dataframe para la raíz y muestra el recuento de las palabras más utilizadas
    df = pd.DataFrame(stem2)
    df=df[0].value_counts()
    st.write(df) # Mostrando el dataframe


    st.subheader('# Graficando el top de palabras mas usadas')
    df=df[:20]
    fig4=px.bar(
        df,
        x=df.values,
        y=df.index,
        color=df.index,
        title='Top de palabras mas usadas',
        orientation='h',
        height=600,
        width=800
    )
    
    st.plotly_chart(fig4)