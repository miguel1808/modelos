import streamlit as st
from multiapp import MultiApp
from apps import home,lstm_web,portfolio_web,gru_web,svm_web

app = MultiApp()

st.markdown(""" #Inteligencia de negocios -Equipo A """)

# Add all your application here

app.add_app("Home", home.app)
app.add_app("LSTM", lstm_web.app)
app.add_app("PCA and Hierarchical Portfolio Optimisation", portfolio_web.app)
app.add_app("GRU", gru_web.app)
app.add_app("SVM",svm_web.app)




# The main app
app.run()