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
import datetime as dt

#from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
#from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
#from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
