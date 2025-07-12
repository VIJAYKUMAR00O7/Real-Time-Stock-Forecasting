#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# --------------------------------
# App Title
# --------------------------------
st.set_page_config(layout="wide")
st.title("📈 Real-Time Market Price Forecasting (LSTM Model)")

# --------------------------------
# Sidebar - Inputs
# --------------------------------
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., TATAMOTORS.NS):", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("Days to Predict", 1, 5, 2)

# --------------------------------
# Load Data
# --------------------------------
@st.cache_data
def load_data(ticker, start_date):
    df = yf.download(ticker, start=start_date, end=pd.to_datetime("today").strftime('%Y-%m-%d'))
    return df[['Close']].dropna()

df = load_data(ticker, start_date)

# --------------------------------
# Preprocess for LSTM
# --------------------------------
sequence_length = 60

def prepare_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

X_train, y_train, scaler = prepare_data(df.values)

# --------------------------------
# Build and Train LSTM Model
# --------------------------------
@st.cache_resource
def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    return model

model = build_and_train_model(X_train, y_train)

# --------------------------------
# Prediction Function
# --------------------------------
def predict_market_price(df, model, scaler, days=2):
    df_scaled = scaler.transform(df[-sequence_length:])
    seq_input = np.reshape(df_scaled, (1, sequence_length, 1))

    predictions = []
    input_seq = seq_input.copy()
    
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    prediction_dates = pd.date_range(start=pd.to_datetime("today"), periods=days, freq='B')

    return pd.DataFrame(predictions, index=prediction_dates, columns=["LSTM_Prediction"])

# --------------------------------
# Run Prediction
# --------------------------------
predictions = predict_market_price(df[['Close']], model, scaler, days=forecast_days)

# --------------------------------
# Display Results
# --------------------------------
st.subheader("📊 Market Close Price Over Time")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['Close'], label='Historical Close Price')
ax.plot(predictions.index, predictions['LSTM_Prediction'], marker='o', linestyle='--', label='LSTM Forecast')
ax.set_title(f"{ticker} Close Price & Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# --------------------------------
# Display Prediction Table
# --------------------------------
st.subheader("📅 Forecast Table")
st.dataframe(predictions.style.highlight_max(axis=0, color='lightgreen'))


