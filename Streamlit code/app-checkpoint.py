#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


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

# Import Prophet library
from prophet import Prophet 
import plotly.graph_objects as go # Using plotly for interactive plots in Streamlit

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# --------------------------------
# App Title
# --------------------------------
st.set_page_config(layout="wide")
st.title("📈 Real-Time Market Price Forecasting (LSTM & Prophet Models)")

# --------------------------------
# Sidebar - Inputs
# --------------------------------
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., TATAMOTORS.NS):", "TATAMOTORS.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
forecast_days = st.sidebar.slider("Days to Predict", 1, 5, 2)

# --------------------------------
# Load Data (for both models)
# --------------------------------
@st.cache_data
def load_data(ticker, start_date):
    end_date = pd.to_datetime("today").strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    # Ensure 'Close' column exists and drop NaNs
    if 'Close' not in df.columns:
        st.error(f"Error: 'Close' column not found for {ticker}. Please check the ticker symbol.")
        st.stop()
    return df[['Close']].dropna()

df = load_data(ticker, start_date)

if df.empty:
    st.warning("No data downloaded for the specified ticker and date range. Please adjust inputs.")
    st.stop()

# --------------------------------
# Prophet Model Functions
# --------------------------------

# Note: fetch_stock_data is integrated into load_data above for consistency.
# This prepare_prophet_data takes the dataframe from load_data.
def prepare_prophet_data(df_prophet):
    df_prophet = df_prophet.rename(columns={'Close': 'y'})
    df_prophet.index.name = 'ds' # Rename index to 'ds' for Prophet
    df_prophet.reset_index(inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    # Ensure 'y' column is numeric and 1D
    # Using .values.flatten() to explicitly handle potential multi-dimensional Series issues
    df_prophet['y'] = pd.to_numeric(df_prophet['y'].values.flatten(), errors='coerce') 
    
    # Drop rows with NaN values introduced by conversion or already present
    df_prophet.dropna(inplace=True)

    # Check for infinite values and remove rows containing them
    inf_count = df_prophet['y'].isin([np.inf, -np.inf]).sum()
    if inf_count > 0:
        st.warning(f"Found {inf_count} infinite values in Prophet 'y' column. Dropping corresponding rows.")
        df_prophet = df_prophet[~df_prophet['y'].isin([np.inf, -np.inf])].copy()

    # Crucial check: Ensure DataFrame is not empty after all cleaning steps
    if df_prophet.empty:
        raise ValueError("Prophet DataFrame is empty after processing and dropping NaNs/Infs. No sufficient data to train the model.")

    # Ensure 'y' is float type explicitly
    df_prophet['y'] = df_prophet['y'].astype(float)

    # Sort by 'ds' to ensure Prophet gets time series in correct order
    df_prophet = df_prophet.sort_values(by='ds').reset_index(drop=True)

    return df_prophet

@st.cache_resource # Cache the model to avoid retraining on every rerun
def train_and_forecast_prophet(df_prophet, periods=2):
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods) # Return only relevant forecast columns and periods

# --------------------------------
# Preprocess for LSTM
# --------------------------------
sequence_length = 60

def prepare_data_lstm(df_lstm):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_lstm)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Ensure df for LSTM is numpy array of values
X_train, y_train, scaler = prepare_data_lstm(df.values)

if X_train.shape[0] == 0:
    st.warning("Not enough historical data to train the LSTM model with the given sequence length. Please select an earlier start date.")
    st.stop()


# --------------------------------
# Build and Train LSTM Model
# --------------------------------

@st.cache_resource # Cache the model to avoid retraining on every rerun
def build_and_train_model_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    return model

model_lstm = build_and_train_model_lstm(X_train, y_train)

# --------------------------------
# LSTM Prediction Function
# --------------------------------
def predict_market_price_lstm(df_lstm_predict, model_lstm, scaler, days=2):
    # Ensure df_lstm_predict is a numpy array of values for scaling
    df_scaled = scaler.transform(df_lstm_predict.values[-sequence_length:])
    seq_input = np.reshape(df_scaled, (1, sequence_length, 1))

    predictions = []
    input_seq = seq_input.copy()
    
    for _ in range(days):
        pred = model_lstm.predict(input_seq, verbose=0)
        predictions.append(pred[0][0])
        # Update input sequence for next prediction
        input_seq = np.append(input_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    # Generate future dates, accounting for weekends
    prediction_dates = pd.date_range(start=df_lstm_predict.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')

    return pd.DataFrame(predictions, index=prediction_dates, columns=["LSTM_Prediction"])

# --------------------------------
# Run Predictions for both models
# --------------------------------

# LSTM Prediction
predictions_lstm = predict_market_price_lstm(df[['Close']], model_lstm, scaler, days=forecast_days)

# Prophet Prediction
df_prophet_processed = prepare_prophet_data(df.copy()) # Use a copy to avoid modifying original df for LSTM
predictions_prophet = train_and_forecast_prophet(df_prophet_processed, periods=forecast_days)
predictions_prophet.rename(columns={'ds': 'Date', 'yhat': 'Prophet_Prediction'}, inplace=True)
predictions_prophet.set_index('Date', inplace=True)

# --------------------------------
# Display Results (Combined)
# --------------------------------
st.subheader("📊 Market Close Price Over Time (Historical & Forecasts)")

fig = go.Figure()

# Historical Data
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close Price'))

# LSTM Forecast
fig.add_trace(go.Scatter(x=predictions_lstm.index, y=predictions_lstm['LSTM_Prediction'], 
                         mode='lines+markers', name='LSTM Forecast',
                         line=dict(dash='dash')))

# Prophet Forecast
fig.add_trace(go.Scatter(x=predictions_prophet.index, y=predictions_prophet['Prophet_Prediction'], 
                         mode='lines+markers', name='Prophet Forecast',
                         line=dict(dash='dot')))

fig.update_layout(title=f"{ticker} Close Price & Forecast",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Display Combined Forecast Table
# --------------------------------
st.subheader("📅 Combined Forecast Table")

# Combine predictions into a single DataFrame
combined_predictions = pd.concat([predictions_lstm, predictions_prophet], axis=1)

st.dataframe(combined_predictions.style.highlight_max(axis=1, color='lightgreen').format(formatter="{:.2f}"))

st.markdown("""
**Note:**
* **Historical Close Price:** Actual past market close prices.
* **LSTM Forecast:** Predictions from the Long Short-Term Memory neural network model.
* **Prophet Forecast:** Predictions from Facebook's Prophet forecasting model.
""")

