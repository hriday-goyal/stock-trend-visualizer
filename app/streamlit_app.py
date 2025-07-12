# 📈 Streamlit App for Stock Price Trend & Prediction (Random Forest Model)

import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load Random Forest model
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
import joblib

@st.cache_resource
def load_lstm_model():
    model = load_model("app/lstm_model.h5")
    scaler = joblib.load("app/lstm_scaler.pkl")
    return model, scaler

model, scaler = load_lstm_model()
    return model

model = load_model()

# ✅ Sidebar - Stock selection
st.sidebar.header("📊 Stock Price Predictor")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS)", value="AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ✅ Load stock data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.columns.name = None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

df = load_data(symbol, start, end)

if df.empty:
    st.warning("No data found. Please check the stock symbol and date range.")
    st.stop()

# ✅ Add indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# RSI calculation
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# ✅ Prepare features for prediction
df_lagged = df[['Close', 'SMA_20', 'EMA_20', 'RSI_14']].copy()
for i in range(1, 6):
    df_lagged[f'lag_{i}'] = df['Close'].shift(i)

df_lagged.dropna(inplace=True)
features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'SMA_20', 'EMA_20', 'RSI_14']
X_latest = df_lagged[features].tail(1)

# ✅ Predict
if not X_latest.empty:
    prediction = model.predict(X_latest)[0]
    currency = "₹" if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO") else "$"
    st.success(f"📌 Predicted Next Day Closing Price: **{currency}{prediction:.2f}**")

    # ✅ Plot actual vs predicted (last 30 days)
    try:
        st.subheader("📊 Actual vs Predicted Closing Price (Last 30 Days)")
        df_plot = df_lagged.tail(30).copy()
        X_plot = df_plot[features]
        y_actual = df_plot['Close']
        y_predicted = model.predict(X_plot)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_actual.index, y_actual, label="Actual Price", color='blue')
        ax.plot(y_actual.index, y_predicted, label="Predicted Price", color='red', linestyle='--')
        ax.set_title(f"{symbol} - Actual vs Predicted Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate comparison plot: {e}")

# ✅ Show raw data
with st.expander("📂 View Raw Data"):
    st.dataframe(df.tail(20))

    else:
        st.warning("No data found for the selected symbol and date range.")
else:
    st.info("Please enter a stock symbol to begin.")
