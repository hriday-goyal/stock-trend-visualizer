import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("app/linear_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title
st.title("📈 Stock Price Trend Visualizer & Predictor")

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

# Load stock data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)

    # Fix multilevel columns if they appear
    df.columns.name = None
    df = df.loc[:, ~df.columns.duplicated()]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

if symbol:
    df = load_data(symbol, start_date, end_date)

    if not df.empty:
        st.subheader(f"📊 {symbol} Stock Closing Price")
        try:
            st.line_chart(df[['Close', 'SMA_20', 'EMA_20']])
        except KeyError:
            st.warning("Some columns are missing. Showing only available data.")
            st.line_chart(df[['Close']])

        # Prepare lag features for prediction
        df_lagged = df[['Close']].copy()
        for i in range(1, 6):
            df_lagged[f'lag_{i}'] = df_lagged['Close'].shift(i)
        df_lagged.dropna(inplace=True)

        X_latest = df_lagged.drop('Close', axis=1).tail(1)

        if not X_latest.empty:
            prediction = model.predict(X_latest)[0]
            currency = "₹" if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO") else "$"
            st.success(f"📌 Predicted Next Day Closing Price: **{currency}{prediction:.2f}**")

        # Show raw data
        with st.expander("📂 View Raw Data"):
            st.dataframe(df.tail(10))

    else:
        st.warning("No data found for the selected symbol and date range.")
else:
    st.info("Please enter a stock symbol to begin.")
