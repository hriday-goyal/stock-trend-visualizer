import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="📈 Stock Price Trend Visualizer & Predictor")
st.title("📈 Stock Price Trend Visualizer & LSTM Predictor")
st.markdown("""
This app uses historical stock data and a trained LSTM model to forecast stock prices.
""")

# === Load Model & Scaler ===
@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_model.h5")
    scaler = joblib.load("lstm_scaler.pkl")
    return model, scaler

model, scaler = load_lstm_model()

# === Multi-step Forecast Function ===
def multistep_forecast(df, steps, seq_len=10):
    df_copy = df.copy()
    df_copy['SMA_20'] = df_copy['Close'].rolling(20).mean()
    df_copy['EMA_20'] = df_copy['Close'].ewm(span=20, adjust=False).mean()
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))

    for i in range(1, 6):
        df_copy[f'lag_{i}'] = df_copy['Close'].shift(i)

    features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14',
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']

    preds = []
    window = df_copy[features].tail(seq_len).values
    window_scaled = scaler.transform(window)

    for _ in range(steps):
        X = window_scaled.reshape(1, seq_len, -1)
        next_scaled = model.predict(X, verbose=0)[0][0]
        new_scaled_row = np.zeros((len(features),))
        new_scaled_row[0] = next_scaled
        new_scaled_row[1:] = window_scaled[-1, 1:]
        window_scaled = np.vstack([window_scaled, new_scaled_row])[1:]
        next_close = scaler.inverse_transform(
            np.hstack([new_scaled_row.reshape(1, -1), np.zeros((1, 0))]))[0][0]
        preds.append(next_close)

    return preds

# === Sidebar Inputs ===
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Enter stock symbol (e.g. AAPL, RELIANCE.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2022-01-01"))

# === Fetch Data ===
end_date = pd.Timestamp.today()
df = yf.download(symbol, start=start_date, end=end_date)

if df.empty:
    st.error("No data found. Please check the stock symbol or date range.")
    st.stop()

st.subheader(f"Showing data for: {symbol}")
st.line_chart(df['Close'])

# === Predict next day ===
st.subheader("📅 Next Day Price Prediction")
df_ind = df.copy()
df_ind['SMA_20'] = df_ind['Close'].rolling(20).mean()
df_ind['EMA_20'] = df_ind['Close'].ewm(span=20, adjust=False).mean()
delta = df_ind['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df_ind['RSI_14'] = 100 - (100 / (1 + rs))

for i in range(1, 6):
    df_ind[f'lag_{i}'] = df_ind['Close'].shift(i)

features = ['Close', 'SMA_20', 'EMA_20', 'RSI_14',
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
df_ind.dropna(inplace=True)

X_last = df_ind[features].tail(10).values.reshape(1, 10, len(features))
pred_scaled = model.predict(X_last, verbose=0)[0][0]
pred_unscaled = scaler.inverse_transform(
    np.hstack([[[pred_scaled] + [0] * (len(features) - 1)]]
))[0][0]

currency = "₹" if symbol.upper().endswith(".NS") or symbol.upper().endswith(".BO") else "$"
st.success(f"Predicted next day closing price: {currency}{pred_unscaled:.2f}")

# === Multi-step Forecast Section ===
st.subheader("🧠 Extended Forecast (LSTM)")
horizons = [1, 7, 30, 365]
future_prices = {h: multistep_forecast(df.copy(), h)[-1] for h in horizons}

cols = st.columns(len(horizons))
for i, h in enumerate(horizons):
    cols[i].metric(f"{h}-Day Forecast", f"{currency}{future_prices[h]:.2f}")

# === Forecast Plot ===
last_date = df.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=max(horizons))

plt.figure(figsize=(12, 5))
plt.plot(df.index[-60:], df['Close'].tail(60), label="Actual Price", color='blue')
plt.plot(future_dates, [future_prices[h] for h in horizons],
         label="Forecast", color='red', linestyle='--', marker='o')
plt.title(f"{symbol} - Forecast Up To {max(horizons)} Days")
plt.legend()
st.pyplot(plt)

# Optional raw data view
with st.expander("📂 View Raw Data"):
    st.dataframe(df.tail(30))
