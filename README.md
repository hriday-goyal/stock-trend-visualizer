# 📈 Stock Price Trend Visualizer & Predictor

This project is a real-time stock trend dashboard that fetches historical stock data, calculates technical indicators (SMA & EMA), and predicts the next-day closing price using a trained machine learning model.

### 🌐 Live App  
👉 [Launch the App on Streamlit Cloud](https://stock-trend-visualizer-modeldate10july2025.streamlit.app)

---

## 🧠 Key Features

- 📊 Visualizes historical stock prices with SMA and EMA overlays
- 🤖 Predicts the next-day closing price using Linear Regression
- 📡 Fetches real-time data from Yahoo Finance (`yfinance`)
- 🚀 Deployed on Streamlit Cloud, no local setup required

---

## 🛠️ Tech Stack

| Tool          | Purpose                            |
|---------------|-------------------------------------|
| `Python`      | Core programming language           |
| `Streamlit`   | Interactive web app framework       |
| `scikit-learn`| ML model training & prediction      |
| `yfinance`    | Real-time financial data retrieval  |
| `pandas`      | Data preprocessing                  |
| `matplotlib`  | Optional charting support           |

---

## 📂 Repository Structure
stock-trend-visualizer/
├── app/
│ ├── streamlit_app.py ← Main app script
│ └── linear_model.pkl ← Trained ML model
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 03_model_training.ipynb
├── requirements.txt
├── README.md

---

## 🧪 To Run Locally

```bash
git clone https://github.com/hriday-goyal/stock-trend-visualizer
cd stock-trend-visualizer/app
pip install -r ../requirements.txt
streamlit run streamlit_app.py

---
## About the Creator
Built by Hriday Goyal, an aspiring data scientist and high school innovator passionate about AI, finance, and real-world problem solving.
