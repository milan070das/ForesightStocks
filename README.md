# 📈 ForesightStocks — AI-Powered Stock Market Forecasting App

**ForesightStocks** is an intelligent web application that uses deep learning models and time series analysis to predict stock prices for selected companies. It allows users to interactively choose a stock, view its historical data, visualize future price trends, and receive predictions with compelling graphs and metrics.

---

## 🚀 Features

- 📊 **Interactive Stock Selection** – Choose from a variety of popular stock tickers (e.g., AAPL, TSLA, MSFT).
- 📈 **Time Series Forecasting** – Predict future stock prices using LSTM-based deep learning models.
- 📉 **Data Visualization** – Dynamic charts for historical and predicted stock data using Plotly.
- 🧠 **AI Explanation Assistant** – Ask financial or AI-related queries via integrated Gemini-powered chatbot.
- 📅 **Date Range Filtering** – Select custom date ranges to visualize stock trends.
- 🌙 **Dark Mode UI (planned)** – Comfortable viewing experience for long sessions.

---

## 🛠️ Tech Stack

| Layer          | Technology                           |
|----------------|--------------------------------------|
| Frontend       | Streamlit                            |
| Backend        | Python                               |
| ML Framework   | TensorFlow/Keras, Scikit-learn       |
| Visualization  | Plotly, Matplotlib                   |
| Data Source    | Yahoo Finance (via yfinance)         |
| AI Assistant   | Google Gemini API (Generative AI)    |

---

## 📁 Folder Structure

```plaintext
ForesightStocks/
├── app.py                         # Main Streamlit application
├── model/
│   └── stock_lstm_model.h5       # Pretrained LSTM model (or models per stock)
├── utils/
│   ├── data_loader.py            # Fetches stock data
│   ├── model_utils.py            # LSTM model logic
│   └── plot_utils.py             # Graphs and visualizations
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
└── .gitignore
