import base64
import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go

st.set_page_config(page_title="ForesightStocks", layout="wide")

@st.cache_data
def get_img_as_base64(file):
    with open(file,"rb") as f:
        data=f.read()
    return base64.b64encode(data).decode()

img=get_img_as_base64("background.jpg")

page_bg_img=f"""
<style>

[data-testid="stAppViewContainer"] {{
background-image:url("data:background/png;base64,{img}");
background-size:cover;
}}

[data-testid="stHeader"] {{
background:rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

st.title("ForeSight Stocks")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

selected_stocks = st.text_input("Enter stock symbol or company name for prediction")
copyselected_stocks=selected_stocks
st.subheader("Years of prediction:")
n_years = st.slider("",1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Validating stock symbol...")

# Validate stock symbol
try:
    stock_exists = False
    while not stock_exists:
        df = pd.read_csv("NAMES.csv") 
        selected_stocks=selected_stocks.upper()
        df = df[df['NAME'].str.contains(selected_stocks)] 
        try:
            selected_stocks = df.iloc[0,1]
        except:
            selected_stocks = copyselected_stocks.upper()
        print(selected_stocks)
        data = load_data(selected_stocks)
        stock_exists = True
        

    if not data.empty:
        data_load_state.text("Load data...Done!")
        st.subheader('Raw data')
        st.write(data.tail())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.update_layout(title="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        data_load_state.text(" Failed to Load Data :(")
        st.write("Invalid stock symbol. Enter a valid stock symbol for prediction")

    # Forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    st.write(forecast.tail())


    # Forecast data plot with traces
    fig_forecast = go.Figure()

    # Add observed data trace
    fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers', name='Observed Data'))

    # Add forecasted data trace
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Add upper and lower confidence interval traces
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        name='Upper CI'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        name='Lower CI'
    ))

    fig_forecast.update_layout(
        title='Forecast Plot',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig_forecast)

    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
except:
    st.write("Waiting for input")
