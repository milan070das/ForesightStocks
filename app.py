import base64
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go


# ---------------- UI / Background ----------------
@st.cache_data
def get_img_as_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(image_path: str = "bg.jpg"):
    try:
        img = get_img_as_base64(image_path)
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:url("data:image/png;base64,{img}");
            background-size:cover;
        }}
        [data-testid="stHeader"] {{
            background:rgba(0,0,0,0);
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image not found: {image_path}")


# ---------------- Data helpers ----------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


@st.cache_data
def load_names_csv(path: str = "NAMES.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    # group_by="column" helps keep OHLC columns non-multiindex in most cases
    df = yf.download(
        ticker,
        START,
        TODAY,
        progress=False,
        group_by="column",
        auto_adjust=False,
    )

    # If MultiIndex columns appear, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    return df


def resolve_ticker(user_text: str, names_df: pd.DataFrame) -> str:
    q = (user_text or "").strip().upper()
    if not q:
        return ""

    # Try resolving from company names
    if not names_df.empty and "NAME" in names_df.columns:
        hits = names_df[names_df["NAME"].astype(str).str.upper().str.contains(q, na=False)]
        if not hits.empty:
            # Your earlier code assumed the ticker is at column index 1.
            return str(hits.iloc[0, 1]).strip().upper()

    # Fallback: treat input as ticker
    return q


def make_prophet_train_df(data: pd.DataFrame) -> pd.DataFrame:
    df_train = data.loc[:, ["Date", "Close"]].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
    df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")

    df_train = df_train.dropna(subset=["ds", "y"])
    return df_train


# ---------------- App ----------------
set_background("background.jpg")
st.title("ForeSight Stocks")

with st.form("predict_form"):
    user_input = st.text_input("Enter stock symbol or company name for prediction")
    n_years = st.slider("Years of prediction:", 1, 10, 1)
    submitted = st.form_submit_button("Run forecast")

if not submitted:
    st.info("Enter a symbol/name and click Run forecast.")
    st.stop()

period = n_years * 365

# Load company-name mapping (optional)
try:
    names_df = load_names_csv("NAMES.csv")
except FileNotFoundError:
    names_df = pd.DataFrame()
    st.warning("NAMES.csv not found. Company-name lookup disabled.")

ticker = resolve_ticker(user_input, names_df)
st.write(f"Using symbol: {ticker}")

if not ticker:
    st.error("Please enter a stock symbol or company name.")
    st.stop()

# Download stock data
status = st.text("Loading data...")
data = load_data(ticker)

if data is None or data.empty:
    status.text("Failed to load data.")
    st.error("No data returned. Check the symbol or try again later.")
    st.stop()

if "Date" not in data.columns or "Close" not in data.columns:
    status.text("Failed to load usable OHLC data.")
    st.error(f"Expected columns not found. Got columns: {list(data.columns)}")
    st.stop()

status.text("Load data...Done!")

# Raw data display + historical plot
st.subheader("Raw data")
st.write(data.tail())

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
fig_hist.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
fig_hist.update_layout(title="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_hist, use_container_width=True)

# Forecasting with Prophet
df_train = make_prophet_train_df(data)

# Debug checks (can remove later)
st.caption(f"Training rows: {len(df_train)} | y dtype: {df_train['y'].dtype}")

if df_train.empty:
    st.error("Training data became empty after cleaning. Try another symbol.")
    st.stop()

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

# Forecast plot (Plotly) + correct CI fill ordering
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], mode="markers", name="Observed Data"))
fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))

# Lower first, then upper with fill='tonexty' to create a band
fig_forecast.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    name="Lower CI",
))
fig_forecast.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    fill="tonexty",
    fillcolor="rgba(0,100,80,0.2)",
    name="Upper CI",
))

fig_forecast.update_layout(
    title="Forecast Plot",
    xaxis_title="Date",
    yaxis_title="Close Price",
    xaxis_rangeslider_visible=True,
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Prophet component plots (Matplotlib figure)
st.subheader("Forecast components")
st.pyplot(m.plot_components(forecast))

