import base64
import streamlit as st
from datetime import date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Machine Learning & Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(page_title="ForeSight Stocks - ML Edition", layout="wide")

# -------------------- INDIAN CURRENCY HELPERS --------------------
def format_inr(value: float) -> str:
    """
    Format a number in Indian style with Rupee symbol.
    Example: 1234567.89 -> ₹ 12,34,567.89
    """
    if pd.isna(value):
        return "₹ 0"
    try:
        negative = value < 0
        value = abs(float(value))
        s = f"{value:,.2f}"  # 1,234,567.89
        # Convert to Indian style: 12,34,567.89
        if len(s.split(",")) <= 2:
            formatted = s
        else:
            parts = s.split(".")
            int_part = parts[0]
            dec_part = parts[1]
            first = int_part[-3:]
            rest = int_part[:-3]
            rest = rest.replace(",", "")
            indian = ""
            while len(rest) > 2:
                indian = "," + rest[-2:] + indian
                rest = rest[:-2]
            indian = rest + indian
            formatted = indian + "," + first + "." + dec_part
        return ("- " if negative else "") + "₹ " + formatted
    except Exception:
        return f"₹ {value}"

# ==================== BACKGROUND IMAGE ====================
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

try:
    img = get_img_as_base64("background.jpg")
    if img:
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
except:
    pass

st.title("🎯 ForeSight Stocks - AI-Powered Investment Advisor")
st.markdown(
    "*Check stock price history, get a simple BUY/SELL suggestion, and see which factors influenced that suggestion (using SHAP).*"
)

# ==================== CONFIGURATION ====================
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
LOOKBACK_DAYS = 60  # Days to analyze recent pattern

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_data(ticker):
    """Load historical stock data"""
    data = yf.download(ticker, START, TODAY, progress=False)
    # Flatten MultiIndex columns if they exist (e.g. ('Close','TCS.NS') -> 'Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

def create_features(df):
    """
    Create simple numerical features from price and volume.
    These are like summary statistics that help the model see patterns.
    """
    df = df.copy()

    # Price-based features
    df["Price_Change"] = df["Close"].diff()
    df["Price_Change_Pct"] = df["Close"].pct_change() * 100
    df["Price_High_Low"] = df["High"] - df["Low"]
    df["Price_Open_Close"] = df["Close"] - df["Open"]

    # Moving averages (average closing price over last N days)
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    # Momentum / strength indicators
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["RSI"] = calculate_rsi(df["Close"], period=14)
    df["MACD"], df["Signal"] = calculate_macd(df["Close"])

    # Volatility (how much price swings)
    df["Volatility_20"] = df["Price_Change_Pct"].rolling(window=20).std()
    df["Volatility_50"] = df["Price_Change_Pct"].rolling(window=50).std()

    # Volume features
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

    # Bollinger Bands (price range around average)
    ma = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = ma + (std * 2)
    df["BB_Lower"] = ma - (std * 2)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Average True Range (overall daily movement range)
    df["ATR"] = calculate_atr(df, period=14)

    # Rate of Change (percentage change vs 12 days ago)
    df["ROC"] = ((df["Close"] - df["Close"].shift(12)) / df["Close"].shift(12)) * 100

    return df.dropna()

def calculate_rsi(data, period=14):
    """RSI: shows if stock is overbought (>70) or oversold (<30)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    MACD: compares a fast and slow moving average.
    Positive MACD means recent prices are higher than older prices (upward strength).
    """
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_atr(df, period=14):
    """ATR: average of daily high-low range. Higher ATR = more daily movement."""
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift()),
        ),
    )
    atr = df["TR"].rolling(window=period).mean()
    return atr

def create_target(df, lookahead_days=30):
    """
    Create target: 1 if price goes UP in next lookahead_days, else 0.
    Used to teach the model when BUY would have been good historically.
    """
    df = df.copy()
    df["Future_Price"] = df["Close"].shift(-lookahead_days)
    df["Buy_Signal"] = (df["Future_Price"] > df["Close"]).astype(int)
    return df.dropna()

def train_ensemble_model(X_train, y_train):
    """Train three different models and use them together."""
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    return xgb_model, gb_model, rf_model

def calculate_profitability(current_price, predicted_price, confidence):
    """
    Approximate potential gain:
    - price_change_pct: expected % move in price
    - potential_profit: price_change_pct scaled by model confidence
    """
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    potential_profit = price_change_pct * confidence
    return price_change_pct, potential_profit

def analyze_price_patterns(df):
    """
    Summarise recent behaviour:
    - current price
    - average prices
    - RSI (strength)
    - recent movement and volatility
    """
    recent_data = df.tail(LOOKBACK_DAYS)

    analysis = {
        "current_price": recent_data["Close"].iloc[-1],
        "ma_20": recent_data["MA_20"].iloc[-1],
        "ma_50": recent_data["MA_50"].iloc[-1],
        "ma_200": recent_data["MA_200"].iloc[-1],
        "rsi": recent_data["RSI"].iloc[-1],
        "momentum": recent_data["Momentum_10"].iloc[-1],
        "volatility": recent_data["Volatility_20"].iloc[-1],
        "price_trend": "UPTREND"
        if recent_data["Close"].iloc[-1] > recent_data["MA_200"].iloc[-1]
        else "DOWNTREND",
        "volume_trend": "HIGH"
        if recent_data["Volume_Ratio"].iloc[-1] > 1.2
        else "NORMAL",
    }

    return analysis

def get_recommendation_color(recommendation):
    if recommendation == "STRONG BUY":
        return "#00B300"
    elif recommendation == "BUY":
        return "#66CC66"
    elif recommendation == "HOLD":
        return "#FFD700"
    elif recommendation == "SELL":
        return "#FFA500"
    else:
        return "#FF4C4C"

def get_recommendation_emoji(recommendation):
    emojis = {
        "STRONG BUY": "✅",
        "BUY": "👍",
        "HOLD": "⏸️",
        "SELL": "⚠️",
        "STRONG SELL": "⛔",
    }
    return emojis.get(recommendation, "")

# ==================== MAIN APPLICATION ====================
col1, col2 = st.columns([3, 1])

with col1:
    selected_stocks = st.text_input(
        "Enter stock symbol or company name",
        placeholder="e.g., TCS.NS, RELIANCE.NS, INFY.NS",
    )

with col2:
    lookahead_days = st.number_input(
        "Days ahead to check if price will rise",
        min_value=7,
        max_value=90,
        value=30,
    )

if selected_stocks:
    selected_stocks_original = selected_stocks

    try:
        # Stock validation
        data_load_state = st.info("⏳ Checking stock name and downloading data...")

        try:
            df_names = pd.read_csv("NAMES.csv")
            selected_stocks_upper = selected_stocks.upper()
            df_filtered = df_names[df_names["NAME"].str.contains(selected_stocks_upper)]

            try:
                selected_stocks = df_filtered.iloc[0, 1]
            except:
                selected_stocks = selected_stocks_upper
        except:
            selected_stocks = selected_stocks.upper()

        # Load data
        data = load_data(selected_stocks)

        if not data.empty:
            data_load_state.success("✅ Data downloaded successfully!")

            # Create features and labels
            with st.spinner("🔄 Preparing data and training the model..."):
                df_features = create_features(data.copy())
                df_with_target = create_target(
                    df_features.copy(), lookahead_days=lookahead_days
                )

                # Columns used for the model (we remove raw columns like Date, Open, etc.)
                feature_columns = [
                    col
                    for col in df_with_target.columns
                    if col
                    not in [
                        "Date",
                        "Close",
                        "Future_Price",
                        "Buy_Signal",
                        "Open",
                        "High",
                        "Low",
                        "Volume",
                        "Adj Close",
                    ]
                ]

                X = df_with_target[feature_columns].fillna(0)
                y = df_with_target["Buy_Signal"]

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train ensemble
                xgb_model, gb_model, rf_model = train_ensemble_model(
                    X_train_scaled, y_train
                )

                # Predictions for test set
                xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
                gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
                rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]

                # Ensemble prediction (average probability of "price will go up")
                ensemble_pred = (xgb_pred + gb_pred + rf_pred) / 3

                # Current day prediction
                latest_features = X.iloc[-1:].values
                latest_features_scaled = scaler.transform(latest_features)

                current_buy_prob_xgb = xgb_model.predict_proba(
                    latest_features_scaled
                )[0, 1]
                current_buy_prob_gb = gb_model.predict_proba(
                    latest_features_scaled
                )[0, 1]
                current_buy_prob_rf = rf_model.predict_proba(
                    latest_features_scaled
                )[0, 1]

                current_confidence = (
                    current_buy_prob_xgb
                    + current_buy_prob_gb
                    + current_buy_prob_rf
                ) / 3

            # ==================== DISPLAY RESULTS ====================
            st.markdown("---")

            # 1. RAW DATA AND VISUALIZATION
            with st.expander("📊 Recent price history", expanded=False):
                st.subheader("Last 30 trading days")
                display_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                df_display = data[display_cols].tail(30).copy()
                # Format numeric columns to Indian Rupees where appropriate
                for col in ["Open", "High", "Low", "Close"]:
                    df_display[col] = df_display[col].apply(format_inr)
                st.dataframe(df_display, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Open"],
                        name="Opening Price",
                        mode="lines",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=data["Date"],
                        y=data["Close"],
                        name="Closing Price",
                        mode="lines",
                    )
                )
                fig.update_layout(
                    title="Stock Price Trend (2015 to today)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. PATTERN ANALYSIS
            st.subheader("🔍 Simple view of recent behaviour")
            pattern_analysis = analyze_price_patterns(df_features)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Latest closing price",
                    format_inr(pattern_analysis["current_price"]),
                )
            with col2:
                st.metric(
                    "Average price (last 50 days)",
                    format_inr(pattern_analysis["ma_50"]),
                )
            with col3:
                st.metric(
                    "Strength (RSI 14 days)",
                    f"{pattern_analysis['rsi']:.2f}",
                    help="Above 70 = may be overbought, below 30 = may be oversold.",
                )
            with col4:
                st.metric(
                    "Price swing (20-day volatility)",
                    f"{pattern_analysis['volatility']:.2f}%",
                    help="Rough idea of how much the price moves day-to-day.",
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                friendly_trend = (
                    "Overall upward trend"
                    if pattern_analysis["price_trend"] == "UPTREND"
                    else "Overall downward trend"
                )
                st.info(f"📈 Trend: {friendly_trend}")
            with col2:
                volume_text = (
                    "Higher than usual trading activity"
                    if pattern_analysis["volume_trend"] == "HIGH"
                    else "Normal trading activity"
                )
                st.info(f"📦 Trading activity: {volume_text}")
            with col3:
                momentum_status = (
                    "Price has been rising in recent days"
                    if pattern_analysis["momentum"] > 0
                    else "Price has been falling in recent days"
                )
                st.info(f"🎯 Short-term move: {momentum_status}")

            # 3. MODEL PREDICTIONS AND RECOMMENDATIONS
            st.markdown("---")
            st.subheader("🤖 Model suggestion (simple BUY / SELL view)")

            rsi = pattern_analysis["rsi"]

            if current_confidence >= 0.75 and rsi < 70:
                recommendation = "STRONG BUY"
                reason = (
                    "Model is very sure the price will go up soon and the stock "
                    "does not look overbought."
                )
            elif current_confidence >= 0.65 and rsi < 70:
                recommendation = "BUY"
                reason = (
                    "Model expects the price to increase and the strength level looks healthy."
                )
            elif 0.45 <= current_confidence < 0.55:
                recommendation = "HOLD"
                reason = (
                    "Signals are mixed. It may be better to wait and watch for clearer direction."
                )
            elif current_confidence >= 0.35 and rsi > 30:
                recommendation = "SELL"
                reason = (
                    "Model sees a fair chance of price going down from here."
                )
            else:
                recommendation = "STRONG SELL"
                reason = (
                    "Model is quite confident the price may fall and indicators are not favourable."
                )

            recommendation_color = get_recommendation_color(recommendation)
            recommendation_emoji = get_recommendation_emoji(recommendation)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    f"""
                <div style="background-color: {recommendation_color}; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h2 style="color: white; margin: 0;">
                        {recommendation_emoji} {recommendation}
                    </h2>
                    <p style="color: white; margin: 5px 0;">{reason}</p>
                    <p style="color: white; margin: 5px 0; font-size: 12px;">
                        <b>Note:</b> This is not financial advice. Use it only as an additional input
                        along with your own research.
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.metric(
                    "Model's confidence that price will rise",
                    f"{current_confidence*100:.2f}%",
                )

            st.markdown("#### Confidence from each model")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "XGBoost model (tree-based)",
                    f"{current_buy_prob_xgb*100:.2f}%",
                )
            with col2:
                st.metric(
                    "Gradient Boosting model",
                    f"{current_buy_prob_gb*100:.2f}%",
                )
            with col3:
                st.metric(
                    "Random Forest model",
                    f"{current_buy_prob_rf*100:.2f}%",
                )

            # Profitability approximation
            forecast_data = df_features.tail(1).copy()
            current_price = pattern_analysis["current_price"]

            if pattern_analysis["price_trend"] == "UPTREND":
                forecasted_price = current_price * (1 + (current_confidence * 0.1))
            else:
                forecasted_price = current_price * (1 - ((1 - current_confidence) * 0.1))

            price_change_pct, potential_profit = calculate_profitability(
                current_price, forecasted_price, current_confidence
            )

            st.markdown(
                f"#### Simple estimate for next {lookahead_days} days (based on current trend)"
            )
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Expected price movement",
                    f"{price_change_pct:.2f}%",
                    delta=format_inr(forecasted_price - current_price),
                )
            with col2:
                st.metric(
                    "Potential profit score",
                    f"{potential_profit:.2f}%",
                    help="Rough profit estimate = expected move × model confidence.",
                )
            with col3:
                profit_status = (
                    "Looks potentially profitable ✅"
                    if potential_profit > 0
                    else "Looks risky ⚠️"
                )
                st.metric("Overall view", profit_status)

            # 4. SHAP ANALYSIS (BEGINNER-FRIENDLY TEXT)
            st.markdown("---")
            st.subheader("🔬 Why did the model say BUY / SELL?")

            st.markdown(
                """
SHAP helps answer a simple question:

> **For today's prediction, which factors pushed the model more towards BUY, and which pushed it towards SELL?**

Think of it as a detailed score card:
- Positive SHAP value → pushes prediction **towards price going up (BUY)**  
- Negative SHAP value → pushes prediction **towards price going down (SELL)**
"""
            )

            with st.spinner("Calculating factor importance (SHAP values)..."):
                # Use XGBoost model for SHAP
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_train_scaled[:500])

                # ---- Summary Plot (which factors matter most overall) ----
                st.markdown("#### Overall most important factors")
                fig_shap_summary, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_train_scaled[:500],
                    feature_names=feature_columns,
                    plot_type="bar",
                    show=False,
                )
                st.pyplot(fig_shap_summary)
                plt.close(fig_shap_summary)

                # ---- Force Plot (today's explanation) ----
                st.markdown("#### For today: which factors pushed the decision?")
                st.caption(
                    "Blue areas push towards BUY (price up). "
                    "Red areas push towards SELL (price down)."
                )
                try:
                    plt.clf()
                    shap.force_plot(
                        explainer.expected_value,
                        explainer.shap_values(latest_features_scaled)[0],
                        latest_features_scaled[0],
                        feature_names=feature_columns,
                        matplotlib=True,
                        show=False,
                        figsize=(14, 3),
                    )
                    fig_force = plt.gcf()
                    st.pyplot(fig_force, bbox_inches="tight", use_container_width=True)
                    plt.close(fig_force)
                except Exception as e:
                    st.warning(f"Could not show detailed SHAP force plot: {str(e)}")

                # ---- Dependence plots (top 3 factors) ----
                st.markdown("#### How the top 3 factors affect the model")
                st.caption(
                    "Each chart shows how changing that factor changes the model's view about price going up."
                )

                feature_importance = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(feature_importance)[-3:][::-1]

                cols = st.columns(3)
                for idx, feature_idx in enumerate(top_features_idx):
                    with cols[idx]:
                        fig_dep, ax = plt.subplots(figsize=(6, 4))
                        shap.dependence_plot(
                            feature_idx,
                            shap_values,
                            X_train_scaled[:500],
                            feature_names=feature_columns,
                            ax=ax,
                            show=False,
                        )
                        st.pyplot(fig_dep)
                        plt.close(fig_dep)

            # 5. MODEL PERFORMANCE METRICS
            st.markdown("---")
            st.subheader("📈 How well does the model do on past data?")

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            test_pred = xgb_model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred)
            recall = recall_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Overall correctness (Accuracy)",
                    f"{accuracy*100:.2f}%",
                    help="Out of 100 past cases, how many did the model classify correctly?",
                )
            with col2:
                st.metric(
                    "Correct BUY calls (Precision)",
                    f"{precision*100:.2f}%",
                    help="Out of all BUY signals given, how many were actually right?",
                )
            with col3:
                st.metric(
                    "Missed BUY cases (Recall)",
                    f"{recall*100:.2f}%",
                    help="Out of all times price went up, how many did the model catch as BUY?",
                )
            with col4:
                st.metric(
                    "Balance score (F1)",
                    f"{f1*100:.2f}%",
                    help="Single score balancing correct BUY calls and missed cases.",
                )

            # 6. TECHNICAL INDICATORS VISUALIZATION
            st.markdown("---")
            st.subheader("📊 Key indicator charts (for curious users)")

            col1, col2 = st.columns(2)

            recent = df_features.tail(100)

            with col1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=recent["Date"],
                        y=recent["RSI"],
                        name="RSI",
                        mode="lines",
                    )
                )
                fig_rsi.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Overbought (70)",
                )
                fig_rsi.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Oversold (30)",
                )
                fig_rsi.update_layout(
                    title="RSI (shows if price may be overbought or oversold)",
                    height=400,
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

            with col2:
                fig_bb = go.Figure()
                fig_bb.add_trace(
                    go.Scatter(
                        x=recent["Date"],
                        y=recent["Close"],
                        name="Close Price",
                        mode="lines",
                    )
                )
                fig_bb.add_trace(
                    go.Scatter(
                        x=recent["Date"],
                        y=recent["BB_Upper"],
                        name="Upper Band",
                        mode="lines",
                        line=dict(color="rgba(255,0,0,0.3)"),
                    )
                )
                fig_bb.add_trace(
                    go.Scatter(
                        x=recent["Date"],
                        y=recent["BB_Lower"],
                        name="Lower Band",
                        mode="lines",
                        line=dict(color="rgba(0,255,0,0.3)"),
                    )
                )
                fig_bb.update_layout(
                    title="Bollinger Bands (price range around its average)",
                    height=400,
                )
                st.plotly_chart(fig_bb, use_container_width=True)

            # 7. FORECAST USING PROPHET
            st.markdown("---")
            st.subheader("🔮 Simple 1-year price outlook (trend only)")

            with st.spinner("Building a simple trend-based forecast..."):
                df_prophet = data[["Date", "Close"]].copy()
                df_prophet.columns = ["ds", "y"]

                m = Prophet(yearly_seasonality=True, daily_seasonality=False)
                m.fit(df_prophet)

                future_days = 365
                future = m.make_future_dataframe(periods=future_days)
                forecast = m.predict(future)

                fig_forecast = go.Figure()
                fig_forecast.add_trace(
                    go.Scatter(
                        x=df_prophet["ds"],
                        y=df_prophet["y"],
                        mode="markers",
                        name="Historical",
                        marker=dict(size=3),
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat"],
                        mode="lines",
                        name="Forecast",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat_upper"],
                        mode="lines",
                        name="Upper range",
                        line=dict(width=0),
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(0,100,80,0.2)",
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat_lower"],
                        mode="lines",
                        name="Lower range",
                        line=dict(width=0),
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(0,100,80,0.2)",
                    )
                )
                fig_forecast.update_layout(
                    title="Estimated price trend for next 1 year (₹)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=500,
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                with st.expander(
                    "See detailed trend and seasonal patterns (for advanced users)",
                    expanded=False,
                ):
                    fig2 = m.plot_components(forecast)
                    st.pyplot(fig2, use_container_width=True)

        else:
            st.error("❌ Could not download data. Please check the stock symbol.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info(
            "Please ensure all required libraries are installed: "
            "streamlit, yfinance, prophet, scikit-learn, xgboost, shap, plotly"
        )

else:
    st.info("👈 Enter a stock symbol to get started")
