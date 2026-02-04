import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

LOOKBACK = 100

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.header("📈 Stock Market Predictor")

# Load model (keep file in same folder)
model = load_model("gold_lstm_model.keras")

# User input
stock = st.text_input("Enter Stock Symbol", "GOOG").strip()

# Date range
start = "2015-01-01"

# IMPORTANT: yfinance 'end' is EXCLUSIVE.
# Using (today + 1) ensures we definitely include the latest available daily candle.
end = date.today() + timedelta(days=1)

# Download stock data
data = yf.download(stock, start=start, end=end, progress=False)

st.subheader("Stock Data (Raw)")
st.write(data)

# ---------- Validations ----------
if data is None or data.empty:
    st.error(
        "No data returned from Yahoo Finance.\n\n"
        "Possible reasons:\n"
        "- Invalid symbol\n"
        "- Temporary Yahoo/yfinance issue\n"
        "- Network restriction\n\n"
        "Try: AAPL, MSFT, GOOG, TSLA"
    )
    st.stop()

# Robust close extraction
close_col = data.get("Close", None)
if close_col is None:
    st.error("Downloaded data does not contain 'Close' column.")
    st.stop()

# If Close is a DataFrame (MultiIndex case), select first column
if isinstance(close_col, pd.DataFrame):
    close_series = close_col.iloc[:, 0]
else:
    close_series = close_col

# Drop NaNs just in case
close_series = close_series.dropna()

if len(close_series) < LOOKBACK + 2:
    st.error(
        f"Not enough data to run this model.\n\n"
        f"Need at least {LOOKBACK + 2} closing prices, got {len(close_series)}."
    )
    st.stop()

# Convert to DataFrame for sklearn
close_df = pd.DataFrame(close_series.values, columns=["Close"])

# Train-test split (ensure train is not empty)
split_idx = int(len(close_df) * 0.80)
if split_idx < 1:
    split_idx = max(1, len(close_df) - 1)

data_train = close_df.iloc[:split_idx].copy()
data_test = close_df.iloc[split_idx:].copy()

if data_train.empty:
    st.error("Training split became empty. Increase date range or use a symbol with more history.")
    st.stop()

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

# Build test window with past LOOKBACK days
past_lookback = data_train.tail(LOOKBACK)
data_test_full = pd.concat([past_lookback, data_test], ignore_index=True)

data_test_scaled = scaler.transform(data_test_full)

# ---------------- Moving Averages ---------------- #
st.subheader("Price vs MA50")
ma_50 = close_series.rolling(50).mean()

fig1 = plt.figure(figsize=(8, 6))
plt.plot(close_series.values, label="Price")
plt.plot(ma_50.values, label="MA50")
plt.legend()
st.pyplot(fig1)
plt.close(fig1)

st.subheader("Price vs MA50 vs MA100")
ma_100 = close_series.rolling(100).mean()

fig2 = plt.figure(figsize=(8, 6))
plt.plot(close_series.values, label="Price")
plt.plot(ma_50.values, label="MA50")
plt.plot(ma_100.values, label="MA100")
plt.legend()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Price vs MA100 vs MA200")
ma_200 = close_series.rolling(200).mean()

fig3 = plt.figure(figsize=(8, 6))
plt.plot(close_series.values, label="Price")
plt.plot(ma_100.values, label="MA100")
plt.plot(ma_200.values, label="MA200")
plt.legend()
st.pyplot(fig3)
plt.close(fig3)

# ---------------- Prepare Test Data ---------------- #
x, y = [], []

for i in range(LOOKBACK, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - LOOKBACK : i])
    y.append(data_test_scaled[i, 0])

x = np.array(x)
y = np.array(y).reshape(-1, 1)

if x.shape[0] == 0:
    st.error(
        "No samples generated for prediction.\n\n"
        "This usually happens when test set is too small.\n"
        "Try using a longer date range or reduce LOOKBACK."
    )
    st.stop()

# Prediction on test set
predict_scaled = model.predict(x)
predict = scaler.inverse_transform(predict_scaled)
y_true = scaler.inverse_transform(y)

# ---------------- Final Comparison Plot ---------------- #
st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_true, label="Original Price")
plt.plot(predict, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
plt.close(fig4)

final_df = pd.DataFrame(
    {"Original_Price": y_true.flatten(), "Predicted_Price": predict.flatten()}
)
st.subheader("Final Original vs Predicted Prices")
st.write(final_df.tail(15))

last_available_date = data.index.max().date()
st.success(f"📅 Latest candle in downloaded data: {last_available_date}")

# ---------------- TODAY Prediction ---------------- #
# We predict the next candle after the latest available candle.
last_lookback_values = close_df.tail(LOOKBACK).values  # shape (LOOKBACK, 1)
last_lookback_scaled = scaler.transform(last_lookback_values)

X_today = np.array([last_lookback_scaled])  # shape (1, LOOKBACK, 1)
today_pred_scaled = model.predict(X_today)
today_pred = scaler.inverse_transform(today_pred_scaled)

today_date = date.today()

st.subheader("📌 Today's Predicted Price")
st.info(f"Predicted closing price for {today_date}: ₹{today_pred[0][0]:.2f}")

st.caption(
    "Note: The model predicts the next candle after the latest available candle from Yahoo Finance. "
    "If today is a holiday/weekend or today's candle isn't available yet, this is effectively the next trading day's prediction."
)

st.subheader("Stock Summary")
st.write(f"Total records: {len(close_series)}")
st.write(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
