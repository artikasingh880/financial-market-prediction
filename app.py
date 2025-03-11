import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # If using a trained ML model
from prophet import Prophet  # If using Prophet for forecasting

import joblib

# Load model with full path
model = joblib.load(r"C:\Users\artik\financial_model.pkl")  


# Load the trained model (example for ML models)
model = joblib.load("financial_model.pkl")  # Change as per your model

# Streamlit UI
st.title("📈 Financial Market Prediction System")

# User Input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-03-07"))

# Fetch Data
if st.button("Predict"):
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found! Check the stock symbol or date range.")
    else:
        st.success("Data Loaded Successfully!")

        # Display Data
        st.write(df.tail())

        # Plot Close Price
        st.subheader("Stock Closing Price")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # ML Model Prediction (if applicable)
        # X_test = feature_engineering(df)  # Convert data to feature format
        # prediction = model.predict(X_test)
        # st.write("Predicted Value:", prediction[-1])

        # Prophet Forecasting (if applicable)
        df_prophet = df.reset_index()[["Date", "Close"]]
        df_prophet.columns = ["ds", "y"]
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)
        future = prophet_model.make_future_dataframe(periods=30)  # Predict next 30 days
        forecast = prophet_model.predict(future)
        st.subheader("Forecasted Prices")
        st.line_chart(forecast.set_index("ds")["yhat"])


