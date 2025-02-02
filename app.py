import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    
    :param ticker: Stock ticker symbol.
    :param period: Data period (e.g., '6mo' for 6 months).
    :param interval: Data interval (e.g., '1d' for daily).
    :return: DataFrame with historical stock data.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
    return df

def calculate_technical_indicators(df: pd.DataFrame, ema_span: int = 20, bollinger_window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Calculate EMA and Bollinger Bands.
    
    :param df: DataFrame with stock data.
    :param ema_span: Span for the Exponential Moving Average.
    :param bollinger_window: Window size for Bollinger Bands.
    :param num_std: Number of standard deviations for the upper/lower bands.
    :return: DataFrame with new columns for EMA, upper band, and lower band.
    """
    df = df.copy()
    
    # Calculate EMA
    df['EMA'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
    
    # Calculate Simple Moving Average (SMA) for Bollinger Bands
    df['SMA'] = df['Close'].rolling(window=bollinger_window).mean()
    # Calculate standard deviation over the same window
    df['STD'] = df['Close'].rolling(window=bollinger_window).std()
    # Calculate upper and lower Bollinger Bands
    df['Upper Band'] = df['SMA'] + num_std * df['STD']
    df['Lower Band'] = df['SMA'] - num_std * df['STD']
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal Line'] = compute_macd(df['Close'])
    
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series) -> tuple:
    short_ema = series.ewm(span=12, adjust=False).mean()
    long_ema = series.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_llm_insights(ticker: str, df: pd.DataFrame) -> str:
    """
    Integrate with the LLM (Ollama) API to generate insights based on the technical analysis.
    This version uses the provided API call logic.
    
    :param ticker: Stock ticker symbol.
    :param df: DataFrame with stock data and technical indicators.
    :return: Generated insights as a string.
    """
    # Prepare a prompt based on the latest available data
    latest_data = df.iloc[-1]
    prompt = (
        f"Provide a detailed analysis for {ticker} based on the following data:\n"
        f"Latest Close Price: {latest_data['Close']:.2f}\n"
        f"20-day EMA: {latest_data['EMA']:.2f}\n"
        f"Upper Bollinger Band: {latest_data['Upper Band']:.2f}\n"
        f"Lower Bollinger Band: {latest_data['Lower Band']:.2f}\n\n"
        "What potential trends or trading signals can be observed?"
    )
    
    # Use the provided logic for the API call with the new model and request format
    data_payload = {
        "model": "deepseek-r1:1.5b",  # Updated model version
        "prompt": prompt,  # Directly using prompt
        "stream": False
    }
    api_url = "http://localhost:11434/api/generate"  # Use your local Ollama endpoint
    
    try:
        response = requests.post(api_url, json=data_payload, timeout=300)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()
        # Extract the AI reply from the response JSON
        ai_reply = response_json.get("response", "No insights returned.")  # Updated response key
    except Exception as e:
        st.error(f"Error contacting LLM API: {e}")
        ai_reply = "Unable to generate insights at this time."
    
    return ai_reply

def plot_stock_data(df: pd.DataFrame, ticker: str):
    """
    Create an interactive Plotly chart with stock price, EMA, and Bollinger Bands.
    
    :param df: DataFrame with stock data and technical indicators.
    :param ticker: Stock ticker symbol.
    :return: Plotly Figure.
    """
    fig = go.Figure()
    
    # Plot the closing price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    
    # Plot EMA
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA (20)'))
    
    # Plot Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Band', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Band', line=dict(dash='dash')))
    
    # Shade the Bollinger Bands area
    fig.add_trace(go.Scatter(
        x=np.concatenate([df.index, df.index[::-1]]),
        y=pd.concat([df['Upper Band'], df['Lower Band'][::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Bollinger Band Area'
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price and Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_volume(df: pd.DataFrame, ticker: str):
    fig = go.Figure(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    fig.update_layout(title=f"{ticker} Trading Volume", xaxis_title="Date", yaxis_title="Volume", template="plotly_white")
    return fig

def plot_rsi(df: pd.DataFrame, ticker: str):
    fig = go.Figure(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{ticker} RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI", template="plotly_white")
    return fig

def plot_macd(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], mode='lines', name='Signal Line'))
    fig.update_layout(title=f"{ticker} MACD Indicator", xaxis_title="Date", yaxis_title="MACD", template="plotly_white")
    return fig

def plot_candlestick(df: pd.DataFrame, ticker: str):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    return fig

# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.title("Real-Time AI Stock Advisor")
    st.write("This app fetches real-time stock data, performs technical analysis (EMA and Bollinger Bands), "
             "and generates insights using an LLM (Ollama).")
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
    period = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "5y"], index=2)
    interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    
    # Button to trigger analysis
    if st.sidebar.button("Analyze"):
        # Fetch data
        with st.spinner("Fetching stock data..."):
            df = fetch_stock_data(ticker, period, interval)
        
        if df is not None and not df.empty:
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Plot the stock data and technical indicators
            st.subheader(f"{ticker} Stock Chart")
            fig = plot_stock_data(df, ticker)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Trading Volume")
            st.plotly_chart(plot_volume(df, ticker), use_container_width=True)
            st.subheader("RSI Indicator")
            st.plotly_chart(plot_rsi(df, ticker), use_container_width=True)
            st.subheader("MACD Indicator")
            st.plotly_chart(plot_macd(df, ticker), use_container_width=True)
            st.subheader("Candlestick Chart")
            st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)

            # Get LLM insights using the updated API call logic
            st.subheader("AI-Generated Insights")
            with st.spinner("Generating insights..."):
                insights = get_llm_insights(ticker, df)
            st.write(insights)
            
            # Display raw data option
            with st.expander("View Raw Data"):
                st.dataframe(df.tail(20))
        else:
            st.error("No data to display.")

if __name__ == "__main__":
    main()