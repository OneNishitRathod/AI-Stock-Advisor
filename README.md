# Real-Time AI Stock Advisor

## Overview
The Real-Time AI Stock Advisor is a Streamlit-based web application that provides stock market insights using real-time data, technical analysis indicators, and AI-generated insights. It fetches historical stock data, calculates key technical indicators (EMA, Bollinger Bands, RSI, MACD), and visualizes them using interactive charts. The app also integrates an AI model (DeepSeek) to generate insights based on the latest stock trends.

## Features

- Fetch Real-Time Stock Data using the yfinance library.

- Technical Analysis Indicators:

    - Exponential Moving Average (EMA)

    - Bollinger Bands

    - Relative Strength Index (RSI)

    - Moving Average Convergence Divergence (MACD)

- Interactive Visualizations powered by plotly:

    - Candlestick Chart

    - Volume Chart

    - RSI Chart

    - MACD Chart

- AI-Generated Insights using deepseek-r1:1.5b via a local API.

- User-Friendly Streamlit UI for easy interaction.

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.12.18

- pip (Python package manager)

- Streamlit

- Required dependencies (see below)

#### Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-ai-stock-advisor.git
cd real-time-ai-stock-advisor
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Start the Application
```bash
streamlit run app.py
```

### API Integration
This application integrates with the DeepSeek AI model via a local API endpoint:
```bash
curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model": "deepseek-r1:1.5b", "prompt": "How are you today?", "stream": false}'
```
Ensure that the DeepSeek model is running locally before launching the app.

## Usage

1. Enter a stock ticker (e.g., AAPL) in the sidebar.

2. Select a time period (1mo, 3mo, 6mo, etc.).

3. Choose a data interval (1d, 1wk, 1mo).

4. Click Analyze to fetch stock data and generate insights.

5. View:
    - Stock Charts (Candlestick, Volume, RSI, MACD)

    - Technical Indicators

    - AI-Generated Insights

## Technologies Used

Frontend: Streamlit

Data Processing: Pandas, NumPy, yFinance

Visualization: Plotly

AI Model: DeepSeek R1:1.5B Model

Backend: Python, Requests

## License

This project is open-source and available under the MIT License.

## Contributors

- Nishit Rathod