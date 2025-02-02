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

### Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-ai-stock-advisor.git
cd ai-stock-advisor
```

### Install Ollama and Pull the deepseek-r1:8b model
  ```bash
  Navigate to https://ollama.com/
  ollama --version
  ollama pull deepseek-r1:8b
  ```
Note: If you prefer using WSL (Windows Subsystem for Linux), you can install Ollama using the shell script as described on their website.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start the Application
```bash
streamlit run app.py
```

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

1. Frontend: Streamlit

2. Data Processing: Pandas, NumPy, yFinance

3. Visualization: Plotly

4. AI Model: DeepSeek R1:1.5B Model

5. Backend: Python, Requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Nishit Rathod
