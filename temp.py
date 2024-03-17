import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from ta import ema, rsi, bollinger_bands  # Technical Analysis library

# Define the URLs for sectors
sector_url = "https://etfdb.com/etfs/sector/#sector-power-rankings__return-leaderboard&sort_name=aum_position&sort_order=asc&page=1"


# Function to scrape top 4 sectors from a URL
def scrape_top_sectors(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('table', id="sector-power-rankings")
  tbody = table.find('tbody')
  sectors = []
  for row in tbody.find_all('tr'):  # Get the first 4 rows after the header
    sector = row.find('td', attrs={'data-th': 'Sector'}).find('a').text.strip()
    sectors.append(sector)
  return sectors


# Scrape top 4 sectors
top_sectors = scrape_top_sectors(sector_url)
print(top_sectors)


def scrape_etf_symbols(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('table', id='etfs')
  if not table:
    return []

  # Extract ETF symbols from table rows
  symbols = []
  tbody = table.find('tbody')
  if tbody:
    for row in tbody.find_all('tr'):
      symbol_cell = row.find('td', attrs={'data-th': 'Symbol'})
      if symbol_cell:
        symbol = symbol_cell.find('a').text.strip()
        symbols.append(symbol)
  return symbols


# Define the sector-specific URL
sector_specific_url = "https://etfdb.com/etfs/sector/{sector}/?search[inverse]=false&search[leveraged]=false#etfs__returns&sort_name=assets_under_management&sort_order=desc&page=1"
# Scrape ETF symbols for each of the top 4 sectors
sector_etfs = []
for sector in top_sectors:
  url = sector_specific_url.format(sector=sector)
  sector_etfs.extend(scrape_etf_symbols(url))


def calculate_combined_risk_score(data):
  # Calculate individual risk scores
  max_sd = data["Close"].std().max(
  )  # Find max standard deviation across analyzed stocks (replace with appropriate logic if needed)
  max_debt_to_equity = data.get_financials()["debtToEquity"].max(
  )  # Find max debt-to-equity ratio
  min_market_cap = data["market_cap"].min()  # Find min market cap

  vscore = (data["Close"].std() / max_sd)
  lscore = (data["debtToEquity"] / max_debt_to_equity)
  mcap_score = ((1 / data["market_cap"]) / (1 / min_market_cap))

  # Create a NumPy array of individual risk scores
  risk_scores = np.array([vscore, lscore, mcap_score])

  # Calculate combined risk score using dot product
  combined_risk_score = np.dot([0.4, 0.3, 0.3], risk_scores)

  return combined_risk_score


def calculate_combined_trend_score(data):
  # Calculate technical indicators
  ema20 = ema(data["Close"], window=20)
  rsi14 = rsi(data["Close"], window=14)
  bollinger_bands = bollinger_bands(data["Close"], window=20, std=2)
  upper_band = bollinger_bands[0]
  lower_band = bollinger_bands[1]

  # Calculate technical indicator scores (implement your scoring logic here)
  ema_score = 1.0 if (data["Close"] > ema20).mean(
  ) else 0.0  # Simple scoring based on EMA trend (rising: 1, falling: 0)
  rsi_score = 0.0 if rsi14.iloc[-1] < 30 else (
      1.0 - (rsi14.iloc[-1] - 30) / 70
  ) if rsi14.iloc[
      -1] > 70 else 0.5  # Score based on RSI value (oversold: 1, overbought: 0, neutral: 0.5)
  bollinger_score = 1.0 if data["Close"].iloc[-1] > upper_band.iloc[
      -1] else 0.0 if data["Close"].iloc[-1] < lower_band.iloc[
          -1] else 0.5  # Score based on Bollinger Band position (breakout: 1, breakdown: 0, within bands: 0.5)

  # Combine individual scores using weighted summation
  # Define weights if not provided (adjust these based on your priorities)
  weights = weights if weights else np.array([0.3, 0.3, 0.4])

  combined_score = np.dot(weights,
                          np.array([ema_score, rsi_score, bollinger_score]))

  # Analyze technical indicators
  overbought_oversold = "Neutral"
  if rsi14.iloc[-1] > 70:
    overbought_oversold = "Overbought"
  elif rsi14.iloc[-1] < 30:
    overbought_oversold = "Oversold"

  price_correction = "Neutral"
  if bollinger_score == 1.0:
    price_correction = "Potential Upward Breakout"
  elif bollinger_score == 0.0:
    price_correction = "Potential Downward Breakdown"

  potential_movement = "Neutral"
  if combined_score > 0.5 and (data["Close"] > ema20).mean():
    potential_movement = "Potential Price Increase"
  elif combined_score < 0.5 and (data["Close"] < ema20).mean():
    potential_movement = "Potential Price Decrease"

  trend = "Upward" if (data["Close"] > ema20).mean() else "Downward" if (
      data["Close"] < ema20).mean() else "Neutral"

  # Create and return results dictionary
  analysis_results = {
      "cs": combined_score,
      "obs": overbought_oversold,
      "Price Correction": price_correction,
      "pm": potential_movement,
      "Trend": trend
  }

  return analysis_results


# Function to calculate indicators for a symbol
def calculate_indicators(symbol):
  # Download historical data
  start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
  end_date = datetime.now().strftime("%Y-%m-%d")
  data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
  # Initialize a new dictionary to store the results
  new_data = {}
  # Create a Ticker object for a specific symbol
  ticker = yf.Ticker(symbol)
  print(ticker.news)
  for k, v in ticker.info.items():
    print(k, '\t', v, '\n')

  new_data['longName'] = ticker.info['longName']
  new_data['L.Year'] = data['Close'].iloc[0]
  new_data['Today'] = data['Close'].iloc[-1]
  new_data['risk_score'] = calculate_combined_risk_score(data)
  # Get average analyst price target (informational)
  new_data['analyst_target'] = data.get("Analyst Ratings",
                                        {}).get("Mean Target Price", None)

  trend_anlaysis = calculate_combined_trend_score(data)
  new_data['trend_score'] = trend_anlaysis['cs']
  new_data['obs'] = trend_anlaysis['obs']
  new_data['movement'] = trend_anlaysis['pm']
  new_data['inverse_risk_score'] = 1 / new_data['risk_score']
  new_data['1Y'] = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) /
                    data['Close'].iloc[0]) * 100
  new_data['30 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[30]) /
                        data['Close'].iloc[30]) * 100
  new_data['7 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[7]) /
                       data['Close'].iloc[7]) * 100

  # Shift the data by one day
  shifted_data = data['Close'].shift(1)
  new_data['MA_15'] = shifted_data.rolling(
      window=15).mean().iloc[-1] / data['Close'].iloc[-1]
  # Now apply the rolling window
  rolling_max = shifted_data.rolling(window=15).max()
  rolling_min = shifted_data.rolling(window=15).min()
  # Continue with your calculations
  new_data['Max Up_15'] = rolling_max.iloc[-1] / data['Close'].iloc[-1]
  new_data['Max Down_15'] = rolling_min.iloc[-1] / data['Close'].iloc[-1]

  # Calculate Avg Volume_15
  new_data['Avg Volume_15'] = data['Volume'].rolling(window=15).mean().iloc[-1]
  # Calculate Volatility
  rolling_returns_30 = data['Close'].pct_change().rolling(
      window=30).mean().iloc[-1]
  volatility_30 = data['Close'].pct_change().rolling(window=30).std().iloc[-1]
  weights_30 = 1 / volatility_30
  new_data['vs_30'] = weights_30 * rolling_returns_30

  # Calculate composite score
  weights = [0.20, 0.20, 0.10, 0.20, 0.10, 0.10, 0.05,
             0.05]  # adjust weights according to your preferences
  factors = [
      '7 day', '30 day', 'MA_15', 'Max Up_15', 'Max Down_15', 'vs_30',
      'inverse_risk_score', 'trend_score'
  ]
  # print(new_data)
  new_data['Composite Score'] = np.dot(
      [new_data[factor] for factor in factors], weights)

  # Add trend
  if new_data['MA_15'] < 1:
    new_data['Trend'] = 'Uptrend'
  elif new_data['MA_15'] > 1:
    new_data['Trend'] = 'Downtrend'
  else:
    new_data['Trend'] = 'Sideways'

  # Add analysis
  if new_data['Max Up_15'] > 1:
    new_data['Analysis'] = 'Bear'
  elif new_data['Max Down_15'] < 1:
    new_data['Analysis'] = 'Bull'
  else:
    new_data['Analysis'] = 'Hold'
  return new_data


# Calculate indicators for all symbols
all_data = {}
# for symbol in all_symbols:
#     all_data[symbol] = calculate_indicators(symbol)

all_data[all_symbols[0]] = calculate_indicators(all_symbols[0])

# print(all_data)

# Convert each dictionary in all_data to a DataFrame
all_data = {symbol: pd.DataFrame([data]) for symbol, data in all_data.items()}
# Combine all data into a single DataFrame
combined_data = pd.concat(all_data)
# Sort by composite score
ranked_data = combined_data.sort_values(by='Composite Score', ascending=False)

ranked_data = ranked_data.round(2)

# Get the composite scores for the top 5 ETFs
top_scores = ranked_data['Composite Score'].nlargest(5)
# Normalize the scores so they add up to 1
weights = top_scores / top_scores.sum()
# Calculate the amount to invest in each ETF
investment = 10000  # your total investment
ranked_data.loc[weights.index, 'Portfolio'] = 100 * weights
ranked_data.loc[weights.index, 'Amount'] = investment * weights

# Print the DataFrame in the console
# print(ranked_data)

# Get the current date
current_date = datetime.today().strftime('%Y-%m-%d')
# Save the DataFrame as a CSV file with the current date in the filename
ranked_data.to_csv(f'ranked_data_{current_date}.csv')


# Function to backtest the top 5 ETFs
def backtest(symbols):
  # Download historical data for the past year
  data = yf.download(symbols + ['SPY'], period="1y")['Close']
  # Calculate daily returns
  returns = data.pct_change()
  # Initialize portfolio with equal weights
  weights = pd.Series(1 / len(symbols), index=symbols)
  # Initialize portfolio values for each ETF
  portfolio_values = pd.DataFrame(0, index=data.index, columns=symbols)
  portfolio_values.iloc[0] = (10000 * weights)  # initial investment
  # Backtest
  for i in range(1, len(data)):
    # Update portfolio values
    portfolio_values.iloc[i] = portfolio_values.iloc[i - 1] * (
        1 + returns.iloc[i][symbols])
    # Rebalance every 180 days
    if i % 15 == 0:
      # Calculate new weights based on composite scores
      scores = pd.Series({
          symbol:
          calculate_indicators(symbol)['Composite Score']
          for symbol in symbols
      })
      weights = scores / scores.sum()
      # Rebalance portfolio values
      total_value = portfolio_values.iloc[i].sum()
      portfolio_values.iloc[i] = total_value * weights
  # Calculate final values and % growth
  final_values = portfolio_values.iloc[-1]
  growth = (final_values / (10000 / len(symbols)) - 1) * 100
  # Create a DataFrame with the results
  results = pd.DataFrame({
      'Symbol': symbols,
      'Final Value': final_values,
      '% Growth': growth
  })
  # Print the results
  print(results)


top_etfs = ranked_data.index.get_level_values(0)[:5]
print(list(top_etfs))
backtest(list(top_etfs))
