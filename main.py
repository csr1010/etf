import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from ta_py import ema, rsi, bands
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from newspaper import Article
from goose3 import Goose  # Import Goose class from goose3 module
from zenrows import ZenRowsClient
from etf_scraper import ETFScraper
from pyetfdb_scraper.etf import ETF,load_etfs

def get_sp500_tickers():
  """
    Extracts S&P 500 ticker symbols from Wikipedia using Beautiful Soup.
    """
  url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "lxml")

  # Updated table ID based on current HTML structure
  table = soup.find("table", {"id": "constituents"})
  tickers = [row.find("td").text.strip() for row in table.find_all("tr")[1:]]
  return tickers


# Define the URLs for sectors
sector_url = "https://etfdb.com/etfs/sector/#sector-power-rankings__return-leaderboard&sort_name=aum_position&sort_order=asc&page=1"

investment_styles = [
    'consistent-growth', 'aggressive-growth'
]
print('investment_styles', investment_styles)


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
# top_sectors = scrape_top_sectors(sector_url)
# print(top_sectors)


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
investment_style_url = 'https://etfdb.com/etfs/investment-style/{style}/?search[inverse]=false&search[leveraged]=false#etfs&sort_name=assets_under_management&sort_order=desc&page=1'
# Scrape ETF symbols for each of the top 4 sectors
etf_tickers = []
for style in investment_styles:
  url = investment_style_url.format(style=style)
  tickers = scrape_etf_symbols(url)
  for ticker in tickers:
    etf_tickers.append((style, ticker))

# all_symbols = list(set(etf_tickers + get_sp500_tickers())),

# all_symbols = get_sp100_tickers();


def get_overall_sentiment(positive_score, negative_score, neutral_score):
  total_score = positive_score + negative_score + neutral_score
  # Calculate the percentage of each sentiment score
  percent_positive = (positive_score / total_score) if total_score else 0
  percent_negative = (negative_score / total_score) if total_score else 0
  percent_neutral = (neutral_score / total_score) if total_score else 0

  # Define thresholds
  threshold_for_dominance = 0.5  # Adjusted to 50% for simplicity
  significant_neutral_threshold = 0.5  # If more than 50% of the sentiment is neutral

  # Determine overall sentiment
  if percent_neutral > significant_neutral_threshold:
    overall_sentiment = "mostly neutral"
  elif abs(
      percent_positive - percent_negative
  ) < threshold_for_dominance and percent_neutral < significant_neutral_threshold:
    overall_sentiment = "mixed"
  elif percent_positive > percent_negative:
    overall_sentiment = "positive"
  else:
    overall_sentiment = "negative"

  return overall_sentiment


def get_stock_performance_sentiment(symbol):
  nltk.download('vader_lexicon')
  nltk.download('punkt')
  analyzer = SentimentIntensityAnalyzer()

  # Initialize variables for sentiment scores
  positive_score = 0
  negative_score = 0
  neutral_score = 0

  stock_keywords = [
      "profit", "loss", "buy", "sell", "bullish", "bearish", "up", "down",
      "stocks", "etfs", "market", "investment"
  ]

  articles = ['https://stockinvest.us/stock/{symbol}/']

  client = ZenRowsClient("35c649685a0df9ea2b01ad6740fc8f0969dad1f0")

  for article in articles:
    url = article.format(symbol=symbol)
    response = client.get(url, params = {"js_render":"true","wait":"2000"})
    # print(response, url)
    extractor = Goose()
    extractCont = extractor.extract(raw_html=response.content.decode('utf-8'))
    text = extractCont.cleaned_text
    # print(text)
    sentences = nltk.sent_tokenize(text)
    relevant_sentences = [
        sentence for sentence in sentences
        if any(keyword in sentence.lower() for keyword in stock_keywords)
    ]

    for sentence in relevant_sentences:
      sentiment = analyzer.polarity_scores(sentence)
      # print("---", relevant_sentences, sentiment)
      positive_score += sentiment.get("pos", 0)
      negative_score += sentiment.get("neg", 0)
      neutral_score += sentiment.get("neu", 0)

  # Determine overall sentiment based on scores
  overall_sentiment = get_overall_sentiment(positive_score, negative_score,
                                            neutral_score)

  # Determine potential future performance based on sentiment
  future_performance = ""
  if overall_sentiment == "positive":
    future_performance = "potential upside"
  elif overall_sentiment == "negative":
    future_performance = "potential downside"

  # Calculate average sentiment scores
  total_articles = len(articles) if articles else 1  # Prevent division by zero
  average_positive = round(positive_score / total_articles, 2)
  average_negative = round(negative_score / total_articles, 2)
  average_neutral = round(neutral_score / total_articles, 2)
  print("---", symbol, positive_score, negative_score, neutral_score)
  print("---", symbol, average_positive, average_negative, average_neutral)
  return {
      "overall_sentiment": overall_sentiment,
      "future_performance": future_performance,
      "average_sentiment": {
          "positive": average_positive,
          "negative": average_negative,
          "neutral": average_neutral
      }
  }


def calculate_combined_risk_score(data, ticker):
  # Volatility score (standard deviation of closing prices)
  volatility_score = data["Close"].std()

  # Assuming financials is a dict-like object with a 'debtToEquity' key
  debt_to_equity = ticker.financials.get("debtToEquity", np.nan)
  leverage_score = debt_to_equity if not np.isnan(debt_to_equity) else np.nan

  # Assuming ticker.info is a dict-like object with a 'marketCap' key
  market_cap = ticker.info.get("marketCap", np.nan)
  # Market cap score (inversely proportional to market cap, ensuring market cap is not zero)
  mcap_score = (1 / market_cap) if market_cap != 0 else np.nan

  # Create a NumPy array of individual risk scores (handle potential NaNs)
  risk_scores = np.array([volatility_score, leverage_score, mcap_score])
  valid_risk_scores = risk_scores[~np.isnan(risk_scores)]  # Remove NaNs

  # Weighted scores - adjust based on available scores
  weights = np.array([0.4, 0.3, 0.3])[:len(valid_risk_scores)]
  # Normalize weights to ensure they sum up to 1
  normalized_weights = weights / np.sum(weights)

  # Calculate combined risk score using dot product
  combined_risk_score = np.dot(
      normalized_weights,
      valid_risk_scores) if valid_risk_scores.any() else np.nan

  return combined_risk_score


def calculate_combined_trend_score(data, ticker):
  ema_val = ema(data["Close"], 14)
  rsi_val = rsi(data["Close"], 14)
  bollinger_bands = bands(data["Close"], 14, 2)
  upper_band = bollinger_bands[0]
  lower_band = bollinger_bands[1]

  # Calculate technical indicator scores more appropriately
  ema_score = 1.0 if data["Close"].iloc[-1] > ema_val[-1] else 0.0
  rsi_score = 0.1 if rsi_val[-1] > 70 else 1.0 if rsi_val[-1] < 30 else 0.5
  bollinger_score = 1.0 if data["Close"].iloc[-1] > upper_band[
      -1] else 0.0 if data["Close"].iloc[-1] < lower_band[-1] else 0.5

  # Analyze technical indicators
  market_condition = "Neutral"
  # Adjust thresholds for RSI to provide a buffer zone
  rsi_overbought_threshold = 70
  rsi_oversold_threshold = 30

  trend = "Upward" if data["Close"].iloc[-1] > ema_val[
      -1] else "Downward" if data["Close"].iloc[-1] < ema_val[-1] else "Neutral"

  # Check for both conditions to agree on overbought/oversold status
  is_overbought = rsi_val[
      -1] > rsi_overbought_threshold and bollinger_score == 1.0
  is_oversold = rsi_val[-1] < rsi_oversold_threshold and bollinger_score == 0.0

  # Determine market condition based on refined criteria
  if is_overbought:
    # Check if recent trend supports the overbought condition
    if trend == "upward":
      market_condition = "Overbought - Caution"
    else:
      market_condition = "Overbought - Potential Reversal"
  elif is_oversold:
    # Check if recent trend supports the oversold condition
    if trend == "downward":
      market_condition = "Oversold - Caution"
    else:
      market_condition = "Oversold - Potential Reversal"

  # Create and return results dictionary
  analysis_results = {
      "rsi": rsi_score,
      "ema": ema_score,
      "market": market_condition,
      "trend": trend
  }

  return analysis_results

cache = {}

def get_etf_holdings(symbol):
  try:
    # Send a GET request to the URL
    url = f"https://etfdb.com/etf/{symbol}/#holdings"
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP request errors
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the holdings table by ID
    holdings_table = soup.find('table', {'id': 'etf-holdings'})

    # Initialize lists to hold symbols and weights
    symbols = []
    weights = []

    # Iterate through each row in the tbody of the holdings table
    for row in holdings_table.find('tbody').find_all('tr'):
        # Extract the symbol from the first td element
        symbol_text = row.find('td').text.strip()
        symbols.append(symbol_text)

        # Extract the weight from the last td element, remove '%' and convert to float
        weight_text = row.find_all('td')[-1].text.strip().replace('%', '')
        weights.append(float(weight_text))

    # Create a DataFrame with symbols and weights
    holdings_df = pd.DataFrame({
        'symbol': symbols,
        'weight': weights
    })

    return holdings_df

  except Exception as e:
      print(f"An error occurred: {e}")
      # Return an empty DataFrame if an error occurs
      return pd.DataFrame({'symbol': [], 'weight': []})
  

def calculate_overlap_percentage(s1, s2):
  if (s1 in cache):
    df1 = cache[s1]
  else:
    df1 = get_etf_holdings(s1)
    cache[s1] = df1
  
  if (s2 in cache):
    df2 = cache[s2]
  else:
    df2 = get_etf_holdings(s2)
    cache[s2] = df2
  
  merged_df = pd.merge(df1, df2, on='symbol', suffixes=('_df1', '_df2'))

  # Step 2: Calculate the minimum weight for each common symbol
  merged_df['min_weight'] = merged_df[['weight_df1', 'weight_df2']].min(axis=1)

  # Step 3: Sum of minimum weights for overlap calculation
  total_weighted_overlap = merged_df['min_weight'].sum()

  # Step 4: Total weight in the first DataFrame (change to '_df2' for the second DataFrame if needed)
  total_weight_df1 = df1['weight'].sum()
  total_weight_df2 = df2['weight'].sum()

  # Step 5: Calculate weighted overlap percentage
  o1 = (total_weighted_overlap / total_weight_df1)
  o2 = (total_weighted_overlap / total_weight_df2)

  print(s1, s2, o1, o2)
  return o1


# Function to calculate indicators for a symbol
def calculate_indicators(style, symbol):
  # Download historical data
  start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
  end_date = datetime.now().strftime("%Y-%m-%d")
  data = None
  ticker = None
  try:
    data = yf.download(symbol,
                       start=start_date,
                       end=end_date,
                       auto_adjust=True)
    # Create a Ticker object for a specific symbol
    ticker = yf.Ticker(symbol)
    longName = ticker.info['longName']
  except Exception as e:
    data = None
    ticker = None
    print("Error occurred:", e)

  if data is None or ticker is None:
    return None
  

  
  # Initialize a new dictionary to store the results
  new_data = {}
  new_data['style'] = style
  new_data['symbol'] = symbol
  new_data['longName'] = ticker.info['longName']
  new_data['peratio'] = 1 / ticker.info['trailingPE']
  # new_data['category'] = ticker.info['category']
  new_data['type'] = ticker.info['legalType']
  new_data['L.Year'] = data['Close'].iloc[0]
  new_data['Today'] = data['Close'].iloc[-1]
  new_data['risk_score'] = calculate_combined_risk_score(data, ticker)
  # Create a new list to store article URLs
  # article_urls = [article.get('link') for article in ticker.news]
  # news = get_stock_performance_sentiment(symbol)
  # new_data['news_overall'] = news['overall_sentiment']
  # new_data['inclination'] = 1 if news['average_sentiment']['positive'] > news[
  #     'average_sentiment']['negative'] else -1
  trend_anlaysis = calculate_combined_trend_score(data, ticker)
  new_data['rsi'] = trend_anlaysis['rsi']
  new_data['market'] = trend_anlaysis['market']
  new_data['m.trend'] = trend_anlaysis['trend']
  new_data['inverse_risk_score'] = 1 / new_data['risk_score']

  new_data['15 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-15]) /
                        data['Close'].iloc[-7]) * 100
  new_data['30 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-30]) /
                        data['Close'].iloc[-30]) * 100
  new_data['60 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-60]) /
                        data['Close'].iloc[-60]) * 100
  new_data['90 day'] = ((data['Close'].iloc[-1] - data['Close'].iloc[-90]) /
                        data['Close'].iloc[-90]) * 100
  new_data['1Y'] = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) /
                    data['Close'].iloc[0]) * 100

  # Shift the data by one day
  shifted_data = data['Close'].shift(0)
  new_data['MA_60'] = shifted_data.rolling(
      window=60).mean().iloc[-1] / data['Close'].iloc[-1]
  # Now apply the rolling window
  rolling_max = shifted_data.rolling(window=60).max()
  rolling_min = shifted_data.rolling(window=60).min()
  # Continue with your calculations
  new_data['Max60'] = (rolling_max.iloc[-1])
  new_data['Min60'] = (rolling_min.iloc[-1])
  new_data['fluctuation_60'] = 1 - (
      (new_data['Max60'] - data['Close'].iloc[-1]) /
      (new_data['Max60'] - new_data['Min60']))

  new_data['Volume_60'] = data['Volume'].rolling(window=60).mean().iloc[-1]
  # Calculate Volatility
  rolling_returns_60 = data['Close'].pct_change().rolling(
      window=60).mean().iloc[-1]
  volatility_60 = data['Close'].pct_change().rolling(window=60).std().iloc[-1]
  weights_60 = 1 / volatility_60
  new_data['volatility_60'] = weights_60 * rolling_returns_60
  # new_data['overlap'] = overlap
  # print("XXXX", new_data)
  # Calculate composite score
  weights = [0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.10]
  factors = [
      '60 day', '90 day', 'fluctuation_60', 'volatility_60',
      'inverse_risk_score', 'rsi', 'peratio'
  ]
  # print(new_data)
  new_data['Composite Score'] = np.dot(
      [new_data[factor] for factor in factors], weights)

  return new_data


# Calculate indicators for all symbols
etf_data = {}
for style, symbol in etf_tickers:
  data = calculate_indicators(style, symbol)
  if data is not None:
      etf_data[symbol] = data
  else:
      print(f"Skipping {style}:{symbol} due to no data.")

# sp_500_data = {}
# for symbol in get_sp500_tickers():
#   data = calculate_indicators(symbol)
#   if data is not None:
#     sp_500_data[symbol] = data
#   else:
#       print(f"Skipping {symbol} due to no data.")

# etf_data[etf_tickers[0]] = calculate_indicators("", "SMH")

# print(all_data)

# Convert each dictionary in all_data to a DataFrame
etf_data = {symbol: pd.DataFrame([data]) for symbol, data in etf_data.items()}
# sp_500_data = {symbol: pd.DataFrame([data]) for symbol, data in sp_500_data.items()}

# Combine all data into a single DataFrame
combined_etf_data = pd.concat(etf_data)
# combined_sp_500_data = pd.concat(sp_500_data)

# Sort by composite score
ranked_etf_data = combined_etf_data.sort_values(by='Composite Score',
                                                ascending=False)
# ranked_sp_500_data = combined_sp_500_data.sort_values(by='Composite Score', ascending=False)

ranked_etf_data = ranked_etf_data.round(2)
# ranked_sp_500_data = ranked_sp_500_data.round(2)

# Filter and categorize ETFs correctly
def categorize_etf(row):
  if "S&P" in row['longName']:
    if row['style'] == 'consistent-growth':
      return 'SPCG'
    elif row['style'] == 'aggressive-growth':
      return 'SPAG'
  else:
    if row['style'] == 'consistent-growth':
      return 'NSPCG'
    elif row['style'] == 'aggressive-growth':
      return 'NSPAG'
  return None

top_etf_symbols_by_category = {}

ranked_etf_data['category'] = ranked_etf_data.apply(categorize_etf, axis=1)

# Exclude ETFs that don't fit into the specified categories
filtered_etf_data = ranked_etf_data.dropna(subset=['category'])
filtered_etf_data['overlap'] = None
max_volume = filtered_etf_data['Volume_60'].max()
# Sort ETFs within each category by their rank
filtered_etf_data.sort_values(by=['category', 'Composite Score'],
                              ascending=[True, False],
                              inplace=True)

for index, row in filtered_etf_data.iterrows():
    category = row['category']
    current_etf_symbol = row['symbol']

    # Check if the top ETF symbol for the current category is already noted
    if category not in top_etf_symbols_by_category:
        # If not, the current ETF is the top ETF for its category (due to sorting)
        top_etf_symbols_by_category[category] = current_etf_symbol
        # The top ETF has 100% overlap with itself by definition
        filtered_etf_data.at[index, 'overlap'] = 0.001
    else:
        # For other ETFs, calculate the overlap with the top ETF of their category
        top_etf_symbol = top_etf_symbols_by_category[category]
        overlap_percentage = calculate_overlap_percentage(current_etf_symbol, top_etf_symbol)
        filtered_etf_data.at[index, 'overlap'] = overlap_percentage

filtered_etf_data['rank_metric'] = 0.70 * filtered_etf_data[
    'Composite Score'] + 0.10 * (filtered_etf_data['Volume_60'] / max_volume) + 0.20 * (1-filtered_etf_data['overlap'])

# Sort ETFs within each category by their rank
filtered_etf_data.sort_values(by=['category', 'rank_metric'],
                              ascending=[True, False],
                              inplace=True)
print(filtered_etf_data)

filtered_etf_data['overlap'] = filtered_etf_data['overlap'].astype(float)

# Step 1: Round the 'overlap' values to two decimal places
filtered_etf_data['overlap_rounded'] = filtered_etf_data['overlap'].round(2)

# Step 2: Mark duplicates in 'overlap_rounded' within each 'category'
# keep=False marks all duplicates as True, including the first occurrence
# ~ (NOT) operator is used to keep rows that are NOT marked as duplicates
is_unique_within_category = ~filtered_etf_data.duplicated(subset=['category', 'overlap_rounded'], keep='first')

# Step 3: Filter out the duplicates based on the above condition
filtered_etf_data = filtered_etf_data[is_unique_within_category]

filtered_etf_data.reset_index(drop=True, inplace=True)

# # Optionally, you can drop the 'overlap_rounded' column if it's no longer needed
# filtered_etf_data = filtered_etf_data_unique.drop(columns=['overlap_rounded'])

print(filtered_etf_data)


# Select top ETFs per category according to allocation percentages
def allocate_etfs(df, allocations):
  allocated_etfs = pd.DataFrame()
  for category, allocation in allocations.items():
      # Select ETFs within the current category
      category_etfs = df[df['category'] == category]
      # Calculate the number of ETFs to select based on the allocation percentage
      num_etfs = round(len(category_etfs) * allocation / 100)
      num_etfs = max(num_etfs, 1)  # Ensure at least one ETF is selected
  
      # Concatenate the selected ETFs to the allocated ETFs DataFrame
      allocated_etfs = pd.concat([allocated_etfs, category_etfs.head(num_etfs)])
  
  return allocated_etfs


# Allocate ETFs based on specified portfolio percentages
portfolio_allocations = {
    'SPCG': 50,
    'SPAG': 20,
    'NSPCG': 20,
    'NSPAG': 10
}

allocated_etfs = allocate_etfs(filtered_etf_data, portfolio_allocations)

# Calculate the investment amount for each selected ETF
total_investment = 15000
allocated_etfs['investment_amount'] = allocated_etfs['category'].apply(
    lambda x: total_investment * (portfolio_allocations[x] / 100))

# Preparing the final display table
final_portfolio = allocated_etfs[[
    'symbol', 'rank_metric', 'longName', 'category', 'investment_amount', 'overlap', 'Composite Score'
]].copy()

# Correctly calculate the investment amount by distributing within categories evenly
for category in portfolio_allocations.keys():
  final_portfolio.loc[final_portfolio['category'] == category,
                      'investment_amount'] /= final_portfolio[
                          final_portfolio['category'] == category].shape[0]

final_portfolio.reset_index(drop=True, inplace=True)
print(final_portfolio)

# Get the current date
current_date = datetime.today().strftime('%Y-%m-%d')
# Save the DataFrame as a CSV file with the current date in the filename
ranked_etf_data.to_csv(f'ranked_etf_data{current_date}.csv')
final_portfolio.to_csv(f'final_portfolio_{current_date}.csv')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  markdown_str = final_portfolio.to_markdown(index=False)

# Combine the date and the markdown table in the content to be written to the file
content_to_write = f"Data as of {current_date}:\n\n{final_portfolio}"

# Write the combined content to readme.md
with open('readme.md', 'w') as file:
    file.write(content_to_write)

# ranked_sp_500_data.to_csv(f'ranked_sp_500_data{current_date}.csv')


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


# top_etfs = ranked_data.index.get_level_values(0)[:5]
# print(list(top_etfs))
# backtest(list(top_etfs))
