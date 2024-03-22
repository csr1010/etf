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
from lxml import html
import re
import math

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
    'socially-responsible', 'consistent-growth', 'aggressive-growth'
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
      etf_cell = row.find('td', attrs={'data-th': 'ETF Name'})
      if symbol_cell:
        symbol = symbol_cell.find('a').text.strip()
        symbol_name = etf_cell.find('a').text.strip()
        symbols.append((symbol, symbol_name))
  return symbols


# Define the sector-specific URL
sector_specific_url = "https://etfdb.com/etfs/sector/{sector}/?search[inverse]=false&search[leveraged]=false#etfs__returns&sort_name=assets_under_management&sort_order=desc&page=1"
investment_style_url = 'https://etfdb.com/etfs/investment-style/{style}/?search[inverse]=false&search[leveraged]=false#etfs&sort_name=assets_under_management&sort_order=desc&page={page}'
# Scrape ETF symbols for each of the top 4 sectors
etf_tickers = []
hm_keywords = ["global", "intl", "international", "world"]
lb_keywords = ["bond", "treasury", "dividend"]

for style in investment_styles:
    pages = [1,2]
    for page in pages:
      url = investment_style_url.format(style=style, page=page)
      tickers = scrape_etf_symbols(url)
      for ticker, name in tickers:
          # Convert name to lowercase for case-insensitive comparison
          name_lower = name.lower()
          if style == "high-momentum":
              if any(keyword in name_lower for keyword in hm_keywords):
                  etf_tickers.append((style, ticker))
          else:
              # For other styles, append without keyword check
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
expense_aum_ratio = {}

def parse_monetary_value(value):
  # Regular expression to match the number and optional 'M' or 'B'
  match = re.search(r"\$?([\d,]+\.?\d*)\s*([MB])?", value)
  if match:
    # Extract the numeric part and remove commas
    number_str = match.group(1).replace(',', '')
    number = float(number_str)

    # Multiply by a million or billion if 'M' or 'B' is present
    if match.group(2) == 'M':
        number *= 1e6
    elif match.group(2) == 'B':
        number *= 1e9

    return number

pe_exp_ratio_map = {}
  
def get_etf_pe_ratio(etf_ticker):
  time.sleep(1)
  urls = [
    "https://stockanalysis.com/etf/{etf_ticker}",
    "https://etfdb.com/etf/{etf_ticker}/#etf-ticker-valuation-dividend",
  ]
  xpaths_pe_ratio = [
    '/html/body/div/div[1]/div[2]/main/div[2]/div[2]/table[1]/tbody/tr[3]/td[2]/text()',
'/html/body/div[2]/div[9]/div[2]/div/div[1]/div/div[3]/div/div[1]/div/div/div/div/div/div[2]/div/div/div[1]/div[4]/text()',
  ]
  xpaths_exp_ratio = [
    '/html/body/div/div[1]/div[2]/main/div[2]/div[2]/table[1]/tbody/tr[2]/td[2]/text()',
    '/html/body/main/div/div[2]/div[2]/div/div[2]/div/div[1]/div/div[2]/div[2]/div/table/tbody/tr[9]/td[2]/text()',
  ]
  for i, url in enumerate(urls):
    print("PE", etf_ticker, url)
    try:
      response = requests.get(url.format(etf_ticker=etf_ticker))
      doc = html.fromstring(response.content)

      pe_ratio = doc.xpath(xpaths_pe_ratio[i])
      exp_ratio = doc.xpath(xpaths_exp_ratio[i])

      print("PE", pe_ratio, exp_ratio)

      if pe_ratio or exp_ratio:
          exp = float(exp_ratio[0].replace("%", "")) if exp_ratio and exp_ratio[0] else 0.02
          if pe_ratio[0] == "N/A":
            pe = 0
          else:
            pe = float(pe_ratio[0])
          pe_exp_ratio_map[etf_ticker] = (pe, exp)
          print("PE IN ", url, pe_exp_ratio_map[etf_ticker])
          return pe_exp_ratio_map[etf_ticker]
    except Exception as e:
        print("pe exception", e)

  pe_exp_ratio_map[etf_ticker] = (0, 0)
  return (0, 0)

def calculate_combined_risk_score(data, ticker_symbol):
  etf_pe_ratio, exp_ratio = pe_exp_ratio_map[ticker_symbol]
  safe_score_normalized = (1/etf_pe_ratio)*10
  print ("safe score", ticker_symbol, etf_pe_ratio, safe_score_normalized)
  
  return {
      "safe_score_normalized": safe_score_normalized
  }

def calculate_combined_trend_score(data):
  # Assuming 'ema', 'rsi', and 'bands' functions are defined and return the necessary values.
  ema_val = ema(data["Close"], 14)
  rsi_val = rsi(data["Close"], 14)
  bollinger_bands = bands(data["Close"], 14, 2)
  upper_band, lower_band = bollinger_bands[0], bollinger_bands[1]
  
  score = 0
  trend = 'neutral'
  situation = 'neutral'
  
  # RSI Conditions
  if rsi_val[-1] > 70:
      situation = 'overbought'
      score -= 3
  elif 60 < rsi_val[-1] <= 70:
      situation = 'close to overbought'
      score -= 1
  elif 30 <= rsi_val[-1] < 40:
      situation = 'close to oversold'
      score += 1
  elif rsi_val[-1] < 30:
      situation = 'oversold'
      score += 3
  
  # EMA Trend
  if data["Close"].iloc[-1] > ema_val[-1]:
      trend = 'up'
      score += 1
  elif data["Close"].iloc[-1] < ema_val[-1]:
      trend = 'down'
      score -= 1
  
  # Bollinger Bands Condition
  if data["Close"].iloc[-1] >= upper_band[-1]:
      score -= 1  # Adjusting score based on Bollinger Band condition
  elif data["Close"].iloc[-1] <= lower_band[-1]:
      score += 1
  
  # Normalize score to be between 0 and 1
  normalized_score = (score + 5) / 10  # Assuming score range is -5 to +5 for normalization
  
  return {
      "overall_score": normalized_score,
      "trend": trend,
      "situation": situation
  }

cache = {}

def get_etf_holdings(symbol):
  try:
    # Send a GET request to the URL
    url = f"https://etfdb.com/etf/{symbol}/#holdings"
    time.sleep(1)
    # url = f"https://stockanalysis.com/etf/{symbol}/holdings/"
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP request errors
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the holdings table by ID
    holdings_table = soup.find('table', {'id': 'etf-holdings'})
    # holdings_table = soup.find('table', {'class': 'svelte-1jtwn20'})

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

  # exp, aum = get_etf_expense_aum(symbol)
  pe,exp = get_etf_pe_ratio(symbol)
  if pe == 0:
    return None
  
  
  # Initialize a new dictionary to store the results
  new_data = {}
  new_data['style'] = style
  new_data['symbol'] = symbol
  new_data['longName'] = ticker.info['longName']
  peratio, exp_ratio = pe_exp_ratio_map[symbol]
  new_data['peratio'] = peratio

  # new_data['exp_ratio'] = exp
  # new_data['category'] = ticker.info['category']
  new_data['type'] = ticker.info['legalType']
  new_data['L.Year'] = data['Close'].iloc[0]
  new_data['Today'] = data['Close'].iloc[-1]
  new_data['inverse_risk_score'] = calculate_combined_risk_score(data, symbol)['safe_score_normalized']
  # Create a new list to store article URLs
  # article_urls = [article.get('link') for article in ticker.news]
  # news = get_stock_performance_sentiment(symbol)
  # new_data['news_overall'] = news['overall_sentiment']
  # new_data['inclination'] = 1 if news['average_sentiment']['positive'] > news[
  #     'average_sentiment']['negative'] else -1
  trend_anlaysis = calculate_combined_trend_score(data)
  new_data['analysis_score'] = trend_anlaysis['overall_score']
  new_data['market'] = trend_anlaysis['situation']
  new_data['trend'] = trend_anlaysis['trend']
  time_periods = [5, 10, 21, 42, 63, 126, 189, 251]
  # period_names = ['1_week', '2_weeks', '1_month', 2_months, 'Q1', 'Q2', 'Q3', 'Q4']
  for period in time_periods:
      new_data[f"{period}_day"] = ((data['Close'].iloc[-1] - data['Close'].iloc[-period]) /
                                  data['Close'].iloc[-period]) * 100

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

  time_periods = [30, 60, 90]
  for period in time_periods:
      # Calculate rolling returns
      rolling_returns = data['Close'].pct_change().rolling(window=period).mean().iloc[-1]
  
      # Calculate volatility
      volatility = data['Close'].pct_change().rolling(window=period).std().iloc[-1]
  
      # Calculate weights
      weights = 1 / volatility
  
      # Store the results in the new_data DataFrame
      new_data[f'volatility_{period}'] = weights * rolling_returns
 
  avg_daily_volume = data['Volume'].mean()
  new_data['volume_score'] = (math.log10(avg_daily_volume) - 1) * 100 / (math.log10(10**9) - 1) + 1
  weights = [0.08, 0.08, 0.06, 0.05, 0.07, 0.08, 0.08, 0.10, 0.10, 0.15, 0.10, 0.025, 0.025]
  factors = ['10_day', '21_day', '42_day', '63_day', '126_day', '189_day', '251_day', 'volatility_60','volatility_90', 'fluctuation_60', 'analysis_score', 'volume_score', 'inverse_risk_score']
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

# ranked_etf_data = ranked_etf_data.round(3)
# ranked_sp_500_data = ranked_sp_500_data.round(2)

# Filter and categorize ETFs correctly
def categorize_etf(row):
  if row['style'] == 'high-beta':
    return "LEVRG"
    
  categories = [
      (["100 ETF", "Portfolio S&P 500"], "EXCLD"),
      (["Gold", "Commodity"], "COMD"),
      (["Sector", "Semiconductor", "Health", "Medical"], "SECTR"),
      (["Global", "Intl", "International", "World", "Developed", "Emerging", "Japan", "Taiwan", "China"], "INTL"),
      (["Momentum"], "MTM"),
      (["Bond", "Dividend", "Treasury", "low-beta"], "YIELD"),
      (["Growth"], "GROW"),
      (["Quality"], "QUAL"),
      (["Mega Cap", "MidCap", "Mid Cap", "Small Cap", "Large Cap", "Mid-Cap"], "CAPS"),
      (["aggressive-growth"], "AGRSV")
  ]
  for keywords, category in categories:
    if any(keyword in row['longName'] for keyword in keywords):
        return category
  
  if row['style'] == 'high-momentum':
    return "MTM"

  if row['style'] == 'aggressive-growth':
    return "AGRSV"
    
  if "S&P" in row['longName']:
    if row['style'] == 'consistent-growth' or row['style'] == 'socially-responsible':
      return 'STBL'
  else:
    if row['style'] == 'consistent-growth' or row['style'] == 'socially-responsible':
      return 'NSPCG'
    
  return 'EXCLD'

top_etf_symbols_by_category = {}

ranked_etf_data['category'] = ranked_etf_data.apply(categorize_etf, axis=1)

# Exclude ETFs that don't fit into the specified categories
filtered_etf_data = ranked_etf_data.dropna(subset=['category'])
filtered_etf_data['overlap'] = None


# Sort ETFs within each category by their rank
filtered_etf_data.sort_values(by=['category', 'Composite Score'],
                              ascending=[True, False],
                              inplace=True)

for index, row in filtered_etf_data.iterrows():
  category = row['category']
  # volume60 = row['Volume_60']
  cs = row['Composite Score']
  current_etf_symbol = row['symbol']

  # Calculate the logarithm base 10 of the number
  # log_num = math.log10(volume60)
  # volume_score = (log_num - 1) * 100 / (math.log10(10**9) - 1) + 1
  overlap_percentage = 0.001
  if category not in top_etf_symbols_by_category:
      # If not, the current ETF is the top ETF for its category (due to sorting)
      top_etf_symbols_by_category[category] = current_etf_symbol
      # The top ETF has 100% overlap with itself by definition
      overlap_percentage = 0.001
      filtered_etf_data.at[index, 'overlap'] = overlap_percentage
  else:
      # For other ETFs, calculate the overlap with the top ETF of their category
      top_etf_symbol = top_etf_symbols_by_category[category]
      overlap_percentage = calculate_overlap_percentage(current_etf_symbol, top_etf_symbol)
      filtered_etf_data.at[index, 'overlap'] = overlap_percentage

  filtered_etf_data.at[index, 'rank_metric'] =  0.60 * cs + 0.40 * (1-overlap_percentage)

# Sort ETFs within each category by their rank
filtered_etf_data.sort_values(by=['category', 'rank_metric'],
                              ascending=[True, False],
                              inplace=True)
print(filtered_etf_data)

filtered_etf_data['overlap'] = filtered_etf_data['overlap'].astype(float)
def trim_decimals(row):
  return math.floor(row['overlap'] * 100) / 100

filtered_etf_data['overlap_rounded'] = filtered_etf_data.apply(trim_decimals, axis=1)

# Step 2: Mark duplicates in 'overlap_rounded' within each 'category'
# keep=False marks all duplicates as True, including the first occurrence
# ~ (NOT) operator is used to keep rows that are NOT marked as duplicates
is_unique_within_category = ~filtered_etf_data.duplicated(subset=['category', 'overlap_rounded'], keep='first')

# Step 3: Filter out the duplicates based on the above condition
filtered_etf_data = filtered_etf_data[is_unique_within_category]

# filtered_etf_data.reset_index(drop=True, inplace=True)

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
      weight, num_etfs = allocation
  
      # Concatenate the selected ETFs to the allocated ETFs DataFrame
      allocated_etfs = pd.concat([allocated_etfs, category_etfs.head(num_etfs)])
  
  return allocated_etfs

# Allocate ETFs based on specified portfolio percentages
portfolio_allocations = {
    # Regular
    'STBL': (.45, 1),
    'GROW': (.10,1),
    'QUAL': (.10,1),
    # 'NSPCG': (.5,2),
    # 'CAPS': (.5,2),
    # Experimentation
    'AGRSV': (.10,1),
    # 'LEVRG': (.10,1),
    # 'MTM': (.5,1),
    # 'INTL': (.5,1),
    'SECTR': (.5,1),
    # future bonds, commodity, real estate, RIET, dividents
    # 'YIELD': (5,1)
    # balance
    'CASH': (0,1)
}

# Specifying default values for each column
default_values = {'symbol': 'CASH', 'category': 'CASH', 'rank_metric': 1, 'longName': "Unallocated Cash", 'overlap': 0, 'Composite Score': 0}
# Create a DataFrame from the new row data
new_row = pd.DataFrame([default_values])
# Adding the row with default values
filtered_etf_data= pd.concat([filtered_etf_data, new_row], ignore_index=True)

allocated_etfs = allocate_etfs(filtered_etf_data, portfolio_allocations)
# Calculate the investment amount for each selected ETF
total_investment = 25000
# allocated_etfs['investment_amount'] = allocated_etfs['category'].apply(
#     lambda x: total_investment * (portfolio_allocations[x][0] / 100))

# Create a new column in df called main_category
allocated_etfs['main_category'] = np.where(allocated_etfs['category'].isin(['LEVRG', 'AGRSV', 'MTM', 'INTL', 'SECTR']), 'EXPERIMENTAL', 'REGULAR')

def allocate_investment(df):
  # Calculate the sum of rank_metric for each main category
  main_category_rank_sum = df.groupby('main_category')['rank_metric'].sum()

  # Calculate the total rank_metric for all ETFs
  total_rank_metric = df['rank_metric'].sum()

  # Calculate the percentage of each main category based on its rank_metric sum
  main_category_percentage = main_category_rank_sum / total_rank_metric * 100

  # Allocate investment based on the percentage of each main category
  experimental_percentage = main_category_percentage[main_category_percentage.index == 'EXPERIMENTAL'].iloc[0]
  regular_percentage = main_category_percentage[main_category_percentage.index == 'REGULAR'].iloc[0]
  experimental_allocation = experimental_percentage / 100 * total_investment
  regular_allocation = regular_percentage / 100 * total_investment

  # Allocate investment to each ETF within the experimental main category
  experimental_etfs = df[df['main_category'] == 'EXPERIMENTAL']
  experimental_etfs['investment_amount'] = experimental_etfs['rank_metric'] / experimental_etfs['rank_metric'].sum() * experimental_allocation
  experimental_etfs['investment_amount'] = experimental_etfs['investment_amount'].round(0)

  # Allocate investment to each ETF within the reg main category
  reg_etfs = df[df['main_category'] == 'REGULAR']
  reg_etfs['investment_amount'] = reg_etfs['rank_metric'] / reg_etfs['rank_metric'].sum() * regular_allocation
  reg_etfs['investment_amount'] = reg_etfs['investment_amount'].round(0)

  allocated_df = pd.concat([experimental_etfs, reg_etfs])
  return allocated_df

# Call the allocate_investment function to allocate investment based on rank_metric and category weights
allocated_etfs = allocate_investment(allocated_etfs)

# Preparing the final display table
final_portfolio = allocated_etfs[[
    'symbol', 'investment_amount', 'rank_metric', '5_day', '251_day', 'main_category', 'category'
]].copy()

# Correctly calculate the investment amount by distributing within categories evenly
# for category in portfolio_allocations.keys():
#   final_portfolio.loc[final_portfolio['category'] == category,
#                       'investment_amount'] /= final_portfolio[
#                           final_portfolio['category'] == category].shape[0]

final_portfolio.reset_index(drop=True, inplace=True)
pd.set_option('display.max_rows', None)
print(final_portfolio)
print("*** Next IND, BTC, EV, YIELD, MOMENTUM, INTL, REIT, BOND, H.BETA, L.BETA ***")

# Get the current date
current_date = datetime.today().strftime('%Y-%m-%d')
# Save the DataFrame as a CSV file with the current date in the filename
filtered_etf_data.to_csv(f'filtered_etf_data{current_date}.csv')
final_portfolio.to_csv(f'final_portfolio_{current_date}.csv')
final_portfolio.to_csv('final_portfolio.csv')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  markdown_str = final_portfolio.to_markdown(index=False)

# Combine the date and the markdown table in the content to be written to the file
content_to_write = f"\n\n Data as of {current_date}:\n\n{markdown_str}"

# Write the combined content to readme.md
with open('readme.md', 'a') as file:
    file.write(content_to_write)
