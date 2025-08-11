from __future__ import annotations
import yfinance as yf
from fredapi import Fred
import pandas as pd
import tweepy
from textblob import TextBlob
import numpy as np
import yfinance as yf
from typing import List, Union, Optional
from config import FRED_API_KEY, TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET

# def load_prices(tickers, start, end):
#     df = yf.download(tickers, start=start, end=end)['Close']
#     df = df.dropna(how='all').ffill().dropna()
#     return df

# def load_fred_data(series_id):
#     fred = Fred(api_key=FRED_API_KEY)
#     series = fred.get_series(series_id)
#     return series

# def get_twitter_sentiment(keyword, count=100):
#     auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
#     auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
#     api = tweepy.API(auth, wait_on_rate_limit=True)
    
#     tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
#     sentiments = []
#     for tweet in tweets:
#         analysis = TextBlob(tweet.text)
#         sentiments.append(analysis.sentiment.polarity)
#     return pd.Series(sentiments)



# ---- Optional imports guarded (so unit tests don’t explode if keys aren’t set)
try:
    from fredapi import Fred
except Exception:
    Fred = None

try:
    import tweepy
    from textblob import TextBlob
except Exception:
    tweepy = None
    TextBlob = None

# -------------------- Stocks --------------------
def load_stock_data(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: str,
    use_adj_close: bool = True,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Load price data for one or many tickers. Returns a DataFrame with
    Date index and one column per ticker (even for a single ticker).
    """
    # yfinance tip: if auto_adjust=True, Adj Close is essentially Close
    df = yf.download(
        tickers if isinstance(tickers, list) else [tickers],
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
    )

    # yfinance returns a MultiIndex columns when multiple tickers OR multiple fields
    # We want just the price level (Adj Close or Close), columns=tickers
    px_col = "Adj Close" if use_adj_close and "Adj Close" in df.columns.get_level_values(0) else "Close"
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[px_col].copy()
    else:
        # Single field returned -> already a single-level frame
        prices = df.copy()
        # If it still has OHLCV columns, pick Close/Adj Close
        if px_col in prices.columns:
            prices = prices[[px_col]]
        elif "Close" in prices.columns:
            prices = prices[["Close"]]

    # Normalize column names to ticker strings
    if isinstance(tickers, str):
        prices.columns = [tickers]
    else:
        prices.columns = list(prices.columns)

    # Clean: drop full-NaN rows, forward-fill small gaps, keep business days
    prices = prices.dropna(how="all").ffill().dropna(how="any")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.asfreq("B").ffill()

    return prices  # shape: [dates x tickers]

# -------------------- FRED --------------------
def load_fred_data(series_ids: Union[str, List[str]], start: Optional[str] = None, end: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Load 1+ FRED series and return an aligned DataFrame on a business-day index.
    """
    if Fred is None:
        raise ImportError("fredapi not available. Install fredapi or guard this call.")

    fred = Fred(api_key=api_key or FRED_API_KEY)
    ids = [series_ids] if isinstance(series_ids, str) else series_ids

    out = {}
    for sid in ids:
        s = fred.get_series(sid)
        s = s.rename(sid).to_frame()
        out[sid] = s

    df = pd.concat(out.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    if start: df = df[df.index >= pd.to_datetime(start)]
    if end:   df = df[df.index <= pd.to_datetime(end)]

    # Business-day frequency + ffill
    df = df.asfreq("B").ffill()
    return df

# -------------------- Twitter Sentiment (best-effort) --------------------
def get_twitter_sentiment(keyword: str, count: int = 100, api_keys: Optional[dict] = None) -> pd.Series:
    """
    Returns a Series of sentiment polarity values.
    If tweepy/TextBlob are not available, returns an empty Series.
    """
    if tweepy is None or TextBlob is None:
        return pd.Series(dtype=float)

    if not api_keys:
        # Fall back to your old config module if you still use it.
        # from config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
        raise ValueError("Pass api_keys={'api_key': ..., 'api_secret': ..., 'access_token': ..., 'access_secret': ...}")

    auth = tweepy.OAuthHandler(api_keys["TWITTER_API_KEY"], api_keys["TWITTER_API_SECRET"])
    auth.set_access_token(api_keys["TWITTER_ACCESS_TOKEN"], api_keys["TWITTER_ACCESS_SECRET"])
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # NOTE: Tweepy v1.1 search_tweets can be rate-limited / deprecated in some setups.
    # You may want to migrate to v2 endpoints later.
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(count)

    rows = []
    for tw in tweets:
        text = getattr(tw, "full_text", tw.text)
        polarity = TextBlob(text).sentiment.polarity
        ts = pd.to_datetime(tw.created_at)
        rows.append((ts, polarity))

    if not rows:
        return pd.Series(dtype=float)

    s = pd.Series([p for _, p in rows], index=[t for t, _ in rows]).sort_index()
    return s

