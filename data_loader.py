import yfinance as yf
from fredapi import Fred
import pandas as pd
import tweepy
from textblob import TextBlob
from config import FRED_API_KEY, TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET

def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def load_fred_data(series_id):
    fred = Fred(api_key=FRED_API_KEY)
    series = fred.get_series(series_id)
    return series

def get_twitter_sentiment(keyword, count=100):
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
    sentiments = []
    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        sentiments.append(analysis.sentiment.polarity)
    return pd.Series(sentiments)
