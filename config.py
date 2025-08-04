import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

FRED_API_KEY =os.getenv("FRED_API_KEY")
NEWS_API_KEY =os.getenv("NEWS_API_KEY")

TWITTER_API_KEY =os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET =os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN =os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET =os.getenv("TWITTER_ACCESS_SECRET")

MODEL_DIR =os.getenv("MODEL_DIR", "./models")
LOGBOOK_PATH =os.getenv("LOGBOOK_PATH", "./logs/trade_log.csv")

__all__ = ["FRED_API_KEY", "NEWS_API_KEY", "TWITTER_API_KEY", "TWITTER_API_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET", "MODEL_DIR", "LOGBOOK_PATH"]
