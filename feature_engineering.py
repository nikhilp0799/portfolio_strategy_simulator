import pandas as pd

def add_rolling_features(df, window_sizes=[5, 10, 20]):
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
    return df.dropna()
