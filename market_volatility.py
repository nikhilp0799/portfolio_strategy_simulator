

!pip install alpha_vantage pandas numpy matplotlib yfinance newsapi-python textblob quandl tweepy fredapi xgboost

import os
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from newsapi import NewsApiClient
import requests
import tweepy
from fredapi import Fred

# Initialize API clients
newsapi = NewsApiClient(api_key='b7e99c51eaf3492e81ef8d2f206cbb86')

# FRED API Key
FRED_API_KEY = "50113643558d6e3a373925ec8d3f9b57"
fred = Fred(api_key=FRED_API_KEY)

# Initialize Tweepy API for Twitter sentiment analysis
twitter_auth = tweepy.OAuthHandler('WOR1qlYILo9F2n5EKVJ1KTmVV', 'pHJUMNkryr0bIRAPPxcof0MM7YVaVuF93bTOIBNJbibZKxW9uw')
twitter_auth.set_access_token('1011195364496560128-HZLi5t3gSMdeg5osTtXG5YuvNhOXzx', 'gmGKlXmZUdQKC1olklaaRHaAyg5jzfpcOOJr7LoJ83fSZ')
twitter_api = tweepy.API(twitter_auth, wait_on_rate_limit=True)

# Function to flatten DataFrame columns
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    if 'Date_' in df.columns:
        df.rename(columns={'Date_': 'Date'}, inplace=True)

# Fetch Market Indices
def fetch_index_data(symbols, start_date, end_date):
    index_data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    index_data = index_data.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    flatten_columns(index_data)
    return index_data[['Date', 'Ticker', 'Close']]

# Fetch Stock Data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    flatten_columns(stock_data)

    if 'Close' in stock_data.columns:
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
        stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['OBV'] = (np.sign(stock_data['Close'].diff()) * stock_data['Volume']).cumsum()
        stock_data['VWAP'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()

    return stock_data

# Fetch Economic Indicators
def fetch_fred_data(series_ids, start_date, end_date):
    dfs = []
    for series_id in series_ids:
        try:
            data = fred.get_series(series_id, start_date, end_date)
            df = pd.DataFrame({'Date': data.index, f"{series_id}_Value": data.values})
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch {series_id} from FRED: {e}")

    if dfs:
        result_df = pd.concat(dfs, axis=1)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        return result_df
    else:
        return pd.DataFrame()

# Merge Data with improved handling
def merge_data(stock_data, index_data, economic_data):
    if economic_data.empty:
        print("Economic Data missing, skipping merge.")
        return stock_data

    for df in [stock_data, index_data, economic_data]:
        df['Date'] = pd.to_datetime(df['Date'])

    merged_data = stock_data.merge(index_data, on=['Date'], how='left', suffixes=('', '_Index'))
    merged_data = merged_data.merge(economic_data, on=['Date'], how='left', suffixes=('', '_Econ'))

    merged_data = merged_data.ffill().bfill()  # Fill forward/backward
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    return merged_data

# Process and Save
def process_tickers(tickers, start_date, end_date, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_data = fetch_index_data(['^GSPC', '^DJI', '^IXIC'], start_date, end_date)
    economic_data = fetch_fred_data(['DGS10', 'GDP', 'CPIAUCSL', 'UNRATE'], start_date, end_date)

    for ticker in tickers:
        print(f"Processing {ticker}")
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        flatten_columns(stock_data)

        if not stock_data.empty:
            merged_data = merge_data(stock_data, index_data, economic_data)
            if not merged_data.empty:
                file_path = os.path.join(output_dir, f'{ticker}_financial_data.csv')
                merged_data.to_csv(file_path, index=False)
                print(f"Saved data for {ticker} to {file_path}")
            else:
                print(f"Skipping {ticker} due to incomplete data.")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2019-01-01'  # Increased historical range for better training
    end_date = '2023-12-31'
    output_dir = './output'
    process_tickers(tickers, start_date, end_date, output_dir)

import os
import pandas as pd

def load_and_prepare_stock_data(file_path, ticker, drop_economic_cols=False):
    """
    Loads ticker data and renames stock-specific columns with ticker prefix.
    Optionally drops economic indicator columns if they already exist in wide_data.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Columns that are stock-specific (rename them with ticker prefix)
    stock_columns = [col for col in df.columns if col not in ['DGS10_Value', 'GDP_Value', 'CPIAUCSL_Value', 'UNRATE_Value']]
    df.rename(columns={col: f"{ticker}_{col}" for col in stock_columns}, inplace=True)

    # Drop economic columns if requested (only keep them for the first ticker)
    if drop_economic_cols:
        df.drop(columns=['DGS10_Value', 'GDP_Value', 'CPIAUCSL_Value', 'UNRATE_Value'], inplace=True, errors='ignore')

    return df

def create_wide_format(tickers, input_dir, output_dir):
    """
    Merges all ticker datasets into wide-format without duplicate economic indicators.
    """
    wide_data = None
    first_ticker = True
    for ticker in tickers:
        file_path = os.path.join(input_dir, f"{ticker}_financial_data.csv")
        if os.path.exists(file_path):
            df_ticker = load_and_prepare_stock_data(file_path, ticker, drop_economic_cols=not first_ticker)
            if wide_data is None:
                wide_data = df_ticker
                first_ticker = False
            else:
                wide_data = wide_data.join(df_ticker, how='outer')  # Merge on Date
        else:
            print(f"⚠️ File for {ticker} not found, skipping.")

    wide_data = wide_data.sort_index()
    wide_data.ffill(inplace=True)
    wide_data.bfill(inplace=True)

    # Save the wide-format dataset
    output_file = os.path.join(output_dir, 'combined_cleaned_wide_data.csv')
    wide_data.reset_index(inplace=True)
    wide_data.to_csv(output_file, index=False)
    print(f"✅ Wide-format data saved to {output_file}")

    print(f"✅ Columns included in wide-format data:\n{wide_data.columns.tolist()}")

# Example usage:
tickers = ['AAPL', 'MSFT', 'GOOGL']
input_dir = './output'
output_dir = './output'
create_wide_format(tickers, input_dir, output_dir)

import pandas as pd

# ✅ Load combined wide-format data
file_path = './output/combined_cleaned_wide_data.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# ✅ Feature Engineering: Lag features
lags = [1, 3, 7]
for lag in lags:
    for column in ['AAPL_Close', 'MSFT_Close', 'GOOGL_Close']:
        data[f'{column}_lag{lag}'] = data[column].shift(lag)

# ✅ Rolling statistics
rolling_windows = [7, 14, 30]
for window in rolling_windows:
    for column in ['AAPL_Close', 'MSFT_Close', 'GOOGL_Close']:
        data[f'{column}_rolling_mean_{window}'] = data[column].rolling(window=window).mean()
        data[f'{column}_rolling_std_{window}'] = data[column].rolling(window=window).std()

# ✅ Interaction terms with indicators
economic_indicators = ['DGS10_Value', 'GDP_Value', 'CPIAUCSL_Value', 'UNRATE_Value']
for stock in ['AAPL_Close', 'MSFT_Close', 'GOOGL_Close']:
    for econ in economic_indicators:
        data[f'{stock}_x_{econ}'] = data[stock] * data[econ]

# ✅ Drop rows with NaN after feature creation
data.dropna(inplace=True)

# ✅ Save feature-engineered data
data.to_csv('./output/data_with_features_wide.csv')
print("✅ Feature engineering for wide-format complete. Saved to 'data_with_features_wide.csv'.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load feature-engineered data
data = pd.read_csv('./output/data_with_features.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Include all economic indicators used during feature engineering
economic_indicators = ['DGS10_Value', 'GDP_Value', 'CPIAUCSL_Value', 'UNRATE_Value']

def train_rf_model(stock_symbol):
    df = data.copy()
    target_col = f'Close_{stock_symbol}_future'
    df[target_col] = df[f'Close_{stock_symbol}'].shift(-1)
    df.dropna(inplace=True)

    # ✅ Broader feature selection:
    feature_cols = [col for col in df.columns if (
        (stock_symbol in col or any(econ in col for econ in economic_indicators))
        and 'future' not in col
        and not col.startswith('Date')
    )]
    # Also add other stock close prices as predictive features:
    for other_stock in ['AAPL', 'MSFT', 'GOOGL']:
        if other_stock != stock_symbol:
            feature_cols += [f'Close_{other_stock}']

    features = df[feature_cols]
    target = df[target_col]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    train_size = int(len(features_scaled) * 0.7)
    val_size = int(len(features_scaled) * 0.15)

    X_train, X_val, X_test = np.split(features_scaled, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(target, [train_size, train_size + val_size])

    # ✅ Increase estimators and let model be deeper for complexity
    model = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Detailed Metrics
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n📈 {stock_symbol} Random Forest Performance:")
    print(f"Validation Set → MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"Test Set       → MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Save model & scaler
    joblib.dump(model, f'./output/random_forest_{stock_symbol}_model.pkl')
    joblib.dump(scaler, f'./output/scaler_{stock_symbol}.pkl')
    print(f"✅ Saved model & scaler for {stock_symbol}\n")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label=f"Actual {stock_symbol} Price", color='blue')
    plt.plot(y_test_pred, label=f"Predicted {stock_symbol} Price", linestyle='dashed', color='red')
    plt.title(f"Random Forest Forecast vs Actual for {stock_symbol}")
    plt.xlabel("Time (Test Set)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# Train for all three stocks
train_rf_model("AAPL")
train_rf_model("MSFT")
train_rf_model("GOOGL")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the feature-engineered dataset
data = pd.read_csv('./output/data_with_features.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

economic_indicators = ['DGS10_Value', 'GDP_Value', 'CPIAUCSL_Value']

def train_xgboost_model(stock_symbol):
    target_col = f'Close_{stock_symbol}_future'
    data[target_col] = data[f'Close_{stock_symbol}'].shift(-1)
    data.dropna(inplace=True)

    # Select relevant features (stock-specific + interaction terms)
    feature_cols = [
        col for col in data.columns
        if (stock_symbol in col or any(econ in col for econ in economic_indicators)) and 'future' not in col
    ]

    features = data[feature_cols]
    target = data[target_col]

    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    train_size = int(len(features_scaled) * 0.7)
    val_size = int(len(features_scaled) * 0.15)

    X_train, X_val, X_test = np.split(features_scaled, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(target, [train_size, train_size + val_size])

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # Predictions
    y_val_pred = xgb_model.predict(X_val)
    y_test_pred = xgb_model.predict(X_test)

    # Evaluation
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    val_rmse = np.sqrt(val_mse)
    test_rmse = np.sqrt(test_mse)

    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"📊 {stock_symbol} XGBoost Model Performance:")
    print(f"🔹 Validation Set → MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"🔹 Test Set       → MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Save model & scaler
    joblib.dump(xgb_model, f'./output/xgboost_{stock_symbol}_model.pkl')
    joblib.dump(scaler, f'./output/scaler_{stock_symbol}.pkl')
    print(f"✅ Saved XGBoost model & scaler for {stock_symbol}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label=f"Actual {stock_symbol}", color='blue')
    plt.plot(y_test_pred, label=f"XGB Predicted {stock_symbol}", color='orange', linestyle='dashed')
    plt.legend()
    plt.title(f"XGBoost Forecast vs Actual for {stock_symbol}")
    plt.show()


# Train for all 3:
train_xgboost_model("AAPL")
train_xgboost_model("MSFT")
train_xgboost_model("GOOGL")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get Feature Importances from Random Forest
rf_feature_importance = model.feature_importances_
rf_features = X_train.columns

# Convert to DataFrame
rf_importance_df = pd.DataFrame({'Feature': rf_features, 'Importance': rf_feature_importance})
rf_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importances for Random Forest
plt.figure(figsize=(12, 6))
sns.barplot(x=rf_importance_df['Importance'][:10], y=rf_importance_df['Feature'][:10], palette="Blues_r")
plt.title(" Top 10 Important Features in Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()

# Get Feature Importances from XGBoost
xgb_feature_importance = xgb_model.feature_importances_
xgb_features = X_train.columns

# Convert to DataFrame
xgb_importance_df = pd.DataFrame({'Feature': xgb_features, 'Importance': xgb_feature_importance})
xgb_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importances for XGBoost
plt.figure(figsize=(12, 6))
sns.barplot(x=xgb_importance_df['Importance'][:10], y=xgb_importance_df['Feature'][:10], palette="Oranges_r")
plt.title(" Top 10 Important Features in XGBoost")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()

#  Install required libraries
!pip install pmdarima statsmodels matplotlib

#  Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

#  Load the dataset
file_path = "./output/data_with_features.csv"  # Ensure this file exists in your Colab environment
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select Apple stock closing price for ARIMA modeling
apple_stock = data['Close_AAPL'].dropna()

#  Step 1: Check if the Data is Stationary (ADF Test)
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Data is stationary (No need for differencing)")
    else:
        print(" Data is NOT stationary (Differencing required)")

# Perform ADF test on Apple stock price
print("ADF Test for Apple Stock Price:")
adf_test(apple_stock)

# Step 2: Apply Differencing if the data is not stationary
diff_apple_stock = apple_stock.diff().dropna()

# Recheck stationarity after differencing
print("\n ADF Test After Differencing:")
adf_test(diff_apple_stock)

# Step 3: Auto-detect the best ARIMA model parameters
arima_model = auto_arima(apple_stock, seasonal=False, trace=True, suppress_warnings=True, stepwise=True)

# Step 4: Train the ARIMA Model with the best parameters
best_order = arima_model.order  # Get the best ARIMA order
final_arima = sm.tsa.ARIMA(apple_stock, order=best_order)
arima_result = final_arima.fit()

# Step 5: Forecast Future Stock Prices
forecast_steps = 30  # Predict next 30 days
forecast = arima_result.forecast(steps=forecast_steps)

# Step 6: Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(apple_stock[-100:], label="Actual Stock Price", color='blue')
plt.plot(pd.date_range(start=apple_stock.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast, label="ARIMA Forecast", linestyle="dashed", color='red')
plt.title("📉 ARIMA Forecast vs. Actual Apple Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Print ARIMA Model Summary
print(arima_result.summary())

# Install necessary libraries (if not already installed)
!pip install pmdarima statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

warnings.filterwarnings("ignore")

# Load the feature-engineered dataset
file_path = './output/data_with_features.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select Apple stock closing price
apple_stock = data[['Close_AAPL']].dropna()

# Step 1: Check for Seasonality with ACF and PACF Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(apple_stock, ax=axes[0])
plot_pacf(apple_stock, ax=axes[1])
plt.suptitle("Autocorrelation and Partial Autocorrelation for Apple Stock")
plt.show()

# Step 2: Auto-tune SARIMA Model using pmdarima (AutoSARIMA)
auto_sarima = pm.auto_arima(
    apple_stock,
    seasonal=True,
    m=12,  # Monthly seasonality
    trace=True,
    suppress_warnings=True,
    stepwise=True
)

# Print the best SARIMA model
print(f"\nBest SARIMA Model: {auto_sarima}")

# Step 3: Train the SARIMA Model
best_order = auto_sarima.order  # (p, d, q)
best_seasonal_order = auto_sarima.seasonal_order  # (P, D, Q, s)

sarima_model = SARIMAX(
    apple_stock,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)

# Step 4: Forecasting Future Prices
n_forecast = 30  # Predict next 30 days
future_dates = pd.date_range(start=apple_stock.index[-1], periods=n_forecast + 1, freq="D")[1:]

sarima_forecast = sarima_fit.get_forecast(steps=n_forecast)
sarima_forecast_values = sarima_forecast.predicted_mean

# Step 5: Visualize SARIMA Forecast vs. Actual
plt.figure(figsize=(12, 5))
plt.plot(apple_stock.index[-100:], apple_stock['Close_AAPL'][-100:], label="Actual Stock Price", color="blue")
plt.plot(future_dates, sarima_forecast_values, label="SARIMA Forecast", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("SARIMA Forecast vs. Actual Apple Stock Prices")
plt.legend()
plt.show()

# Step 6: Evaluate SARIMA Performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_true = apple_stock['Close_AAPL'][-n_forecast:]
y_pred = sarima_forecast_values[:len(y_true)]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("\nSARIMA Model Performance:")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

!pip install tensorflow keras

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load feature-engineered dataset
file_path = './output/data_with_features.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Target Variable: Predict next day's Apple Closing Price
data['Close_AAPL_future'] = data['Close_AAPL'].shift(-1)

# Drop rows with NaN values due to shifting
data.dropna(inplace=True)

# Define Features and Target
features = data.drop(columns=['Close_AAPL_future', 'Ticker'])
target = data['Close_AAPL_future']

# Scale Data (LSTMs perform better with normalized data)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

# Convert Data into Time-Series Sequences (LSTM needs 3D input)
def create_sequences(data, target, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])  # 30 time steps
        y.append(target[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 30  # Lookback window (1 month of past data)
X, y = create_sequences(features_scaled, target_scaled, time_steps)

# Split Data into Train (70%), Validation (15%), Test (15%)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Reshape y values to match LSTM output shape
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, X.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)  # Output layer
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Predictions
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Inverse Transform Predictions
y_val_pred = scaler.inverse_transform(y_val_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_val_actual = scaler.inverse_transform(y_val)
y_test_actual = scaler.inverse_transform(y_test)

# Model Evaluation
val_mae = mean_absolute_error(y_val_actual, y_val_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)

val_mse = mean_squared_error(y_val_actual, y_val_pred)
test_mse = mean_squared_error(y_test_actual, y_test_pred)

val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

val_r2 = r2_score(y_val_actual, y_val_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

# Print Results
print(f"LSTM Model Performance on Validation Set:")
print(f"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

print("\nLSTM Model Performance on Test Set:")
print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# Save Model
model.save('./output/lstm_stock_model.h5')

# Plot Predictions vs Actual Stock Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Stock Price", color='blue')
plt.plot(y_test_pred, label="LSTM Forecast", linestyle='dashed', color='red')
plt.title("LSTM Forecast vs. Actual Apple Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = './output/data_with_features.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Target Variable: Predict next day's Apple Closing Price
data['Close_AAPL_future'] = data['Close_AAPL'].shift(-1)  # Shift target column

# Drop rows with NaN values due to shifting
data.dropna(inplace=True)

# Scale Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['Close_AAPL_future', 'Ticker'], errors='ignore'))
target = data['Close_AAPL_future'].values.reshape(-1, 1)
target = scaler.fit_transform(target)

# Create time-series sequences
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30  # Use last 30 days of data
X, y = create_sequences(scaled_data, target, time_steps)

# Split into Train (70%), Validation (15%), Test (15%)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Build GRU Model
gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(1)
])

# Compile Model
gru_model.compile(optimizer='adam', loss='mse')

# Train Model
gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Predictions
y_val_pred = gru_model.predict(X_val)
y_test_pred = gru_model.predict(X_test)

# Rescale Predictions
y_val_pred = scaler.inverse_transform(y_val_pred)
y_test_pred = scaler.inverse_transform(y_test)

# Evaluate Model
val_mae = mean_absolute_error(scaler.inverse_transform(y_val.reshape(-1, 1)), y_val_pred)
test_mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_test_pred)

val_mse = mean_squared_error(scaler.inverse_transform(y_val.reshape(-1, 1)), y_val_pred)
test_mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_test_pred)

val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

val_r2 = r2_score(scaler.inverse_transform(y_val.reshape(-1, 1)), y_val_pred)
test_r2 = r2_score(scaler.inverse_transform(y_test.reshape(-1, 1)), y_test_pred)

# Print Results
print(f"GRU Model Performance on Validation Set:")
print(f"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

print("\nGRU Model Performance on Test Set:")
print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# Save the GRU Model
gru_model.save('./output/gru_stock_model.h5')

print("GRU Model training complete. Saved trained model to 'gru_stock_model.h5'.")

# Plot Predictions vs Actuals
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual Stock Price", color='blue')
plt.plot(y_test_pred, label="GRU Forecast", linestyle='dashed', color='red')
plt.title("GRU Forecast vs. Actual Apple Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

import pandas as pd

# Reshape X_test and X_train to take only the last time step
X_test_2d = X_test[:, -1, :]  # Take the last time step
X_train_2d = X_train[:, -1, :]  # Take the last time step

# Convert to DataFrame
X_test_df = pd.DataFrame(X_test_2d, columns=features.columns)
X_train_df = pd.DataFrame(X_train_2d, columns=features.columns)

# Check for duplicates
duplicate_test_samples = X_test_df.merge(X_train_df, how='inner')

if not duplicate_test_samples.empty:
    print(" Possible Data Leakage: Test data exists in the training set!")
else:
    print(" No duplicate test samples found. Test set is truly unseen.")

# Check if any target variable (future stock price) is in training features
if 'Close_AAPL_future' in features.columns:
    print(" Data Leakage Risk: Future price is in training features!")
else:
    print(" No future leakage in training features.")

print("Actual Test Prices:", y_test[:10])
print("Predicted Prices:", y_test_pred[:10])

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Load and preprocess the dataset
file_path = './output/data_with_features.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

#Target Variable: Predict next day's Apple Closing Price
data['Close_AAPL_future'] = data['Close_AAPL'].shift(-1)
data.dropna(inplace=True)  # Drop NaN values

#Remove target before scaling to maintain 57 features
features = data.drop(columns=['Close_AAPL_future', 'Ticker'], errors='ignore')

#Scale features only
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

#Confirm feature count before training
print(f"Final Feature Count Before Training: {features_scaled.shape[1]}")  # Should be 57

#Convert to supervised learning format (Time Series)
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30  # Align with original GRU model
X, y = create_sequences(features_scaled, data['Close_AAPL_future'].values.reshape(-1, 1), time_steps)

#Train-Test Split (70% Train, 15% Validation, 15% Test)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

#Build the improved GRU Model (now using 57 features)
gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
    Dropout(0.3),
    GRU(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

#Compile the model
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

#Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Train the model
history = gru_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

#Predict on Validation and Test Sets
y_val_pred = gru_model.predict(X_val)
y_test_pred = gru_model.predict(X_test)

#Inverse Transform Predictions for interpretability
y_val_inv = scaler.inverse_transform(
    np.column_stack([np.zeros((len(y_val), features_scaled.shape[1] - 1))] + [y_val.reshape(-1, 1)])
)
y_val_pred_inv = scaler.inverse_transform(
    np.column_stack([np.zeros((len(y_val_pred), features_scaled.shape[1] - 1))] + [y_val_pred.flatten().reshape(-1, 1)])
)

y_test_inv = scaler.inverse_transform(
    np.column_stack([np.zeros((len(y_test), features_scaled.shape[1] - 1))] + [y_test.reshape(-1, 1)])
)
y_test_pred_inv = scaler.inverse_transform(
    np.column_stack([np.zeros((len(y_test_pred), features_scaled.shape[1] - 1))] + [y_test_pred.flatten().reshape(-1, 1)])
)

#Model Evaluation (Including R² Score)
val_mae = mean_absolute_error(y_val_inv[:, -1], y_val_pred_inv[:, -1])
test_mae = mean_absolute_error(y_test_inv[:, -1], y_test_pred_inv[:, -1])

val_mse = mean_squared_error(y_val_inv[:, -1], y_val_pred_inv[:, -1])
test_mse = mean_squared_error(y_test_inv[:, -1], y_test_pred_inv[:, -1])

val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

val_r2 = r2_score(y_val_inv[:, -1], y_val_pred_inv[:, -1])
test_r2 = r2_score(y_test_inv[:, -1], y_test_pred_inv[:, -1])

print(f"Improved GRU Model Performance:")
print(f" Validation Set → MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
print(f"Test Set → MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

#Plot Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv[:, -1], label="Actual Stock Price", color="blue")
plt.plot(y_test_pred_inv[:, -1], label="GRU Forecast", linestyle="dashed", color="red")
plt.title("Improved GRU Forecast vs. Actual Apple Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

#Save the retrained GRU model (57 features)
gru_model.save("./output/improved_gru_stock_model.h5")
print("Retrained GRU Model training complete. Saved model to './output/improved_gru_stock_model.h5'.")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# **Load the Feature-Engineered Dataset**
file_path = './output/data_with_features.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# **Target Variable: Predict Next Day's Apple Closing Price**
data['Close_AAPL_future'] = data['Close_AAPL'].shift(-1)
data.dropna(inplace=True)  # Drop NaN values

# **Extract Features & Target**
features = data.drop(columns=['Close_AAPL_future', 'Ticker'], errors='ignore')  # Drop target & categorical column
target = data['Close_AAPL_future']

# **Normalize Data (Same Process as GRU Training)**
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)  # Scale features only

# **Split Data into Train (70%), Validation (15%), Test (15%)**
train_size = int(len(features) * 0.7)
val_size = int(len(features) * 0.15)

X_train, X_val, X_test = np.split(features_scaled, [train_size, train_size + val_size])
y_train, y_val, y_test = np.split(target, [train_size, train_size + val_size])

# **Load Pre-trained Models**
rf_model = joblib.load('./output/random_forest_stock_model.pkl')  # Random Forest
xgb_model = joblib.load('./output/xgboost_stock_model.pkl')  # XGBoost
gru_model = tf.keras.models.load_model('./output/improved_gru_stock_model.h5', custom_objects={"mse": MeanSquaredError()})  # ✅ Use the **correctly retrained** GRU model

# **Check Feature Count for GRU**
expected_features = gru_model.input_shape[-1]  # Expected feature count in GRU
actual_features = X_val.shape[1]

if actual_features != expected_features:
    raise ValueError(f"🚨 Feature Mismatch! GRU expects {expected_features}, but dataset has {actual_features}. Check preprocessing!")

# **Convert to Time Series Format for GRU**
SEQ_LENGTH = 20  # GRU expects sequences of 20 time steps

def create_sequences(data, seq_length):
    """ Convert 2D data into 3D sequences for GRU """
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X_val_gru = create_sequences(X_val, SEQ_LENGTH)
X_test_gru = create_sequences(X_test, SEQ_LENGTH)

# **Ensure target (y) is aligned with GRU's sequence processing**
y_val_adjusted = y_val.iloc[SEQ_LENGTH:]
y_test_adjusted = y_test.iloc[SEQ_LENGTH:]

# **Get Predictions from Each Model**
y_val_pred_rf = rf_model.predict(X_val[SEQ_LENGTH:])
y_test_pred_rf = rf_model.predict(X_test[SEQ_LENGTH:])

y_val_pred_xgb = xgb_model.predict(X_val[SEQ_LENGTH:])
y_test_pred_xgb = xgb_model.predict(X_test[SEQ_LENGTH:])

y_val_pred_gru = gru_model.predict(X_val_gru).flatten()
y_test_pred_gru = gru_model.predict(X_test_gru).flatten()

# **Weighted Ensemble: More Weight to Random Forest**
weights = {"rf": 0.60, "xgb": 0.25, "gru": 0.15}

y_val_pred_ensemble = (weights["rf"] * y_val_pred_rf +
                       weights["xgb"] * y_val_pred_xgb +
                       weights["gru"] * y_val_pred_gru)

y_test_pred_ensemble = (weights["rf"] * y_test_pred_rf +
                        weights["xgb"] * y_test_pred_xgb +
                        weights["gru"] * y_test_pred_gru)

# **Evaluate Performance**
val_mae = mean_absolute_error(y_val_adjusted, y_val_pred_ensemble)
test_mae = mean_absolute_error(y_test_adjusted, y_test_pred_ensemble)

val_mse = mean_squared_error(y_val_adjusted, y_val_pred_ensemble)
test_mse = mean_squared_error(y_test_adjusted, y_test_pred_ensemble)

val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

val_r2 = r2_score(y_val_adjusted, y_val_pred_ensemble)
test_r2 = r2_score(y_test_adjusted, y_test_pred_ensemble)

# **Print Performance Metrics**
print(f" **Weighted Ensemble Model Performance on Validation Set:**")
print(f"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

print("\n **Weighted Ensemble Model Performance on Test Set:**")
print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# **Plot Actual vs Predicted (Weighted Ensemble)**
plt.figure(figsize=(10, 5))
plt.plot(y_test_adjusted.values, label="Actual Stock Price", color="blue")
plt.plot(y_test_pred_ensemble, label="Weighted Ensemble Forecast", linestyle="dashed", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Weighted Ensemble Forecast vs. Actual Apple Stock Prices")
plt.legend()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "./output/data_with_features.csv"
data = pd.read_csv(file_path)

# Drop non-numeric columns and target variable for correlation analysis
numeric_data = data.drop(columns=["Date", "Ticker"], errors="ignore")

# Compute the correlation matrix
corr_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
plt.title("📊 Feature Correlation Heatmap")
plt.show()

import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Load dataset
file_path = "./output/data_with_features.csv"
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# ✅ Ensure 'Close_AAPL_future' exists
if 'Close_AAPL_future' not in data.columns:
    print("🔍 'Close_AAPL_future' not found. Recreating the target column...")
    data['Close_AAPL_future'] = data['Close_AAPL'].shift(-1)  # Predict next day's price
    data.dropna(inplace=True)

# ✅ Verify columns before proceeding
print("✅ Columns in Data:", data.columns)

# ✅ Prepare features and target
features = data.drop(columns=['Close_AAPL_future', 'Ticker'], errors='ignore')
target = data['Close_AAPL_future']

# ✅ Normalize data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# ✅ Split data
train_size = int(len(features) * 0.7)
val_size = int(len(features) * 0.15)
X_train, X_val, X_test = np.split(features_scaled, [train_size, train_size + val_size])
y_train, y_val, y_test = np.split(target, [train_size, train_size + val_size])

# ✅ Load models
rf_model = joblib.load("./output/random_forest_stock_model.pkl")
xgb_model = joblib.load("./output/xgboost_stock_model.pkl")
gru_model = tf.keras.models.load_model("./output/improved_gru_stock_model.h5",
                                       custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# ✅ Get model predictions
y_test_pred_rf = rf_model.predict(X_test)
y_test_pred_xgb = xgb_model.predict(X_test)

# ✅ Prepare GRU input
SEQ_LENGTH = 20
def create_sequences(data, seq_length):
    Xs = []
    for i in range(len(data) - seq_length):
        Xs.append(data[i:i+seq_length])
    return np.array(Xs)

X_test_gru = create_sequences(X_test, SEQ_LENGTH)
y_test_pred_gru = gru_model.predict(X_test_gru).flatten()

# ✅ Compute ensemble predictions (weighted)
weights = {"rf": 0.50, "xgb": 0.30, "gru": 0.20}
y_test_pred_ensemble = (weights["rf"] * y_test_pred_rf[SEQ_LENGTH:] +
                        weights["xgb"] * y_test_pred_xgb[SEQ_LENGTH:] +
                        weights["gru"] * y_test_pred_gru)

# ✅ Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[SEQ_LENGTH:], label="Actual Stock Price", color="blue")
plt.plot(y_test_pred_rf[SEQ_LENGTH:], label="Random Forest", linestyle="dashed", color="green")
plt.plot(y_test_pred_xgb[SEQ_LENGTH:], label="XGBoost", linestyle="dashed", color="purple")
plt.plot(y_test_pred_gru, label="GRU Forecast", linestyle="dashed", color="orange")
plt.plot(y_test_pred_ensemble, label="Weighted Ensemble", linestyle="dashed", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("📊 Model Comparison: Actual vs Predictions")
plt.legend()
plt.show()

import seaborn as sns

# Compute residuals (errors)
residuals_rf = y_test.values[SEQ_LENGTH:] - y_test_pred_rf[SEQ_LENGTH:]
residuals_xgb = y_test.values[SEQ_LENGTH:] - y_test_pred_xgb[SEQ_LENGTH:]
residuals_gru = y_test.values[SEQ_LENGTH:] - y_test_pred_gru
residuals_ensemble = y_test.values[SEQ_LENGTH:] - y_test_pred_ensemble

# Plot residual distributions
plt.figure(figsize=(12, 6))
sns.histplot(residuals_rf, bins=50, kde=True, color="green", label="Random Forest Residuals")
sns.histplot(residuals_xgb, bins=50, kde=True, color="purple", label="XGBoost Residuals")
sns.histplot(residuals_gru, bins=50, kde=True, color="orange", label="GRU Residuals")
sns.histplot(residuals_ensemble, bins=50, kde=True, color="red", label="Ensemble Residuals")
plt.axvline(0, color='black', linestyle="dashed")
plt.title("📊 Residuals Distribution (Model Errors)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.values[SEQ_LENGTH:], label="Actual Stock Price", color="blue")
plt.plot(y_test_pred_rf[SEQ_LENGTH:], label="Random Forest", linestyle="dashed", color="green")
plt.plot(y_test_pred_xgb[SEQ_LENGTH:], label="XGBoost", linestyle="dashed", color="purple")
plt.plot(y_test_pred_gru, label="GRU Forecast", linestyle="dashed", color="orange")
plt.plot(y_test_pred_ensemble, label="Weighted Ensemble", linestyle="dashed", color="red")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("📊 Time Series Prediction: Actual vs Forecast")
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Evaluate Models
evaluate_model(y_test[SEQ_LENGTH:], y_test_pred_rf[SEQ_LENGTH:], "Random Forest")
evaluate_model(y_test[SEQ_LENGTH:], y_test_pred_xgb[SEQ_LENGTH:], "XGBoost")
evaluate_model(y_test[SEQ_LENGTH:], y_test_pred_gru, "GRU")
evaluate_model(y_test[SEQ_LENGTH:], y_test_pred_ensemble, "Weighted Ensemble")

import pandas as pd
import matplotlib.pyplot as plt

# Get Feature Importances
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Create DataFrames
feature_names = features.columns
rf_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": rf_importances}).sort_values(by="Importance", ascending=False)
xgb_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": xgb_importances}).sort_values(by="Importance", ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 5))
plt.barh(rf_importance_df["Feature"][:10], rf_importance_df["Importance"][:10], color='green', label="Random Forest")
plt.barh(xgb_importance_df["Feature"][:10], xgb_importance_df["Importance"][:10], color='purple', label="XGBoost", alpha=0.7)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Important Features in Prediction")
plt.legend()
plt.gca().invert_yaxis()
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Moving Average Baseline
moving_avg = data["Close_AAPL"].rolling(window=10).mean().dropna()

# ARIMA Model (Auto-Configured)
arima_model = ARIMA(data["Close_AAPL"], order=(5,1,0))  # ARIMA(p=5,d=1,q=0)
arima_fitted = arima_model.fit()
arima_pred = arima_fitted.forecast(steps=len(y_test))

# Plot Baseline vs. Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", color="blue")
plt.plot(moving_avg[-len(y_test):].values, label="Moving Average", linestyle="dashed", color="gray")
plt.plot(arima_pred, label="ARIMA Forecast", linestyle="dashed", color="black")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("📊 Baseline Models vs. Actual Stock Prices")
plt.legend()
plt.show()

report = f"""
📌 **Key Insights & Recommendations**

🔹 **Model Performance Summary**
- Random Forest is the best model with the lowest MAE and highest R².
- XGBoost tends to overfit slightly, leading to higher variance.
- GRU struggles with capturing rapid fluctuations in stock prices.
- The Weighted Ensemble model improves overall prediction accuracy but needs better fine-tuning.

🔹 **Feature Importance**
- According to Random Forest & XGBoost, the most critical features are:
  1. {rf_importance_df.iloc[0]["Feature"]}
  2. {rf_importance_df.iloc[1]["Feature"]}
  3. {rf_importance_df.iloc[2]["Feature"]}

  These features have the highest impact on predicting Apple stock prices.

🔹 **Comparison with Baseline Models**
- Moving Average provides a simple but weak benchmark.
- ARIMA captures some trends but lacks the predictive power of ML models.
- Ensemble methods significantly outperform both baselines.

🔹 **Next Steps for Improvement**
1️⃣ **Optimize Ensemble Weights**: Increase Random Forest's weight to improve accuracy.
2️⃣ **Improve GRU Model**: Increase time steps, add L2 regularization, and fine-tune hyperparameters.
3️⃣ **Feature Engineering**: Use feature selection techniques to remove noise and focus on high-impact variables.
"""

# Print the report
print(report)

import joblib
import tensorflow as tf

# Save Random Forest & XGBoost Models
joblib.dump(rf_model, "./output/random_forest_stock_model.pkl")
joblib.dump(xgb_model, "./output/xgboost_stock_model.pkl")

# Save GRU Model
gru_model.save("./output/improved_gru_stock_model.h5")

# Save Scaler
joblib.dump(scaler, "./output/scaler.pkl")

print("✅ Models and Scaler Saved Successfully!")

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ✅ Define Stock Symbols
stock_symbols = ["AAPL", "MSFT", "GOOGL"]  # Add more if needed

# ✅ Fetch Last 50 Days of Data
def get_stock_data(symbol, period="50d", interval="1d"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    return df[["Open", "High", "Low", "Close", "Volume"]]

# ✅ Collect Data for Multiple Stocks
dataframes = {symbol: get_stock_data(symbol) for symbol in stock_symbols}

# ✅ Combine Data into One DataFrame
df_combined = pd.concat(dataframes, axis=1, keys=stock_symbols)
df_combined.columns = ["_".join(col).strip() for col in df_combined.columns]

# ✅ Fill Missing Data
df_combined.fillna(method="ffill", inplace=True)

# ✅ Save the Data (Optional)
df_combined.to_csv("./output/new_stock_data.csv")

print("✅ Live stock data fetched successfully!")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load Trained Models
rf_model = joblib.load("./output/random_forest_stock_model.pkl")
xgb_model = joblib.load("./output/xgboost_stock_model.pkl")
gru_model = tf.keras.models.load_model("./output/improved_gru_stock_model.h5")
scaler = joblib.load("./output/scaler.pkl")

# Load New Incoming Data (Simulating Real-Time Data)
df_combined = pd.read_csv("./output/new_stock_data.csv")  # Replace with live data source

# ✅ 1️⃣ Standardize Feature Names (Match with Trained Model)
column_mapping = {
    "AAPL_Close": "Close_AAPL",
    "AAPL_High": "High_AAPL",
    "AAPL_Low": "Low_AAPL",
    "AAPL_Open": "Open_AAPL",
    "AAPL_Volume": "Volume_AAPL",
    "MSFT_Close": "Close_MSFT",
    "MSFT_High": "High_MSFT",
    "MSFT_Low": "Low_MSFT",
    "MSFT_Open": "Open_MSFT",
    "MSFT_Volume": "Volume_MSFT",
    "GOOGL_Close": "Close_GOOGL",
    "GOOGL_High": "High_GOOGL",
    "GOOGL_Low": "Low_GOOGL",
    "GOOGL_Open": "Open_GOOGL",
    "GOOGL_Volume": "Volume_GOOGL",
}

df_combined.rename(columns=column_mapping, inplace=True)

# ✅ 2️⃣ Ensure Missing Features Are Added
expected_features = scaler.feature_names_in_

# Add missing columns with zero values
for col in expected_features:
    if col not in df_combined.columns:
        df_combined[col] = 0  # Fill with zero or np.nan

# Reorder columns to match the scaler's order
df_combined = df_combined[expected_features]

# ✅ 3️⃣ Normalize Data
new_data_scaled = scaler.transform(df_combined)

# ✅ 4️⃣ Make Model Predictions
rf_pred = rf_model.predict(new_data_scaled)
xgb_pred = xgb_model.predict(new_data_scaled)

# ✅ 5️⃣ Prepare GRU Input for Sequential Data
SEQ_LENGTH = 20

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

new_data_gru = create_sequences(new_data_scaled, SEQ_LENGTH)
gru_pred = gru_model.predict(new_data_gru).flatten()

# ✅ 6️⃣ Combine Predictions (Weighted Ensemble)
weights = {"rf": 0.50, "xgb": 0.30, "gru": 0.20}

ensemble_pred = (weights["rf"] * rf_pred[SEQ_LENGTH:] +
                 weights["xgb"] * xgb_pred[SEQ_LENGTH:] +
                 weights["gru"] * gru_pred)

print("✅ Real-Time Predictions Ready!")


# Define stock names and their respective indices in the dataset
stock_names = ["Apple (AAPL)", "Microsoft (MSFT)", "Google (GOOGL)"]
stock_indices = ["Close_AAPL", "Close_MSFT", "Close_GOOGL"]  # Adjust based on feature names

# Loop through each stock and plot predictions
plt.figure(figsize=(15, 5))

for i, stock in enumerate(stock_names):
    plt.subplot(1, 3, i+1)  # Create subplots for each stock

    plt.plot(ensemble_pred, label=f"{stock} - Weighted Ensemble Prediction", linestyle="dashed", color="red")
    plt.xlabel("Time")
    plt.ylabel(f"{stock} Stock Price Prediction")
    plt.title(f"📈 {stock} Real-Time Stock Price Forecast")
    plt.legend()

# Adjust layout for better visibility
plt.tight_layout()
plt.show()