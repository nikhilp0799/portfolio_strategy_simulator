import sys
import pandas as pd
from data_loader import load_stock_data
from feature_engineering import add_rolling_features
from model_forecasting import train_model, predict_model
from signal_generator import generate_signals
from backtester import backtest_strategy
from visualization_utils import generate_signal_overlay
from visualization_utils import plot_price_signals
from execution_agent import execution_agent
from visualization_utils import plot_equity_curve
from visualization_utils import plot_actual_vs_predicted
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
ticker = "CMCSA"
df = load_stock_data(ticker, "2022-01-01", "2024-01-01")

# Ensure 'Close' is a Series
price_series = df['Close']

# 2. Feature engineering
df = add_rolling_features(df)

# 3. Forecast
# Time-based train/test split
split_index = int(len(df) * 0.8)
X = df.drop(columns=['Close'])
y = df['Close']

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train on training data only
model = train_model(X_train, y_train, model_type='lstm')

# Predict on test data
pred_series = predict_model(model, X_test)  # Already aligned

# Evaluate and visualize
actual_trimmed = y_test.reindex(pred_series.index)
plot_actual_vs_predicted(
    actual_prices=actual_trimmed,
    predicted_prices=pred_series,
    dates=pred_series.index,
    save_path="actual_vs_predicted.png"
)




# Debug prediction diffs
diffs = pd.Series(pred_series).diff().dropna()
print("📈 Prediction Differences Summary:")
print(diffs.describe())

# 4. Generate signals — ensure prediction index matches price
lookback = 5
pred_series = pred_series[lookback:]

# Get correct index from original test set (not y_test, as it’s trimmed)
aligned_index = X_test.index[lookback : lookback + len(pred_series)]

# Ensure alignment
pred_series.index = aligned_index


signals = generate_signals(pred_series, threshold=0.005)


print("📍 Unique Signals:", signals.value_counts())

# 5. Backtest
performance = backtest_strategy(price_series, signals)
print("🔁 Backtest Performance:", {
    k: v if not isinstance(v, pd.Series) else v.tail(3).to_dict()  # show tail of long Series
    for k, v in performance.items()
})

# 6. Overlay for visualization
overlay = generate_signal_overlay(df['Close'], signals)
print("📊 Sample Overlay Data:", {k: overlay[k][:3] for k in overlay})

plot_price_signals(y_test, signals)


equity_curve, trade_log = execution_agent(df['Close'], signals, ticker)
plot_equity_curve(equity_curve)

print("📒 First 3 Trades:", trade_log[:3])


sys.exit()
