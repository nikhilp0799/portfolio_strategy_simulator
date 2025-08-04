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

# 1. Load data
df = load_stock_data("AAPL", "2022-01-01", "2023-01-01")

# Ensure 'Close' is a Series
price_series = df['Close']

# 2. Feature engineering
df = add_rolling_features(df)

# 3. Forecast
X = df.drop(columns=['Close'])
y = df['Close']
model = train_model(X, y)
predictions = predict_model(model, X)

# 👇 Debug prediction diffs
diffs = pd.Series(predictions).diff().dropna()
print("📈 Prediction Differences Summary:")
print(diffs.describe())

# 4. Generate signals — ensure prediction index matches price
pred_series = pd.Series(predictions, index=df.index)
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

plot_price_signals(df['Close'], signals)


equity_curve, trade_log = execution_agent(df['Close'], signals)
plot_equity_curve(equity_curve)

print("📒 First 3 Trades:", trade_log[:3])


sys.exit()
