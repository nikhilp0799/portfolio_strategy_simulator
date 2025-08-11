import sys
import pandas as pd
from data_loader import load_stock_data
from feature_engineering import add_rolling_features
from model_forecasting import train_model, predict_model
from signal_generator import generate_signals
from backtester import backtest_strategy
from execution_agent import execution_agent
from visualization_utils import (
    generate_signal_overlay,
    plot_price_signals,
    plot_equity_curve,
    plot_actual_vs_predicted,
)

# 1) Data
ticker = "CMCSA"
prices = load_stock_data(ticker, "2022-01-01", "2024-01-01")
# prices has one column named "CMCSA"
close = prices[ticker].rename("Close")
df = close.to_frame()  # now df has a 'Close' column like before


# 2) Features
df = add_rolling_features(df)

# 3) Split
split_index = int(len(df) * 0.8)
X = df.drop(columns=["Close"])
y = df["Close"]
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 4) Train (swap 'lstm' <-> 'rf' <-> 'xgb' freely)
model = train_model(X_train, y_train, model_type="lstm")

# 5) Predict (returns a pd.Series with the correct index for the chosen model)
pred_series = predict_model(model, X_test)  # <- DO NOT slice this

# 6) Align actuals to predictions (one line, no lookback hacks)
actual_aligned = y_test.reindex(pred_series.index)

# 7) Plot actual vs predicted (metrics inside the plot)
plot_actual_vs_predicted(
    actual_prices=actual_aligned,
    predicted_prices=pred_series,
    dates=pred_series.index,
    save_path="actual_vs_predicted.png",
)

# 8) Signals (works on the pred_series index for any model)
signals = generate_signals(pred_series, threshold=0.005)
print("📍 Unique Signals:", signals.value_counts())

# 9) Backtest on the aligned test prices only
prices_for_bt = actual_aligned  # y_test trimmed to pred index
performance = backtest_strategy(prices_for_bt, signals)
print("🔁 Backtest Performance:", {
    k: v if not isinstance(v, pd.Series) else v.tail(3).to_dict()
    for k, v in performance.items()
})

# 10) Visuals on the same aligned window
overlay = generate_signal_overlay(prices_for_bt, signals)
print("📊 Sample Overlay Data:", {k: overlay[k][:3] for k in overlay})

plot_price_signals(prices_for_bt, signals)

# 11) Execution agent runs on the same aligned prices/signals
equity_curve, trade_log = execution_agent(prices_for_bt, signals, ticker)
plot_equity_curve(equity_curve)
print("📒 First 3 Trades:", trade_log[:3])

sys.exit()
