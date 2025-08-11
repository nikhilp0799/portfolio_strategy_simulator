# simulate_main.py
import pandas as pd
from data_loader import load_stock_data
from simulator import simulate_paths, inject_shock
from feature_engineering import add_rolling_features
from model_forecasting import train_model, predict_model
from signal_generator import generate_signals
from backtester import backtest_strategy
from execution_agent import execution_agent
from visualization_utils import plot_equity_curve, plot_price_signals, plot_actual_vs_predicted

# ---- Config ----
tickers = ["MSFT", "NVDA", "TSLA"]
start, end = "2021-01-01", "2023-01-01"
model_type = "lstm"   # flip between "lstm", "rf", "xgb" with no code changes
horizon = 60
n_paths = 200
shock_day = 10
shock = -0.15         # -15% shock to NVDA on day 10 (optional)

# 1) Load history and estimate drift/cov
hist = {tk: load_stock_data(tk, start, end)["Close"].rename(tk) for tk in tickers}
prices_hist = pd.concat(hist.values(), axis=1).dropna()
rets = prices_hist.pct_change().dropna()

mu = rets.mean() * 252
cov = rets.cov() * 252
prices_last = prices_hist.iloc[-1]

# 2) Simulate correlated future paths
paths = simulate_paths(prices_last, mu, cov, horizon=horizon, n_paths=n_paths)

# (optional) inject a shock to NVDA on day 10
paths = inject_shock(paths, "NVDA", pct_shock=shock, t=shock_day)

# 3) Pick ONE path for ONE ticker to push through your pipeline
path_id = 0
tk = "MSFT"
ser = paths[(tk, path_id)].rename("Close")  # index is t = 0..horizon (integers)
ser.index.name = "t"

# 4) Build features on simulated price series
df_sim = pd.DataFrame(ser)
df_sim = add_rolling_features(df_sim)

# 5) Time split on simulated data
split_idx = int(len(df_sim) * 0.8)
X, y = df_sim.drop(columns=["Close"]), df_sim["Close"]
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 6) Train & predict (model_forecasting returns a Series with the correct index for any model)
model = train_model(X_train, y_train, model_type=model_type)
pred_series = predict_model(model, X_test)  # <-- already aligned to X_test (for LSTM it starts at lookback)

# 7) Align actuals to predictions once
actual_aligned = y_test.reindex(pred_series.index)

# optional: see how the reg metrics look on the sim path
plot_actual_vs_predicted(
    actual_prices=actual_aligned,
    predicted_prices=pred_series,
    dates=pred_series.index,
    save_path="sim_actual_vs_predicted.png",
)

# 8) Signals on predictions (index-safe)
signals = generate_signals(pred_series, threshold=0.005)

# 9) Backtest on the same aligned window
prices_for_bt = actual_aligned
perf = backtest_strategy(prices_for_bt, signals)

# 10) Execution and plots (same aligned window)
equity, trades = execution_agent(prices_for_bt, signals, ticker=tk)
plot_price_signals(prices_for_bt, signals)
plot_equity_curve(equity)

print("🔁 Sim Backtest (tail):", {k: (v.tail(3).to_dict() if isinstance(v, pd.Series) else v) for k, v in perf.items()})
print("📒 First 3 Trades:", trades[:3])
