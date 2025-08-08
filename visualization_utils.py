import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------- Signal Marker Extraction --------
def generate_signal_overlay(prices, signals):
    buy_markers = []
    sell_markers = []
    buy_prices = []
    sell_prices = []

    for i in range(1, len(signals)):
        prev_signal = signals.iloc[i - 1]
        curr_signal = signals.iloc[i]
        price_at_i = prices.iloc[i]

        if prev_signal <= 0 and curr_signal == 1:
            buy_markers.append(i)
            buy_prices.append(price_at_i)
        elif prev_signal >= 0 and curr_signal == -1:
            sell_markers.append(i)
            sell_prices.append(price_at_i)

    return {
        "dates": list(prices.index),
        "prices": list(prices),
        "buy_markers": buy_markers,
        "buy_prices": buy_prices,
        "sell_markers": sell_markers,
        "sell_prices": sell_prices,
    }

# -------- Plotting Price + Signals --------
def plot_price_signals(prices, signals, save_path="price_signals.png"):
    buy_signals = (signals == 1)
    sell_signals = (signals == -1)

    # Align prices to signals index to prevent indexing error
    aligned_prices = prices.loc[signals.index]

    plt.figure(figsize=(14, 6))
    plt.plot(aligned_prices.index, aligned_prices, label='Price')

    plt.plot(aligned_prices[buy_signals].index, aligned_prices[buy_signals], '^', markersize=10, color='g', label='Buy')
    plt.plot(aligned_prices[sell_signals].index, aligned_prices[sell_signals], 'v', markersize=10, color='r', label='Sell')

    plt.title("Price with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------- Plotting Equity Curve --------
def plot_equity_curve(equity_curve, save_path="equity_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)  # Save to file
    plt.close()  # Close the figure to free memory


# -------- Plotting Actual vs Predicted Close Prices --------
def plot_actual_vs_predicted(dates, actual_prices, predicted_prices, save_path="actual_vs_predicted.png"):
    # Compute evaluation metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    # Plot actual vs predicted
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual_prices, label='Actual Close', color='blue', linewidth=2)
    plt.plot(dates, predicted_prices, label='Predicted Close', color='orange', linestyle='--', linewidth=2)

    # Title + metrics
    plt.title(f"Actual vs Predicted Close Prices\nMSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

