
import pandas as pd
import matplotlib.pyplot as plt

# visualization_utils.py

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



def plot_price_signals(prices, signals):
    buy_signals = (signals == 1)
    sell_signals = (signals == -1)

    plt.figure(figsize=(14, 6))
    plt.plot(prices.index, prices, label='Price')
    plt.plot(prices[buy_signals].index, prices[buy_signals], '^', markersize=10, color='g', label='Buy')
    plt.plot(prices[sell_signals].index, prices[sell_signals], 'v', markersize=10, color='r', label='Sell')
    plt.title("Price with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_equity_curve(equity_curve, save_path="equity_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()  # Displays it


