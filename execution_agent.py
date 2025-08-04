import pandas as pd

def execution_agent(prices, signals, initial_capital=100000, position_size_pct=0.1):
    """
    Executes trades based on signals with capital and position tracking.

    Args:
        prices (pd.Series): Price series.
        signals (pd.Series): Signal series (-1, 0, 1).
        initial_capital (float): Starting cash.
        position_size_pct (float): Percentage of capital to allocate per trade.

    Returns:
        equity_curve (pd.Series): Portfolio value over time.
        trade_log (list): List of executed trades.
    """
    cash = initial_capital
    holdings = 0
    portfolio_value = []
    trade_log = []

    for date, signal in signals.items():
        price = prices.loc[date]
        if isinstance(price, pd.Series):
            price = price.values[0]


        # Buy Signal
        if signal == 1 and cash > 0:
            size = (cash * position_size_pct) // price
            if size > 0:
                holdings += size
                cost = size * price
                cash -= cost
                trade_log.append({
                    "date": date,
                    "type": "BUY",
                    "price": price,
                    "shares": size,
                    "cash": cash
                })

        # Sell Signal
        elif signal == -1 and holdings > 0:
            revenue = holdings * price
            cash += revenue
            trade_log.append({
                "date": date,
                "type": "SELL",
                "price": price,
                "shares": holdings,
                "cash": cash
            })
            holdings = 0

        # Track current portfolio value
        total_value = cash + holdings * price
        portfolio_value.append((date, total_value))

    # Convert to DataFrame/Series
    equity_df = pd.DataFrame(portfolio_value, columns=["date", "equity"])
    equity_df.set_index("date", inplace=True)
    equity_curve = equity_df["equity"]

    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_csv("trade_log.csv", index=False)


    return equity_curve, trade_log
