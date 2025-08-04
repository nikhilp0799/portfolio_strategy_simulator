import numpy as np
import pandas as pd

def backtest_strategy(prices: pd.Series, signals: pd.Series, initial_cash=1.0):
    """
    Backtests a trading strategy based on price data and buy/sell signals.
    """
    # Ensure both are Series
    prices = prices.squeeze()
    signals = signals.squeeze()

    # Daily returns
    returns = prices.pct_change().fillna(0)

    # Shift signals (yesterday’s signal applies today)
    aligned_signals = signals.shift(1).reindex(returns.index).fillna(0)

    # Element-wise strategy returns
    strategy_returns = aligned_signals * returns

    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_cash

    # Sharpe ratio
    std = strategy_returns.std()
    sharpe_ratio = (strategy_returns.mean() / std * np.sqrt(252)) if std > 0 else np.nan

    # Max drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # CAGR
    total_days = (returns.index[-1] - returns.index[0]).days
    cagr = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0])**(365.0 / total_days) - 1 if total_days > 0 else 0

    return {
        'Cumulative Return': cumulative_returns,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'CAGR': cagr,
    }
