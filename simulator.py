# simulator.py
import numpy as np
import pandas as pd

def simulate_paths(prices_last, mu, cov, horizon=60, n_paths=1000, dt=1/252):
    tickers = mu.index
    L = np.linalg.cholesky(cov.values + 1e-12*np.eye(len(mu)))  # guard for numerics
    drift = (mu.values - 0.5*np.diag(cov.values)) * dt
    paths = np.zeros((horizon+1, len(tickers), n_paths))
    paths[0] = prices_last.values[:, None]

    for t in range(1, horizon+1):
        z = np.random.randn(len(tickers), n_paths)
        shocks = L @ z * np.sqrt(dt)
        r = drift[:, None] + shocks
        paths[t] = paths[t-1] * (1 + r)  # arithmetic returns

    idx = pd.RangeIndex(horizon+1, name="t")
    cols = pd.MultiIndex.from_product([tickers, range(n_paths)], names=["ticker", "path"])
    return pd.DataFrame(paths.reshape(horizon+1, -1), index=idx, columns=cols)

def inject_shock(paths_df, ticker, pct_shock, t=1):
    cols = [c for c in paths_df.columns if c[0] == ticker]
    paths_df.loc[t, cols] *= (1 + pct_shock)
    return paths_df

def inject_correlated_shocks(paths_df, base_ticker, pct_shock, corr_map, t=1):
    # corr_map like {"NVDA": {"TSLA": 0.4, "AMD": 0.7}, ...}
    base_cols = [c for c in paths_df.columns if c[0] == base_ticker]
    paths_df.loc[t, base_cols] *= (1 + pct_shock)
    for other_ticker, corr in corr_map.get(base_ticker, {}).items():
        cols = [c for c in paths_df.columns if c[0] == other_ticker]
        paths_df.loc[t, cols] *= (1 + pct_shock * corr)
    return paths_df
