# dependence.py
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def compute_returns(prices):
    return prices.pct_change().dropna()

def ledoit_wolf_cov(returns):
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    mu  = pd.Series(returns.mean().values, index=returns.columns)
    return mu, cov

def ewma_cov(returns, lam=0.94):
    # RiskMetrics-style covariance
    x = returns.values
    mu = returns.mean()
    S = np.cov(x, rowvar=False) * 0  # init zeros
    w = (1 - lam) * lam ** np.arange(x.shape[0]-1, -1, -1)
    w = w / w.sum()
    x_c = x - mu.values
    for t, wt in enumerate(w):
        S += wt * np.outer(x_c[t], x_c[t])
    cov = pd.DataFrame(S, index=returns.columns, columns=returns.columns)
    return mu, cov
