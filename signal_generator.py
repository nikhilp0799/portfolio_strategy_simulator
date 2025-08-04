import numpy as np
import pandas as pd

def generate_signals(predictions, threshold=0.5):
    if not isinstance(predictions, pd.Series):
        raise ValueError("predictions must be a pandas Series with a datetime index")

    diffs = predictions.diff().fillna(0)
    signals = []

    for change in diffs:
        if change > threshold:
            signals.append(1)   # BUY
        elif change < -threshold:
            signals.append(-1)  # SELL
        else:
            signals.append(0)   # HOLD

    signals_series = pd.Series(signals, index=predictions.index)

    print("📈 Prediction Differences Summary:")
    print(diffs.describe())

    return signals_series
