from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

def train_model(X_train, y_train, model_type='rf', **kwargs):
    if X_train.isnull().values.any() or y_train.isnull().values.any():
        raise ValueError("Training data contains NaNs")

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        model.fit(X_train, y_train)
        return model

    elif model_type == 'xgb':
        model = XGBRegressor(random_state=42, **kwargs)
        model.fit(X_train, y_train)
        return model

    elif model_type == 'lstm':
        lookback = 5  # Number of past days to look back

        # Scale inputs
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_x.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

        # Build rolling sequences
        X_seq = []
        y_seq = []
        for i in range(lookback, len(X_scaled)):
            X_seq.append(X_scaled[i - lookback:i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)            # shape: (samples, time_steps=5, features)
        y_seq = np.array(y_seq)            # shape: (samples, 1)

        # Define model
        model = Sequential([
            Input(shape=(lookback, X_seq.shape[2])),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_seq, y_seq, epochs=50, verbose=0)

        return {
            'model': model,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'lookback': lookback
        }


    else:
        raise ValueError("Unsupported model type")



def predict_model(model, X_test):
    if isinstance(model, dict) and 'model' in model:  # LSTM case
        scaler_x = model['scaler_x']
        scaler_y = model['scaler_y']
        lookback = model['lookback']

        X_scaled = scaler_x.transform(X_test)

        X_seq = [X_scaled[i - lookback:i] for i in range(lookback, len(X_scaled))]
        X_seq = np.array(X_seq)

        predictions_scaled = model['model'].predict(X_seq, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

        aligned_index = X_test.index[lookback:]
        return pd.Series(predictions, index=aligned_index)

    else:  # RF or XGB
        predictions = model.predict(X_test)
        return pd.Series(predictions, index=X_test.index)



