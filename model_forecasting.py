from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_model(X_train, y_train, model_type='rf', **kwargs):
    if X_train.isnull().values.any() or y_train.isnull().values.any():
        raise ValueError("Training data contains NaNs")

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
    elif model_type == 'xgb':
        model = XGBRegressor(random_state=42, **kwargs)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model


def predict_model(model, X_test):
    return model.predict(X_test)
