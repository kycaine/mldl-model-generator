from sklearn.base import BaseEstimator
import pandas as pd

def train_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """
    Trains the scikit-learn model on the provided data.
    """
    model.fit(X, y)
    return model
