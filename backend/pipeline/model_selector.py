from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator

def select_model(model_type: str, model_name: str) -> BaseEstimator:
    """
    Instantiates the appropriate scikit-learn model.
    """
    if model_type == 'classification':
        if model_name == 'Logistic Regression':
            return LogisticRegression()
        elif model_name == 'Random Forest Classifier':
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unsupported classification model: {model_name}")
            
    elif model_type == 'regression':
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Random Forest Regressor':
            return RandomForestRegressor()
        else:
            raise ValueError(f"Unsupported regression model: {model_name}")
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
