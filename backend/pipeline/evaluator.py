from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, model_type: str) -> dict:
    """
    Evaluates the trained model and returns metrics and plot data.
    """
    predictions = model.predict(X)
    
    # Create sample data for plotting (max 50 points)
    sample_size = min(50, len(y))
    plot_data = []
    
    if isinstance(y, pd.Series):
        y_actual = y.tolist()
    else:
        y_actual = list(y)
        
    y_pred = list(predictions)
    
    for i in range(sample_size):
        a_val = y_actual[i]
        p_val = y_pred[i]
        
        # Convert to float if it looks like a number, otherwise keep as is (for categories)
        try:
            if isinstance(a_val, (np.number, int, float, np.integer, np.floating)):
                a_val = float(a_val)
            if isinstance(p_val, (np.number, int, float, np.integer, np.floating)):
                p_val = float(p_val)
        except:
            pass
            
        plot_data.append({
            "index": i,
            "actual": a_val,
            "predicted": p_val
        })

    if model_type == 'classification':
        return {
            "metrics": {
                "accuracy": float(accuracy_score(y, predictions)),
                "precision": float(precision_score(y, predictions, average='weighted', zero_division=0)),
                "recall": float(recall_score(y, predictions, average='weighted', zero_division=0))
            },
            "plot_data": plot_data
        }
    elif model_type == 'regression':
        mse = mean_squared_error(y, predictions)
        return {
            "metrics": {
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y, predictions))
            },
            "plot_data": plot_data
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
