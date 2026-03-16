import joblib
import os
from sklearn.base import BaseEstimator

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def export_model(model: BaseEstimator, preprocessors: dict, file_id: str, model_name: str) -> str:
    """
    Saves the trained model and preprocessors to a .joblib file and returns the path.
    """
    # Create a wrapper dictionary or pipeline object to save
    export_object = {
        "model": model,
        "preprocessors": preprocessors,
        "model_name": model_name
    }
    
    filename = f"{file_id}_{model_name.replace(' ', '_')}.joblib"
    file_path = os.path.join(MODELS_DIR, filename)
    
    joblib.dump(export_object, file_path)
    return file_path
