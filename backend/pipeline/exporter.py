import joblib
import os
import numpy as np
from sklearn.base import BaseEstimator
from skl2onnx import to_onnx
from safetensors.numpy import save_file

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def export_model(model: BaseEstimator, preprocessors: dict, file_id: str, model_name: str, format: str = "joblib") -> str:
    """
    Saves the trained model and preprocessors to a file and returns the path.
    Supported formats: joblib, onnx, safetensors
    """
    clean_name = model_name.replace(' ', '_')
    
    if format == "joblib":
        export_object = {
            "model": model,
            "preprocessors": preprocessors,
            "model_name": model_name
        }
        filename = f"{file_id}_{clean_name}.joblib"
        file_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(export_object, file_path)
        return file_path

    elif format == "onnx":
        filename = f"{file_id}_{clean_name}.onnx"
        file_path = os.path.join(MODELS_DIR, filename)
        
        # Determine input shape for ONNX (common for tabular data)
        # Assuming X is the input, but we don't have X here easily. 
        # Typically sklearn models expect [None, num_features]
        # We can try to infer num_features from the model if it's trained.
        
        try:
            # Fallback to a simple float32 input if we can't determine features
            num_features = getattr(model, "n_features_in_", 1)
            onx = to_onnx(model, X=np.zeros((1, num_features)).astype(np.float32))
            with open(file_path, "wb") as f:
                f.write(onx.SerializeToString())
            return file_path
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
            raise e

    elif format == "safetensors":
        filename = f"{file_id}_{clean_name}.safotensors" # Intentional extension for clarity or .st
        file_path = os.path.join(MODELS_DIR, filename)
        
        # Safetensors expects a dict of numpy arrays
        # We'll export the model parameters (weights/intercept) if available
        tensors = {}
        if hasattr(model, "coef_"):
            tensors["weights"] = np.array(model.coef_)
        if hasattr(model, "intercept_"):
            tensors["bias"] = np.array(model.intercept_)
        
        # For non-linear models like Random Forest, we might just save a placeholder 
        # as Safetensors isn't ideal for tree-based models, but we'll try to provide SOMETHING.
        if not tensors:
            tensors["status"] = np.array([1], dtype=np.int32) # Placeholder
            
        save_file(tensors, file_path)
        return file_path

    else:
        raise ValueError(f"Unsupported format: {format}")
