import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def engineer_features(df: pd.DataFrame, file_id: str, target_col: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical features and scales numeric features.
    The target column is NOT scaled if it is numeric, but is encoded if it is categorical.
    """
    transformed_df = df.copy()
    preprocessors = {
        'scalers': {},
        'encoders': {}
    }
    
    for col in transformed_df.columns:
        dtype = transformed_df[col].dtype
        
        if col == target_col:
            # For target column:
            if not pd.api.types.is_numeric_dtype(dtype):
                # Encode if categorical
                transformed_df[col] = transformed_df[col].astype(str)
                encoder = LabelEncoder()
                transformed_df[col] = encoder.fit_transform(transformed_df[col])
                preprocessors['encoders'][col] = encoder
            # If numeric target, leave it as is (don't scale!)
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            # Scale numeric features
            scaler = StandardScaler()
            transformed_df[col] = scaler.fit_transform(transformed_df[[col]])
            preprocessors['scalers'][col] = scaler
        else:
            # Encode categorical features
            transformed_df[col] = transformed_df[col].astype(str)
            encoder = LabelEncoder()
            transformed_df[col] = encoder.fit_transform(transformed_df[col])
            preprocessors['encoders'][col] = encoder
            
    return transformed_df, preprocessors
