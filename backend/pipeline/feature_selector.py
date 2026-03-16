import pandas as pd

def select_features(df: pd.DataFrame, target_column: str, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates the dataset into target variable (y) and feature variables (X).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in dataset.")
            
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y
