import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by handling missing values and removing duplicates.
    """
    cleaned_df = df.copy()
    
    # 1. Remove complete duplicate rows
    cleaned_df.drop_duplicates(inplace=True)
    
    # 2. Handle missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isna().sum() > 0:
            dtype = cleaned_df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                # For numeric, fill with median
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
            else:
                # For categorical/text, fill with mode or 'Unknown'
                if not cleaned_df[col].mode().empty:
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                    
    # Optional: Remove columns that are 100% missing after processing (rare if handled)
    cleaned_df.dropna(axis=1, how='all', inplace=True)
    
    return cleaned_df
