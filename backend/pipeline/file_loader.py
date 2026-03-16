import os
import pandas as pd

def load_file(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV or Excel file into a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.csv':
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            # Fallback for different encodings or separators
            return pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
    elif ext.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
