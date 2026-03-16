import pandas as pd

def detect_schema(df: pd.DataFrame) -> dict:
    """
    Detects column names, data types, and missing values count.
    """
    schema = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isna().sum())
        unique_count = int(df[col].nunique())
        
        # Simplify data type for frontend
        if 'int' in dtype or 'float' in dtype:
            simple_type = 'numeric'
        elif 'datetime' in dtype:
            simple_type = 'datetime'
        else:
            simple_type = 'categorical'
            
        schema.append({
            "name": col,
            "type": simple_type,
            "raw_type": dtype,
            "missing_values": missing_count,
            "unique_values": unique_count
        })
        
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": schema
    }
