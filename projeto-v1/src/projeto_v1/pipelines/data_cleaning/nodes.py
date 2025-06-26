"""
This module contains the data cleaning logic for the Bank Marketing dataset.
The cleaning includes:
- Removing duplicates (if configured)
- Replacing '999' in `pdays` with -1
- Ensuring correct data types (downcasting numerics)
- Keeping 'unknown' values for categorical columns as-is
"""

import pandas as pd

def clean_bank_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cleans the input dataframe based on the specified config.
    
    Args:
        df (pd.DataFrame): Raw input data.
        config (dict): Cleaning configuration including:
            - drop_duplicates (bool)
            - pdays_missing_value (int)

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    original_shape = df.shape

    # Remove duplicates if required
    if config.get("drop_duplicates", True):
        df = df.drop_duplicates()
        assert df.shape[0] <= original_shape[0], "Duplicate removal failed"

    # Replace 999 in 'pdays' with configured missing value
    if "pdays" in df.columns:
        df["pdays"] = df["pdays"].replace(999, config.get("pdays_missing_value", -1))

    # Downcast numeric columns to reduce memory usage
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df
