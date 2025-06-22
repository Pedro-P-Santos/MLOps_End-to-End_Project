import pandas as pd
import yaml
from pathlib import Path
from src.projeto_v1.pipelines.data_cleaning.nodes import clean_bank_data

with open("conf/base/parameters_data_cleaning.yml", "r") as f:
    parameters = yaml.safe_load(f)["clean_data"]

df_sample = pd.read_csv(Path("data/02_intermediate/cleaned_data.csv")).sample(frac=0.1, random_state=42)

def test_clean_removes_duplicates():
    """Test that duplicates are removed if enabled."""
    df = pd.concat([df_sample, df_sample.iloc[0:1]])  # Add a duplicate
    cleaned = clean_bank_data(df, parameters)
    assert cleaned.shape[0] == df_sample.shape[0], "Duplicates were not removed"

def test_clean_replaces_999_pdays():
    """Test that 999 in pdays is replaced with the missing value placeholder."""
    df = df_sample.copy()
    df.loc[0, "pdays"] = 999
    cleaned = clean_bank_data(df, parameters)
    assert cleaned["pdays"].iloc[0] == parameters["pdays_missing_value"], "999 was not replaced"

def test_numeric_columns_downcasted():
    """Test that numeric columns are downcasted to save memory."""
    df = df_sample.copy()
    cleaned = clean_bank_data(df, parameters)
    int_cols = cleaned.select_dtypes(include=["int32"]).columns
    float_cols = cleaned.select_dtypes(include=["float32"]).columns
    assert len(int_cols) > 0 or len(float_cols) > 0, "Numeric columns were not downcasted"
