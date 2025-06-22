import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.ingestion.nodes import ingestion

df_sample = pd.read_csv(
    Path("data/01_raw/bank-additional-full.csv"), sep=";", encoding="utf-8-sig"
).sample(frac=0.1, random_state=42)

def test_ingestion_is_dataframe():
    """Test that the ingestion function returns a DataFrame."""
    result = ingestion(df_sample)
    assert isinstance(result, pd.DataFrame), "Ingestion did not return a DataFrame"

def test_ingestion_column_count():
    """Test that the ingested DataFrame has the expected number of columns."""
    result = ingestion(df_sample)
    assert result.shape[1] == 21, "Unexpected number of columns in ingested data"

def test_target_column_unique_values():
    """Test that the target column has only 2 unique values."""
    result = ingestion(df_sample)
    assert result["y"].nunique() == 2, "Target column does not have 2 unique values"
