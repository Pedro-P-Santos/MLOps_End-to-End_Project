"""
This is a boilerplate test file for pipeline 'ingestion'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.ingestion.nodes import ingestion

df = pd.read_csv(Path("data/01_raw/bank-additional-full.csv"), sep=";", encoding="utf-8-sig")

def test_ingestion_returns_dataframe():
    result = ingestion(df)
    assert isinstance(result, pd.DataFrame), "Ingestion should return a DataFrame"

def test_ingested_data_column_count():
    result = ingestion(df)
    assert result.shape[1] == 21, "Expected 21 columns in ingested data"

def test_ingested_data_has_target_column():
    result = ingestion(df)
    assert "y" in result.columns, "Missing target column 'y'"

def test_ingested_data_no_nulls():
    result = ingestion(df)
    assert not result.isnull().values.any(), "Ingested data contains null values"