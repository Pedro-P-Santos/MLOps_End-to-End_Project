"""
This is a boilerplate test file for pipeline 'data_expectations'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.data_expectations.nodes import validate_data

df = pd.read_csv(Path("data/02_intermediate/ingested_data.csv"))

def test_validated_data_is_dataframe():
    result = validate_data(df)
    assert isinstance(result, pd.DataFrame), "Validation should return a DataFrame"

def test_validated_data_has_no_nulls():
    result = validate_data(df)
    assert not result.isnull().values.any(), "Validated data contains nulls"

def test_validated_data_contains_target_column():
    result = validate_data(df)
    assert "y" in result.columns, "Validated data missing target column 'y'"
