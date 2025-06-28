"""
This is a boilerplate test file for pipeline 'data_cleaning'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.projeto_v1.pipelines.data_cleaning.nodes import clean_bank_data
from src.projeto_v1.pipelines.data_cleaning.pipeline import create_pipeline

def test_clean_bank_data_with_project_config():
    # Load cleaning config from parameters.yml
    config_path = Path("conf/base/parameters.yml")
    with config_path.open("r") as file:
        parameters = yaml.safe_load(file)
    config = parameters["data_cleaning_config"]

    # Create dummy DataFrame with duplicate, 999 in 'pdays', numeric + categorical columns
    df = pd.DataFrame({
        "age": [30, 30, 40],
        "duration": [1000.0, 1000.0, 1500.5],
        "pdays": [999, 999, 5],
        "job": ["admin.", "admin.", "unknown"],
        "y": ["yes", "yes", "no"]
    })

    cleaned_df = clean_bank_data(df, config)

    # Expect duplicate removal
    assert cleaned_df.shape[0] == 2

    # Expect pdays 999 to be replaced with config value
    assert config["pdays_missing_value"] in cleaned_df["pdays"].values
    assert 999 not in cleaned_df["pdays"].values

    # Expect numeric downcast
    assert cleaned_df["age"].dtype.name.startswith("int")
    assert cleaned_df["duration"].dtype.name.startswith("float")

    # 'unknown' should not be removed from categorical columns
    assert "unknown" in cleaned_df["job"].values

    # Check no column was dropped
    assert set(cleaned_df.columns) == {"age", "duration", "pdays", "job", "y"}

    # Check pipeline node metadata
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "clean_bank_data_node"
