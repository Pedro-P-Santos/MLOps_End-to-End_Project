"""
This is a boilerplate test file for pipeline 'data_expectations'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from src.projeto_v1.pipelines.data_expectations.nodes import build_custom_expectations_for_rawdata
from src.projeto_v1.pipelines.data_expectations.pipeline import create_pipeline

def test_build_custom_expectations_for_rawdata():
    df = pd.read_csv("data/02_intermediate/ingested_data.csv")

    result = build_custom_expectations_for_rawdata(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (41188, 21)
    assert result["y"].nunique() == 2
    assert result["pdays"].mode().iloc[0] == 999
    assert 25 <= result["age"].mode().iloc[0] <= 35
    
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "data_expectations_raw_node"