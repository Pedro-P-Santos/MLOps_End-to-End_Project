"""
This is a boilerplate test file for pipeline 'ingestion'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from src.projeto_v1.pipelines.ingestion.nodes import ingestion
from src.projeto_v1.pipelines.ingestion.pipeline import create_pipeline

def test_ingestion_pipeline():
    df = pd.DataFrame({
        "age": [25, 35],
        "job": ["admin.", "technician"],
        "marital": ["married", "single"],
        "education": ["secondary", "tertiary"],
        "default": ["no", "yes"],
        "housing": ["yes", "no"],
        "loan": ["no", "no"],
        "contact": ["cellular", "telephone"],
        "month": ["may", "jun"],
        "day_of_week": ["mon", "tue"],
        "duration": [100, 200],
        "campaign": [1, 2],
        "pdays": [999, 3],
        "previous": [0, 1],
        "poutcome": ["nonexistent", "success"],
        "emp.var.rate": [1.1, -0.1],
        "cons.price.idx": [93.2, 94.5],
        "cons.conf.idx": [-36.4, -41.8],
        "euribor3m": [4.857, 1.344],
        "nr.employed": [5191, 5099],
        "y": ["yes", "no"]
    })
    
    result = ingestion(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 21)
    assert result["y"].nunique() == 2

    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "ingestion_node"

