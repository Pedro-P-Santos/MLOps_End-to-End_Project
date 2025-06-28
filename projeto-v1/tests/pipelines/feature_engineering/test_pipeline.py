"""
This is a boilerplate test file for pipeline 'feature_engineering'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
from src.projeto_v1.pipelines.feature_engineering.nodes import run_feature_engineering
from src.projeto_v1.pipelines.feature_engineering.pipeline import create_pipeline

def test_run_feature_engineering_minimal():
    # Mock input data based on the required features in the pipeline
    df = pd.DataFrame({
    "age": [23, 45, 61, 30, 40, 50],
    "campaign": [1, 2, 3, 4, 5, 10], 
    "previous": [0, 1, 2, 0, 1, 3],
    "cons.price.idx": [92.893, 93.200, 93.994, 92.5, 93.0, 94.0],
    "cons.conf.idx": [-46.2, -40.0, -30.1, -42.0, -43.0, -44.0],
    "euribor3m": [1.344, 4.855, 0.635, 1.5, 3.3, 0.9],
    "education": ["university.degree", "basic.9y", "unknown", "high.school", "professional.course", "illiterate"],
    "default": ["no", "yes", "unknown", "no", "yes", "unknown"],
    "housing": ["yes", "no", "yes", "no", "yes", "no"],
    "loan": ["no", "yes", "no", "yes", "no", "yes"],
    "pdays": [-1, 99, -1, -1, 0, -1],
    "emp.var.rate": [1.1, -0.1, -1.8, 0.5, -0.2, 0.0],
    "nr.employed": [5099.1, 5191.0, 5001.0, 5100.0, 5200.0, 5000.0],
    "job": ["admin.", "student", "retired", "technician", "blue-collar", "management"],
    "poutcome": ["success", "failure", "nonexistent", "success", "failure", "nonexistent"],
    "marital": ["single", "married", "divorced", "married", "single", "divorced"]
})


    result = run_feature_engineering(df)

    # Check shape: should have many more columns
    assert result.shape[1] > df.shape[1], "No new features were created"

    # Check key engineered columns
    expected_cols = [
        "age_binned_quantile", "campaign_bin", "previous_bin",
        "cpi_top_value", "cpi_above_75th", "cci_top_value", "cci_above_75th",
        "euribor_bin", "education_mapped", "young_housing_loan", "middle_aged_loan",
        "contacted_before", "emp_rate_x_employed", "has_any_loan",
        "loan_risk_score", "is_student_or_retired", "successful_prev_contact",
        "economic_pressure_index", "marital_edu_combo"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing expected column: {col}"

    # Check categorical bin labels
    assert result["euribor_bin"].dtype.name == "category"
    
    # Validate transformations
    assert result["young_housing_loan"].iloc[0] == 1
    assert result["middle_aged_loan"].iloc[1] == 1
    assert result["contacted_before"].iloc[0] == 0  # pdays == -1
    assert result["contacted_before"].iloc[1] == 1  # pdays == 99

    # Check if pipeline object loads
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "data_engineering_node"
