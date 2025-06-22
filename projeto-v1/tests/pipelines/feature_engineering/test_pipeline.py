"""
This is a boilerplate test file for pipeline 'feature_engineering'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.feature_engineering.nodes import run_feature_engineering

df = pd.read_csv(Path("data/02_intermediate/validated_data.csv"))

def test_feature_engineering_output_is_dataframe():
    result = run_feature_engineering(df)
    assert isinstance(result, pd.DataFrame), "Feature engineering must return a DataFrame"

def test_feature_engineering_creates_expected_features():
    result = run_feature_engineering(df)
    expected_columns = [
        "age_binned_quantile", "campaign_bin", "previous_bin", "cpi_top_value", "cpi_above_75th",
        "cci_top_value", "cci_above_75th", "euribor_bin", "education_mapped", "young_housing_loan",
        "middle_aged_housing_loan", "senior_housing_loan", "young_loan", "middle_aged_loan",
        "senior_loan", "contacted_before", "emp_rate_x_employed", "has_any_loan",
        "loan_risk_score", "is_student_or_retired", "successful_prev_contact",
        "economic_pressure_index", "marital_edu_combo"
    ]
    missing = [col for col in expected_columns if col not in result.columns]
    assert not missing, f"Missing engineered columns: {missing}"

def test_engineered_data_no_nulls_in_key_features():
    result = run_feature_engineering(df)
    critical = ["campaign_bin", "previous_bin", "euribor_bin"]
    for col in critical:
        assert result[col].isnull().sum() == 0, f"{col} contains nulls"
