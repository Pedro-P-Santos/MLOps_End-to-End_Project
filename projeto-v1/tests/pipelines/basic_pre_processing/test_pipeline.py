"""
This is a boilerplate test file for pipeline 'basic_pre_processing'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
"""
This is a boilerplate test file for pipeline 'basic_pre_processing'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import numpy as np
from src.projeto_v1.pipelines.basic_pre_processing.nodes import preprocess_train_test, encode_targets
from src.projeto_v1.pipelines.basic_pre_processing.pipeline import create_pipeline


def test_preprocess_train_test_and_encode_targets():
    # Dummy train and test sets
    X_train = pd.DataFrame({
        # Numerical
        "age": [25, 45, np.nan],
        "campaign": [1, 2, 3],
        "pdays": [999, 5, 2],
        "previous": [0, 1, 2],
        "emp.var.rate": [1.1, -0.1, np.nan],
        "cons.price.idx": [93.2, 93.4, 93.1],
        "cons.conf.idx": [-36.4, -42.0, -40.0],
        "euribor3m": [4.86, 1.34, 0.63],
        "nr.employed": [5191, 5228, 5195],
        "emp_rate_x_employed": [5730.1, 5432.1, 5012.4],
        "loan_risk_score": [0.1, 0.5, 0.8],
        "economic_pressure_index": [1.0, 2.5, 3.0],

        # Binary (object and numeric)
        "default": ["no", "yes", "unknown"],
        "housing": ["yes", "no", "unknown"],
        "loan": ["no", "no", "yes"],
        "cpi_above_75th": [1, 0, 1],
        "cci_above_75th": [0, 1, 0],
        "young_housing_loan": [0, 1, 0],
        "middle_aged_housing_loan": [1, 0, 1],
        "senior_housing_loan": [0, 0, 0],
        "young_loan": [0, 0, 1],
        "middle_aged_loan": [1, 1, 0],
        "senior_loan": [0, 0, 0],
        "contacted_before": [1, 0, 1],
        "is_student_or_retired": [0, 1, 0],
        "successful_prev_contact": [0, 1, 1],
        "has_any_loan": [1, 1, 1],

        # High cardinality
        "job": ["admin.", "technician", "services"],
        "education": ["university.degree", "high.school", "basic.9y"],
        "campaign_bin": ["low", "medium", "high"],
        "cci_top_value": ["yes", "no", "yes"],
        "marital_edu_combo": ["single_degree", "married_school", "divorced_basic"],

        # Low cardinality
        "marital": ["single", "married", "divorced"],
        "poutcome": ["success", "failure", "nonexistent"],
        "age_binned_quantile": ["young", "middle", "senior"],
        "previous_bin": ["none", "some", "many"],
        "cpi_top_value": ["low", "medium", "high"],
        "euribor_bin": ["low", "medium", "high"],
        "education_mapped": ["basic", "intermediate", "advanced"],

        # Features to drop
        "duration": [100, 200, 300],
        "contact": ["cellular", "telephone", "unknown"],
        "month": ["may", "jul", "aug"],
        "day_of_week": ["mon", "tue", "wed"],
        "contact_efficiency": [0.5, 0.7, 0.9]
    })

    X_test = X_train.copy()

    # Dummy target
    y_train = pd.DataFrame({"y": ["yes", "no", "yes"]})
    y_test = pd.DataFrame({"y": ["no", "no", "yes"]})

    # Run preprocessing
    X_train_proc, X_test_proc = preprocess_train_test(X_train.copy(), X_test.copy())
    assert isinstance(X_train_proc, pd.DataFrame)
    assert isinstance(X_test_proc, pd.DataFrame)

    # Expected shape checks
    assert X_train_proc.shape[0] == 3
    assert X_test_proc.shape[0] == 3

    # Should not contain dropped features
    dropped_cols = ["duration", "contact", "month", "day_of_week", "contact_efficiency"]
    for col in dropped_cols:
        assert col not in X_train_proc.columns

    # Should include encoded versions of categorical features
    onehot_columns = [col for col in X_train_proc.columns if "_" in col]
    assert any("marital" in col for col in onehot_columns)
    assert any("poutcome" in col for col in onehot_columns)

    # Run target encoding
    y_train_enc, y_test_enc = encode_targets(y_train.copy(), y_test.copy())
    assert (y_train_enc == pd.Series([1, 0, 1])).all()
    assert (y_test_enc == pd.Series([0, 0, 1])).all()

    # Pipeline node check
    pipeline = create_pipeline()
    node_names = [node.name for node in pipeline.nodes]
    assert "preprocess_data_node" in node_names
    assert "encode_targets_node" in node_names

