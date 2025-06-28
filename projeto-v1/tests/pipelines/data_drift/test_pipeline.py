"""
This is a boilerplate test file for pipeline 'data_drift'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using pytest framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import numpy as np
import pytest

import sys
from unittest.mock import MagicMock

# Mock evidently package and submodules to avoid import errors at test collection
sys.modules['evidently'] = MagicMock()
sys.modules['evidently.report'] = MagicMock()
sys.modules['evidently.metric_preset'] = MagicMock()
sys.modules['evidently.core'] = MagicMock()
sys.modules['evidently.legacy'] = MagicMock()

from src.projeto_v1.pipelines.data_drift.nodes import run_all_drift_checks
from src.projeto_v1.pipelines.data_drift.pipeline import create_pipeline

@pytest.fixture(scope="module")
def dummy_drift_data():
    n_samples = 50
    np.random.seed(42)
    data = pd.DataFrame({
        'is_student_or_retired': np.random.randint(0, 2, n_samples),
        'cpi_top_value_92.893': np.random.rand(n_samples),
        'poutcome_success': np.random.randint(0, 2, n_samples),
        'cci_top_value': np.random.rand(n_samples) * 100,
        'marital_single': np.random.randint(0, 2, n_samples),
        'previous_bin_1': np.random.randint(0, 2, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'age_binned_quantile_0.0': np.random.randint(0, 2, n_samples),
        'education_mapped_basic_education': np.random.randint(0, 2, n_samples),
        'poutcome_nonexistent': np.random.randint(0, 2, n_samples),
        'middle_aged_housing_loan': np.random.randint(0, 2, n_samples),
        'previous_bin_0': np.random.randint(0, 2, n_samples),
        'previous': np.random.randint(0, 10, n_samples),
        'euribor_bin_high': np.random.randint(0, 2, n_samples),
        'emp.var.rate': np.random.uniform(-3.0, 3.0, n_samples),
        'age_binned_quantile_4.0': np.random.randint(0, 2, n_samples),
        'cpi_top_value_93.994': np.random.rand(n_samples),
        'cpi_top_value_93.918': np.random.rand(n_samples),
        'euribor_bin_very_high': np.random.randint(0, 2, n_samples),
        'loan_risk_score': np.random.uniform(300, 850, n_samples),
        'nr.employed': np.random.uniform(5000, 6000, n_samples),
        'education_mapped_higher_education': np.random.randint(0, 2, n_samples),
        'cons.conf.idx': np.random.uniform(-50, 50, n_samples),
        'pdays': np.random.randint(-1, 30, n_samples),
        'successful_prev_contact': np.random.randint(0, 2, n_samples),
        'senior_loan': np.random.randint(0, 2, n_samples),
        'age_binned_quantile_2.0': np.random.randint(0, 2, n_samples),
        'senior_housing_loan': np.random.randint(0, 2, n_samples),
        'cci_above_75th': np.random.randint(0, 2, n_samples),
        'young_housing_loan': np.random.randint(0, 2, n_samples),
        'euribor_bin_very_low': np.random.randint(0, 2, n_samples),
        'young_loan': np.random.randint(0, 2, n_samples),
        'middle_aged_loan': np.random.randint(0, 2, n_samples),
        'contacted_before': np.random.randint(0, 2, n_samples),
        'default': np.random.randint(0, 2, n_samples),
        'euribor_bin_low': np.random.randint(0, 2, n_samples),
        'age_binned_quantile_3.0': np.random.randint(0, 2, n_samples),
        'campaign': np.random.randint(1, 10, n_samples),
        'cpi_above_75th': np.random.randint(0, 2, n_samples),
        'previous_bin_3': np.random.randint(0, 2, n_samples),
        'poutcome_failure': np.random.randint(0, 2, n_samples),
        'marital_married': np.random.randint(0, 2, n_samples),
        'euribor3m': np.random.uniform(0.5, 5.0, n_samples),
        'economic_pressure_index': np.random.uniform(-5, 5, n_samples),
        'previous_bin_2': np.random.randint(0, 2, n_samples),
        'cons.price.idx': np.random.uniform(90, 95, n_samples),
        'cpi_top_value_other': np.random.randint(0, 2, n_samples),
    })

    y = np.random.randint(0, 2, n_samples)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42)

    final_features = list(data.columns)

    parameters_model_selection = {
        "mlflow_experiment_name": "TestExperiment",
        "model_registry_name": "TestModelRegistry"
    }

    parameters_grid = {
        "hyperparameters": {
            "RandomForestClassifier": {"n_estimators": [10, 50], "max_depth": [3, 5]},
            "GradientBoostingClassifier": {"n_estimators": [10], "learning_rate": [0.1]},
            "LogisticRegression": {"C": [1.0]},
            "KNeighborsClassifier": {"n_neighbors": [3]},
            "XGBClassifier": {"n_estimators": [10]}
        }
    }

    return X_train, X_test, y_train, y_test, parameters_model_selection, parameters_grid, final_features




@pytest.fixture(autouse=True)
def mock_utils(monkeypatch):
    import pandas as pd
    import numpy as np

    def mock_ks_test(train_df, serving_df, columns):
        return {col: {"statistic": 0.1, "p_value": 0.5, "interpretation": "No drift"} for col in columns}

    def mock_psi_report(train_df, serving_df, report_path="fake_report.html"):
        return {"status": "Report saved (mock)", "path": report_path}

    #def mock_fisher_test_binary_feature(train_df, serving_df, column_name):
        return {"p_value": 0.5, "odds_ratio": 1.0, "contingency_table": pd.DataFrame()}

    def mock_fisher_test_target(y_train, y_test):
        return {"p_value": 0.5, "odds_ratio": 1.0, "contingency_table": pd.DataFrame()}

    def mock_pca_reconstruction_error(train_df, serving_df, n_components=0.95):
        return {
            "ks_p_value": 0.5,
            "ks_statistic": 0.1,
            "interpretation": "No drift",
            "train_errors": np.zeros(len(train_df)),
            "serving_errors": np.zeros(len(serving_df)),
        }

    def mock_create_drifted_df(train_df):
        drifted = train_df.copy()
        drifted["age"] = drifted["age"] + 1
        return drifted

    monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.ks_test", mock_ks_test)
    monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.psi_report", mock_psi_report)
    #monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.fisher_test_binary_feature", mock_fisher_test_binary_feature)
    monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.fisher_test_target", mock_fisher_test_target)
    monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.pca_reconstruction_error", mock_pca_reconstruction_error)
    monkeypatch.setattr("src.projeto_v1.pipelines.data_drift.utils.create_drifted_df", mock_create_drifted_df)



def test_run_all_drift_checks_returns_expected_structure(dummy_drift_data):
    X_train, X_test, y_train, y_test, _, _, feature_columns = dummy_drift_data

    import pandas as pd
    y_train = pd.Series(y_train)  # convert to Series
    y_test = pd.Series(y_test)    # convert to Series

    results = run_all_drift_checks(
        train_df=X_train,
        serving_df=X_test,
        y_train=y_train,
        y_serving=y_test,
        feature_columns=feature_columns,
        psi_report_path_serving="fake_path_serving.html",
        psi_report_path_drifted="fake_path_drifted.html"
    )

    assert isinstance(results, dict)
    assert "serving" in results and "drifted" in results
    assert "ks" in results["serving"]
    assert "psi" in results["serving"]
    assert "fisher_target" in results["serving"]
    assert "pca" in results["serving"]
    assert "ks" in results["drifted"]
    assert "psi" in results["drifted"]
    assert "pca" in results["drifted"]





def test_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "drift_detection_node"