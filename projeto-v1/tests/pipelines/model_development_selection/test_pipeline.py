"""
This is a boilerplate test file for pipeline 'model_development_selection'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.projeto_v1.pipelines.model_development_selection.nodes import model_selection
from src.projeto_v1.pipelines.model_development_selection.pipeline import create_pipeline



@pytest.fixture(scope="module")
def real_dummy_data():
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


@patch("src.projeto_v1.pipelines.model_development_selection.nodes.mlflow")
@patch("src.projeto_v1.pipelines.model_development_selection.nodes.optuna.create_study")
def test_model_selection_with_realistic_data(mock_create_study, mock_mlflow, real_dummy_data):
    X_train, X_test, y_train, y_test, params_model_sel, params_grid, final_features = real_dummy_data

    mock_study = MagicMock()
    mock_study.best_params = {"n_estimators": 10, "max_depth": 3}
    mock_study.best_value = 0.8
    mock_study.optimize.return_value = None
    mock_create_study.return_value = mock_study

    # Setup mlflow mock methods
    mock_mlflow.get_experiment_by_name.return_value = None
    mock_mlflow.create_experiment.return_value = "123"
    mock_mlflow.start_run.return_value.__enter__.return_value = None
    mock_mlflow.start_run.return_value.__exit__.return_value = None
    mock_mlflow.sklearn.autolog.return_value = None
    mock_mlflow.log_metric.return_value = None
    mock_mlflow.log_artifact.return_value = None
    mock_mlflow.sklearn.log_model.return_value = None
    mock_mlflow.set_tag.return_value = None

    champion_model, X_train_scaled, X_test_scaled = model_selection(
        X_train,
        X_test,
        y_train,
        y_test,
        params_model_sel,
        params_grid,
        final_features
    )

    # Assertions
    assert hasattr(champion_model, "predict"), "Champion model should have predict method"
    assert isinstance(X_train_scaled, pd.DataFrame), "X_train_scaled should be a DataFrame"
    assert isinstance(X_test_scaled, pd.DataFrame), "X_test_scaled should be a DataFrame"
    assert X_train_scaled.shape[1] == len(final_features)
    assert X_test_scaled.shape[1] == len(final_features)

    # Check scaled data mean ~ 0
    assert abs(X_train_scaled.mean().mean()) < 1e-6

    # Check mocks were called
    mock_study.optimize.assert_called()
    mock_mlflow.log_metric.assert_called()
    mock_mlflow.sklearn.log_model.assert_called()

pipeline = create_pipeline()
assert pipeline.nodes
assert pipeline.nodes[0].name == "model_selection_node"
