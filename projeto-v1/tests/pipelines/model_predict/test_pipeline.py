"""
This is a boilerplate test file for pipeline 'model_predict'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from unittest.mock import MagicMock
from src.projeto_v1.pipelines.model_predict.nodes import model_predict  
from src.projeto_v1.pipelines.model_predict.pipeline import create_pipeline

import pytest
import numpy as np
import pandas as pd

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


def test_model_predict_with_real_dummy(real_dummy_data):
    # Unpack the fixture
    X_train, X_test, y_train, y_test, _, _, final_features = real_dummy_data

    # Create a dummy model mock
    dummy_model = MagicMock()
    # Mock predict to return zeros for each row in X_test
    dummy_model.predict.return_value = [0] * len(X_test)

    # Call the node function
    predictions, summary = model_predict(X_test, dummy_model, final_features)

    # Check outputs
    assert isinstance(predictions, pd.DataFrame), "Predictions should be a DataFrame"
    assert "y_pred" in predictions.columns, "'y_pred' column must be present"
    assert len(predictions) == len(X_test), "Predictions length mismatch"
    assert all(pred == 0 for pred in predictions["y_pred"]), "All predictions should be zero"

    assert isinstance(summary, dict), "Summary should be a dictionary"
    assert "y_pred" in summary, "Summary should contain 'y_pred' stats"

    # Check that predict was called once
    dummy_model.predict.assert_called_once()
    
    # Grab the actual argument passed to predict
    called_arg = dummy_model.predict.call_args[0][0]
    
    # Verify it's a DataFrame with the expected columns and number of rows
    assert isinstance(called_arg, pd.DataFrame)
    assert list(called_arg.columns) == final_features
    assert called_arg.shape[0] == X_test.shape[0]


pipeline = create_pipeline()
assert pipeline.nodes
assert pipeline.nodes[0].name == "tests"
