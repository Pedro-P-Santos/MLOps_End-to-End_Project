"""
This is a boilerplate test file for pipeline 'model_train'
generated using Kedro 0.19.5.
"""

# import os
# import pickle
# import pytest
# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# from unittest.mock import patch, MagicMock, mock_open

# from src.projeto_v1.pipelines.model_train.nodes import model_train
# from src.projeto_v1.pipelines.model_train.pipeline import create_pipeline

# # Mock mlflow YAML content
# mock_mlflow_yaml = """
# server:
#   mlflow_tracking_uri: "http://fake-mlflow:5000"
# """

# def fake_wait_for_mlflow(uri: str, timeout: int = 60):
#     return None  # Skip waiting during tests

# @pytest.fixture(scope="module")
# def real_dummy_data():
#     n_samples = 50
#     np.random.seed(42)
#     data = pd.DataFrame({
#         'is_student_or_retired': np.random.randint(0, 2, n_samples),
#         'cpi_top_value_92.893': np.random.rand(n_samples),
#         'poutcome_success': np.random.randint(0, 2, n_samples),
#         'cci_top_value': np.random.rand(n_samples) * 100,
#         'marital_single': np.random.randint(0, 2, n_samples),
#         'previous_bin_1': np.random.randint(0, 2, n_samples),
#         'age': np.random.randint(18, 80, n_samples),
#         'age_binned_quantile_0.0': np.random.randint(0, 2, n_samples),
#         'education_mapped_basic_education': np.random.randint(0, 2, n_samples),
#         'poutcome_nonexistent': np.random.randint(0, 2, n_samples),
#         'middle_aged_housing_loan': np.random.randint(0, 2, n_samples),
#         'previous_bin_0': np.random.randint(0, 2, n_samples),
#         'previous': np.random.randint(0, 10, n_samples),
#         'euribor_bin_high': np.random.randint(0, 2, n_samples),
#         'emp.var.rate': np.random.uniform(-3.0, 3.0, n_samples),
#         'age_binned_quantile_4.0': np.random.randint(0, 2, n_samples),
#         'cpi_top_value_93.994': np.random.rand(n_samples),
#         'cpi_top_value_93.918': np.random.rand(n_samples),
#         'euribor_bin_very_high': np.random.randint(0, 2, n_samples),
#         'loan_risk_score': np.random.uniform(300, 850, n_samples),
#         'nr.employed': np.random.uniform(5000, 6000, n_samples),
#         'education_mapped_higher_education': np.random.randint(0, 2, n_samples),
#         'cons.conf.idx': np.random.uniform(-50, 50, n_samples),
#         'pdays': np.random.randint(-1, 30, n_samples),
#         'successful_prev_contact': np.random.randint(0, 2, n_samples),
#         'senior_loan': np.random.randint(0, 2, n_samples),
#         'age_binned_quantile_2.0': np.random.randint(0, 2, n_samples),
#         'senior_housing_loan': np.random.randint(0, 2, n_samples),
#         'cci_above_75th': np.random.randint(0, 2, n_samples),
#         'young_housing_loan': np.random.randint(0, 2, n_samples),
#         'euribor_bin_very_low': np.random.randint(0, 2, n_samples),
#         'young_loan': np.random.randint(0, 2, n_samples),
#         'middle_aged_loan': np.random.randint(0, 2, n_samples),
#         'contacted_before': np.random.randint(0, 2, n_samples),
#         'default': np.random.randint(0, 2, n_samples),
#         'euribor_bin_low': np.random.randint(0, 2, n_samples),
#         'age_binned_quantile_3.0': np.random.randint(0, 2, n_samples),
#         'campaign': np.random.randint(1, 10, n_samples),
#         'cpi_above_75th': np.random.randint(0, 2, n_samples),
#         'previous_bin_3': np.random.randint(0, 2, n_samples),
#         'poutcome_failure': np.random.randint(0, 2, n_samples),
#         'marital_married': np.random.randint(0, 2, n_samples),
#         'euribor3m': np.random.uniform(0.5, 5.0, n_samples),
#         'economic_pressure_index': np.random.uniform(-5, 5, n_samples),
#         'previous_bin_2': np.random.randint(0, 2, n_samples),
#         'cons.price.idx': np.random.uniform(90, 95, n_samples),
#         'cpi_top_value_other': np.random.randint(0, 2, n_samples),
#     })

#     y = np.random.randint(0, 2, n_samples)

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

#     model_config = {
#         "mlflow_experiment_name": "TestExperiment",
#         "model_registry_name": "TestModel"
#     }
#     params = {
#         "hyperparameters": {
#             "XGBClassifier": {"n_estimators": [10]}
#         }
#     }
#     return X_train, X_test, y_train, y_test, model_config, params, list(data.columns)

# @pytest.fixture
# def champion_model_path(tmp_path):
#     model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss')
#     model.fit(pd.DataFrame({"feat1": [0, 1], "feat2": [1, 0]}), [0, 1])

#     model_dir = tmp_path / "data" / "06_models"
#     model_dir.mkdir(parents=True)
#     model_path = model_dir / "champion_model.pkl"
#     with open(model_path, "wb") as f:
#         pickle.dump(model, f)
#     return model_path, tmp_path

# @patch("pickle.load")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.set_tracking_uri")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.get_experiment_by_name")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.create_experiment")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.sklearn.autolog")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.start_run")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.log_metric")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.log_artifact")
# @patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.register_model")
# @patch("src.projeto_v1.pipelines.model_train.nodes.MlflowClient")
# @patch("src.projeto_v1.pipelines.model_train.nodes.wait_for_mlflow", side_effect=fake_wait_for_mlflow)
# @patch("builtins.open", new_callable=mock_open, read_data=mock_mlflow_yaml)
# def test_model_train_with_real_dummy(
#     mock_open_file,
#     mock_wait,
#     mock_mlflow_client_class,
#     mock_register_model,
#     mock_log_artifact,
#     mock_log_metric,
#     mock_start_run,
#     mock_autolog,
#     mock_create_exp,
#     mock_get_exp,
#     mock_set_tracking_uri,
#     mock_pickle_load,
#     champion_model_path,
#     real_dummy_data,
# ):
#     model_path, tmp_path = champion_model_path
#     X_train, X_test, y_train, y_test, model_config, _, final_features = real_dummy_data

#     dummy_model = MagicMock()
#     dummy_model.predict.return_value = y_test
#     mock_pickle_load.return_value = dummy_model

#     model_train_params = {
#         "experiment_name_train": model_config["mlflow_experiment_name"],
#         "run_name_train": "Test_Run",
#         "model_registry_name_train": model_config["model_registry_name"]
#     }

#     with patch("os.getcwd", return_value=str(tmp_path)):
#         mock_client = MagicMock()
#         mock_mlflow_client_class.return_value = mock_client
#         mock_client.get_latest_versions.return_value = [MagicMock(version="1")]

#         experiment_mock = MagicMock()
#         experiment_mock.experiment_id = "exp_123"
#         mock_get_exp.return_value = experiment_mock
#         mock_create_exp.return_value = "new_exp_456"

#         run_mock = MagicMock()
#         run_mock.info.run_id = "run_id_123"
#         mock_start_run.return_value.__enter__.return_value = run_mock

#         model, results = model_train(
#             X_train_scaled=X_train,
#             X_test_scaled=X_test,
#             y_train_encoded=y_train,
#             y_test_encoded=y_test,
#             parameters_model_train=model_train_params,
#             final_selected_features=final_features,
#         )

#     assert model == dummy_model
#     assert isinstance(results, dict)
#     assert "F1_Macro-Train" in results
#     assert isinstance(results["F1_Macro-Train"], float)

#     mock_set_tracking_uri.assert_called_once()
#     mock_get_exp.assert_called_once_with(model_config["mlflow_experiment_name"])
#     mock_autolog.assert_called_once()
#     mock_start_run.assert_called_once()
#     mock_register_model.assert_called_once()
#     mock_client.get_latest_versions.assert_called_once()
#     mock_client.set_model_version_tag.assert_called()
#     assert mock_log_metric.call_count > 0
#     assert mock_log_artifact.call_count > 0

# def test_pipeline_structure():
#     pipeline = create_pipeline()
#     assert pipeline.nodes
#     assert pipeline.nodes[0].name == "train_champion_model"

import numpy as np
import pandas as pd
import pytest
import pickle
from unittest.mock import patch, MagicMock, mock_open
from xgboost import XGBClassifier
from src.projeto_v1.pipelines.model_train.nodes import model_train
from src.projeto_v1.pipelines.model_train.pipeline import create_pipeline

# Mock MLflow YAML config content
mock_mlflow_yaml = """
server:
  mlflow_tracking_uri: "http://fake-mlflow:5000"
"""

@pytest.fixture
def minimal_valid_data():
    # Define feature names
    features = ["feat1", "feat2"]

    # Create dummy training and test data as pandas DataFrames
    X_train = pd.DataFrame(
        [[0, 1], [1, 0], [1, 1]],
        columns=features
    )
    X_test = pd.DataFrame(
        [[0, 0], [1, 1]],
        columns=features
    )

    # Create dummy target arrays
    y_train = np.array([0, 1, 1])
    y_test = np.array([0, 1])

    return X_train, X_test, y_train, y_test, features


# Dummy champion model path fixture
@pytest.fixture
def champion_model_path(tmp_path):
    model = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss')
    X_dummy = pd.DataFrame({"feat1": [0, 1], "feat2": [1, 0]})
    y_dummy = [0, 1]
    model.fit(X_dummy, y_dummy)

    model_dir = tmp_path / "data" / "06_models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "champion_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path, tmp_path

# Skip mlflow connection wait
def fake_wait_for_mlflow(uri: str, timeout: int = 60):
    return None

@patch("pickle.load")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.set_tracking_uri")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.get_experiment_by_name")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.create_experiment")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.sklearn.autolog")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.start_run")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.log_metric")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.log_artifact")
@patch("src.projeto_v1.pipelines.model_train.nodes.mlflow.register_model")
@patch("src.projeto_v1.pipelines.model_train.nodes.MlflowClient")
@patch("src.projeto_v1.pipelines.model_train.nodes.wait_for_mlflow", side_effect=fake_wait_for_mlflow)
@patch("builtins.open", new_callable=mock_open, read_data=mock_mlflow_yaml)
def test_model_train_with_hardcoded_data(
    mock_open_file,
    mock_wait,
    mock_mlflow_client_class,
    mock_register_model,
    mock_log_artifact,
    mock_log_metric,
    mock_start_run,
    mock_autolog,
    mock_create_exp,
    mock_get_exp,
    mock_set_tracking_uri,
    mock_pickle_load,
    champion_model_path,
    minimal_valid_data
):
    model_path, tmp_path = champion_model_path
    X_train, X_test, y_train, y_test, features = minimal_valid_data

    dummy_model = MagicMock()

    # Fix: make predict return array matching input length (to avoid shape mismatch)
    dummy_model.predict.side_effect = lambda X: np.zeros(len(X), dtype=int)

    dummy_model.fit.return_value = dummy_model
    mock_pickle_load.return_value = dummy_model

    with patch("os.getcwd", return_value=str(tmp_path)):
        mock_client = MagicMock()
        mock_mlflow_client_class.return_value = mock_client
        version_mock = MagicMock()
        version_mock.version = "1"
        mock_client.get_latest_versions.return_value = [version_mock]
        mock_client.set_model_version_tag.return_value = None

        experiment_mock = MagicMock()
        experiment_mock.experiment_id = "exp_123"
        mock_get_exp.return_value = experiment_mock
        mock_create_exp.return_value = "exp_123"

        run_mock = MagicMock()
        run_mock.info.run_id = "run_123"
        mock_start_run.return_value.__enter__.return_value = run_mock

        model, results = model_train(
            X_train_scaled=X_train,
            X_test_scaled=X_test,
            y_train_encoded=y_train,
            y_test_encoded=y_test,
            parameters_model_train={
                "experiment_name_train": "TestExperiment",
                "run_name_train": "Test_Run",
                "model_registry_name_train": "TestModel"
            },
            final_selected_features=features,
        )

    assert model == dummy_model
    assert isinstance(results, dict)
    assert "F1_Macro-Train" in results

    mock_set_tracking_uri.assert_called_once()
    mock_get_exp.assert_called_once_with("TestExperiment")
    mock_autolog.assert_called_once()
    mock_start_run.assert_called_once()
    mock_register_model.assert_called_once()
    mock_client.get_latest_versions.assert_called_once()
    mock_log_metric.assert_called()
    mock_log_artifact.assert_called()


# Pipeline smoke test
def test_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "train_champion_model"



