"""
This is a boilerplate test file for pipeline 'model_train'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import os
import pickle
import pandas as pd
import numpy as np
from unittest import mock
from sklearn.ensemble import RandomForestClassifier

from src.model_train.nodes import model_train
from src.model_train.pipeline import create_pipeline

def test_model_train_mock_mlflow():
    # Setup environment variable for config loading
    os.environ["KEDRO_ENV"] = "local"

    # Create and save a dummy "champion" model for loading in model_train
    dummy_model = RandomForestClassifier(n_estimators=5, random_state=42)
    dummy_model.fit(np.array([[0, 0], [1, 1]]), [0, 1])  # minimal training
    os.makedirs("data/06_models", exist_ok=True)
    with open("data/06_models/champion_model.pkl", "wb") as f:
        pickle.dump(dummy_model, f)

    # Create dummy mlflow.yml file with tracking URI (won't be used but needed to pass yaml load)
    os.makedirs("conf/local", exist_ok=True)
    with open("conf/local/mlflow.yml", "w") as f:
        f.write("server:\n  mlflow_tracking_uri: http://fake-mlflow-server\n")

    # Prepare dummy data inputs
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [0.1, 0.2, 0.3, 0.4]})
    X_test = pd.DataFrame({"feat1": [2, 3], "feat2": [0.15, 0.35]})
    y_train = pd.DataFrame({"target": [0, 1, 0, 1]})
    y_test = pd.DataFrame({"target": [1, 0]})

    selected_features = ["feat1", "feat2"]
    params = {
        "experiment_name_train": "test_experiment",
        "run_name_train": "test_run",
        "model_registry_name_train": "test_model_registry"
    }

    # Patch all MLflow and requests calls to mocks
    with mock.patch("mlflow.set_tracking_uri"), \
         mock.patch("mlflow.get_tracking_uri", return_value="http://fake-mlflow-server"), \
         mock.patch("src.model_train.nodes.wait_for_mlflow"), \
         mock.patch("mlflow.get_experiment_by_name", return_value=None), \
         mock.patch("mlflow.create_experiment", return_value=1), \
         mock.patch("mlflow.sklearn.autolog"), \
         mock.patch("mlflow.start_run") as mock_start_run, \
         mock.patch("mlflow.log_metric"), \
         mock.patch("mlflow.log_artifact"), \
         mock.patch("mlflow.register_model"), \
         mock.patch("mlflow.active_run") as mock_active_run, \
         mock.patch("mlflow.tracking.MlflowClient") as mock_mlflow_client, \
         mock.patch("matplotlib.pyplot.savefig"), \
         mock.patch("matplotlib.pyplot.close"):

        # Mock active run info (needed for run_id)
        mock_run_info = mock.Mock()
        mock_run_info.info.run_id = "fake_run_id"
        mock_active_run.return_value = mock_run_info

        # Mock MLflowClient and its methods
        mock_client_instance = mock.Mock()
        mock_client_instance.get_latest_versions.return_value = [mock.Mock(version="1")]
        mock_mlflow_client.return_value = mock_client_instance

        # Mock context manager behavior for mlflow.start_run
        mock_start_run.return_value.__enter__.return_value = None
        mock_start_run.return_value.__exit__.return_value = None

        # Call the function under test
        model, results = model_train(
            X_train_scaled=X_train,
            X_test_scaled=X_test,
            y_train_encoded=y_train,
            y_test_encoded=y_test,
            parameters_model_train=params,
            final_selected_features=selected_features
        )

    # Assertions to verify expected outputs
    assert model is not None
    assert hasattr(model, "predict")
    assert isinstance(results, dict)
    assert results.get("classifier") == model.__class__.__name__
    for metric in ["F1_Macro-Train", "F1_Macro-Test", "Recall-Train", "Recall-Test", "Precision-Train", "Precision-Test", "F1_DIFF"]:
        assert metric in results
        assert isinstance(results[metric], float)

    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "train_champion_model"

    print("Test passed: model_train works with mocked MLflow calls.")
