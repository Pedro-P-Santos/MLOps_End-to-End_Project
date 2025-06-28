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
from sklearn.ensemble import RandomForestClassifier
import mlflow
from src.projeto_v1.pipelines.model_train.nodes import model_train
from src.projeto_v1.pipelines.model_train.pipeline import create_pipeline

def test_model_train_with_real_mlflow():
    # Set tracking URI to your local MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Optional: clean up if test experiment already exists
    experiment_name = "test_experiment"
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment:
        experiment_id = existing_experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Dummy champion model (used by model_train if needed)
    os.makedirs("data/06_models", exist_ok=True)
    dummy_model = RandomForestClassifier(n_estimators=5, random_state=42)
    dummy_model.fit(np.array([[0, 0], [1, 1]]), [0, 1])
    with open("data/06_models/champion_model.pkl", "wb") as f:
        pickle.dump(dummy_model, f)

    # Prepare input data
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [0.1, 0.2, 0.3, 0.4]})
    X_test = pd.DataFrame({"feat1": [2, 3], "feat2": [0.15, 0.35]})
    y_train = pd.DataFrame({"target": [0, 1, 0, 1]})
    y_test = pd.DataFrame({"target": [1, 0]})
    selected_features = ["feat1", "feat2"]

    # Parameters with real experiment and run
    params = {
        "experiment_name_train": experiment_name,
        "run_name_train": "test_run_with_real_server",
        "model_registry_name_train": "test_model_registry"
    }

    # Run training
    model, results = model_train(
        X_train_scaled=X_train,
        X_test_scaled=X_test,
        y_train_encoded=y_train,
        y_test_encoded=y_test,
        parameters_model_train=params,
        final_selected_features=selected_features
    )

    # Assertions
    assert model is not None
    assert hasattr(model, "predict")
    assert isinstance(results, dict)
    assert results.get("classifier") == model.__class__.__name__

    for metric in [
        "F1_Macro-Train", "F1_Macro-Test", "Recall-Train", "Recall-Test",
        "Precision-Train", "Precision-Test", "F1_DIFF"
    ]:
        assert metric in results
        assert isinstance(results[metric], float)

    # Pipeline structure
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "train_champion_model"

    print("✅ Test passed: model_train works with real MLflow server.")