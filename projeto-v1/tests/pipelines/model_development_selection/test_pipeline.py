"""
This is a boilerplate test file for pipeline 'model_development_selection'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.projeto_v1.pipelines.model_development_selection.nodes import model_selection 
from src.projeto_v1.pipelines.model_development_selection.pipeline import create_pipeline
def test_model_selection():
    # Load configuration from YAML
    config_path = Path("conf/base/parameters.yml")
    with config_path.open("r") as file:
        parameters = yaml.safe_load(file)

    parameters_model_selection = parameters["parameters_model_selection"]
    parameters_grid = parameters["parameters_grid"]

    # Dummy training data
    X_train_preprocessed = pd.DataFrame({
        "age": [25, 32, 47, 51, 38],
        "balance": [100.5, 200.1, 150.2, 120.8, 180.3],
        "duration": [300, 400, 500, 600, 700],
        "job_admin": [1, 0, 1, 0, 1],  # One-hot encoded categorical
        "job_unknown": [0, 1, 0, 1, 0]
    })

    X_test_preprocessed = pd.DataFrame({
        "age": [29, 41],
        "balance": [110.3, 199.4],
        "duration": [350, 450],
        "job_admin": [0, 1],
        "job_unknown": [1, 0]
    })

    y_train_encoded = pd.DataFrame({"y": [1, 0, 1, 0, 1]})
    y_test_encoded = pd.DataFrame({"y": [0, 1]})

    final_selected_features = ["age", "balance", "duration", "job_admin", "job_unknown"]

    # Run model_selection
    champion_model, X_train_scaled, X_test_scaled = model_selection(
        X_train_preprocessed=X_train_preprocessed,
        X_test_preprocessed=X_test_preprocessed,
        y_train=y_train_encoded,
        y_test=y_test_encoded,
        parameters_model_selection=parameters_model_selection,
        parameters_grid=parameters_grid,
        final_selected_features=final_selected_features,
    )

    # Basic assertions
    assert champion_model is not None
    assert isinstance(X_train_scaled, pd.DataFrame)
    assert isinstance(X_test_scaled, pd.DataFrame)
    assert X_train_scaled.shape == X_train_preprocessed.shape
    assert X_test_scaled.shape == X_test_preprocessed.shape
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "model_selection_node"