import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
import mlflow
from mlflow.tracking import MlflowClient
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
from pathlib import Path
import time
import requests

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=Warning)


def wait_for_mlflow(uri: str, timeout: int = 60):
    logger.info(f"Waiting for MLflow server at {uri}...")
    start = time.time()
    while True:
        try:
            response = requests.get(uri)
            if response.status_code == 200:
                logger.info("MLflow is available.")
                break
        except Exception:
            logger.info("MLflow not available yet, retrying...")
        if time.time() - start > timeout:
            raise TimeoutError(f"MLflow not available after {timeout} seconds.")
        time.sleep(3)


def model_train(
        X_train_scaled: pd.DataFrame,
        X_test_scaled: pd.DataFrame,
        y_train_encoded: pd.DataFrame,
        y_test_encoded: pd.DataFrame,
        parameters_model_train: Dict[str, Any],
        final_selected_features: list,
):
    # ---------------- SETUP EXPERIMENTO ----------------
    env = os.getenv("KEDRO_ENV", "local")
    with open(f"conf/{env}/mlflow.yml", "r") as f:
        mlflow_config = yaml.load(f, Loader=yaml.SafeLoader)

    mlflow.set_tracking_uri(mlflow_config["server"]["mlflow_tracking_uri"])
    logger.info(f"Tracking URI set to: {mlflow.get_tracking_uri()}")
    wait_for_mlflow(mlflow.get_tracking_uri())

    experiment_name = parameters_model_train["experiment_name_train"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    run_name = parameters_model_train["run_name_train"]

    # ---------------- AUTOLOG ----------------
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    logger.info(f"Starting MLFlow run {run_name}")
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

        # ---------------- LOAD CHAMPION MODEL ----------------
        try:
            with open(os.path.join(os.getcwd(), "data", "06_models", "champion_model.pkl"), "rb") as f:
                champ_model = pickle.load(f)
        except FileNotFoundError:
            logger.error("Champion model not fetched correctly.")
            raise

        logger.info("Starting Training with Selected Features")
        X_train_scaled = X_train_scaled[final_selected_features]
        X_test_scaled = X_test_scaled[final_selected_features]

        y_train_encoded = np.ravel(y_train_encoded)
        y_test_encoded = np.ravel(y_test_encoded)

        # ---------------- TRAINING ----------------
        model = champ_model.fit(X_train_scaled, y_train_encoded)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # ---------------- METRICS ----------------
        f1_macro_train = f1_score(y_train_encoded, y_train_pred, average="macro")
        f1_macro_test = f1_score(y_test_encoded, y_test_pred, average="macro")
        f1_diff = f1_macro_train - f1_macro_test
        mlflow.log_metric("F1_Macro_Train", f1_macro_train)
        mlflow.log_metric("F1_Macro_Test", f1_macro_test)
        mlflow.log_metric("F1_DIFF", f1_diff)

        recall_train = recall_score(y_train_encoded, y_train_pred, average="macro")
        recall_test = recall_score(y_test_encoded, y_test_pred, average="macro")
        mlflow.log_metric("Recall-train", recall_train)
        mlflow.log_metric("Recall-test", recall_test)

        precision_train = precision_score(y_train_encoded, y_train_pred, average="macro")
        precision_test = precision_score(y_test_encoded, y_test_pred, average="macro")
        mlflow.log_metric("Precision-Train", precision_train)
        mlflow.log_metric("Precision-Test", precision_test)

        results = {
            "classifier": champ_model.__class__.__name__,
            "F1_Macro-Train": f1_macro_train,
            "F1_Macro-Test": f1_macro_test,
            "F1_DIFF": f1_diff,
            "Recall-Train": recall_train,
            "Recall-Test": recall_test,
            "Precision-Train": precision_train,
            "Precision-Test": precision_test,
        }

        # ---------------- REGISTO EM MLFLOW ----------------
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_model_name = parameters_model_train["model_registry_name_train"]
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        logger.info(f"Model registered in MLflow as '{registered_model_name}' from run {run_id}")

        client = MlflowClient()
        model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version

        tags = {
            "author": "Pedro",
            "model_type": "Champ_RF",
            "training_run": run_id,
            "F1_Macro-Train": round(f1_macro_train, 3),
            "stage": "Champion_Pre-Deployment"
        }
        for key, value in tags.items():
            client.set_model_version_tag(
                name=registered_model_name,
                version=model_version,
                key=key,
                value=value
            )

        # ---------------- SHAP EXPLAINABILITY (100 amostras) ----------------
        logger.info("Generating SHAP explainability plots (100 samples)...")
        Path("data/08_reporting").mkdir(parents=True, exist_ok=True)

        X_sample = X_train_scaled.head(100)
        if isinstance(X_sample, np.ndarray):
            X_sample = pd.DataFrame(X_sample, columns=final_selected_features)
        else:
            X_sample = X_sample[final_selected_features]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        shap.initjs()
        try:
            plt.figure()
            shap.summary_plot(shap_values.values[:, :, 1], X_sample, feature_names=X_sample.columns, show=False)
            path = "data/08_reporting/shap_summary.png"
            plt.savefig(path, bbox_inches='tight')
            mlflow.log_artifact(path)
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")

        # ---------------- FEATURE IMPORTANCE ----------------
        importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
        importances.sort_values().plot(kind="barh")
        plt.title("Tree-Based F.I")
        plt.tight_layout()
        fi_path = "data/08_reporting/feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path)

    return model, results
