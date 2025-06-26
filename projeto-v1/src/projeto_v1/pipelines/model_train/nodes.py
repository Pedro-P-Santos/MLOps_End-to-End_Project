# """
# This is a boilerplate pipeline 'model_train'
# generated using Kedro 0.19.5
# """
# import pandas as pd
# import logging
# from typing import Dict, Tuple, Any
# import numpy as np
# import pickle
# import yaml
# import os
# import warnings
# import mlflow
# from mlflow.tracking import MlflowClient
# import shap
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, recall_score, precision_score
# from pathlib import Path

# logger = logging.getLogger(__name__)

# def model_train(
#         X_train_scaled: pd.DataFrame,
#         X_test_scaled: pd.DataFrame,
#         y_train_encoded: pd.DataFrame,
#         y_test_encoded: pd.DataFrame,
#         parameters_model_train: Dict[str, Any],
#         final_selected_features: list,
# ):
# ### Setup new Experiment for Relevant Tracking

# #### URI mlflow
#     with open("conf/local/mlflow.yml", "r") as f:
#         mlflow_config = yaml.load(f, Loader=yaml.SafeLoader)
#     mlflow.set_tracking_uri(mlflow_config["server"]["mlflow_tracking_uri"])

# #### Experiment and Run Names
#     experiment_name = parameters_model_train["experiment_name_train"]
#     experiment = mlflow.get_experiment_by_name(experiment_name)
#     if experiment is None:
#         experiment_id = mlflow.create_experiment(experiment_name)
#     else:
#         experiment_id = experiment.experiment_id

#     run_name = parameters_model_train["run_name_train"]

# #### Enabling Autologging
#     mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

# #### RUN
#     logger.info(f"Starting MLFlow run {run_name}")
#     with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

# #### FETCHING CHAMP MODEL
#         try:
#             with open(os.path.join(os.getcwd(), "data", "06_models", "champion_model.pkl"), "rb") as f:
#                 champ_model = pickle.load(f)
#         except FileNotFoundError:
#             logger.error("Champion model not fetched correctly.")
#             raise

#         results = {}
#         logger.info("Starting Training, with Selected features")
#         X_train_scaled = X_train_scaled[final_selected_features]
#         X_test_scaled = X_test_scaled[final_selected_features]
        
#         assert isinstance(y_train_encoded, pd.DataFrame)
#         assert isinstance(y_test_encoded, pd.DataFrame)

#         y_train_encoded = np.ravel(y_train_encoded)
#         y_test_encoded = np.ravel(y_test_encoded)

#         model = champ_model.fit(X_train_scaled, y_train_encoded)
#         #### PREDS
#         y_train_pred = model.predict(X_train_scaled)
#         y_test_pred = model.predict(X_test_scaled)

#         #### RELEVANT METRICS
#         f1_macro_train = f1_score(y_train_encoded, y_train_pred, average="macro")
#         f1_macro_test = f1_score(y_test_encoded, y_test_pred, average="macro")
#         mlflow.log_metric("F1_Macro_Train", f1_macro_train)
#         mlflow.log_metric("F1_Macro_Test", f1_macro_test)
#         f1_diff = f1_macro_train - f1_macro_test

#         mlflow.log_metric("F1_DIFF", f1_diff)

#         recall_train = recall_score(y_train_encoded, y_train_pred, average="macro")
#         recall_test = recall_score(y_test_encoded, y_test_pred, average="macro")
#         mlflow.log_metric("Recall-train", recall_train)
#         mlflow.log_metric("Recall-test",recall_test)

#         precision_train = precision_score(y_train_encoded, y_train_pred)
#         precision_test = precision_score(y_test_encoded, y_test_pred)
#         mlflow.log_metric("Precision-Train", precision_train)
#         mlflow.log_metric("Precision-Test", precision_test)

#         ## saving in results dict
#         results["classifier"] = champ_model.__class__.__name__
#         results["F1_Macro-Train"] = f1_macro_train
#         results["F1_Macro-Test"] = f1_macro_test    
#         results["F1_DIFF"] = f1_diff
#         results["Recall-Train"] = recall_train
#         results["Recall-Test"] = recall_test
#         results["Precision-Train"] = precision_train
#         results["Precision-Test"] = precision_test

#     ### ENSURING REGITRY 
#         run_id = mlflow.active_run().info.run_id
#         model_uri = f"runs:/{run_id}/model"
#         registered_model_name = parameters_model_train["model_registry_name_train"]

#         mlflow.register_model(model_uri=model_uri, name=registered_model_name)
#         logger.info(f"Model registered in MLflow as '{registered_model_name}' from run {run_id}")
#         client = MlflowClient()
#         model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version

#         ## Tags for Registry
#         tags = {
#             "author": "Pedro",
#             "model_type": "Champ_RF",
#             "training_run": run_id,
#             "F1_Macro-Train": round(f1_macro_train,3),
#             "stage": "Champion_Pre-Deployment"
#         }
#         for key, value in tags.items():
#             client.set_model_version_tag(
#                 name=registered_model_name,
#                 version=model_version,
#                 key=key,
#                 value=value
#             )

#         # SHAP - EXPLAINABILITY
#         explainer_shap = shap.TreeExplainer(model)
#         shap_values = explainer_shap(X_train_scaled)

#         ## Visualize in 08_reporting
#         shap.initjs()
#         class_1 = shap_values[:,:,1]
#         class_0 = shap_values[:,:,0]
#         shap.summary_plot(class_1, X_train_scaled, feature_names = X_train_scaled.columns, show=False)
#         plt.savefig("data/08_reporting/shap_summary_class_1.png")
#         plt.close()
#         shap.summary_plot(class_0, X_train_scaled, feature_names = X_train_scaled.columns, show=False)
#         plt.savefig("data/08_reporting/shap_summary_class_0.png")
#         plt.close()

#         ## Feat. Importance
#         importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
#         importances.sort_values().plot(kind="barh")
#         plt.title("Tree-Based F.I")
#         plt.tight_layout()
#         plt.savefig("data/08_reporting/feature_importance.png")
#         plt.close()

#         mlflow.log_artifact("data/08_reporting/shap_summary_class_1.png")
#         mlflow.log_artifact("data/08_reporting/shap_summary_class_0.png")
#         mlflow.log_artifact("data/08_reporting/feature_importance.png")

#     return model, results


# """
# This is a boilerplate pipeline 'model_train'
# generated using Kedro 0.19.5
# """
# import pandas as pd
# import logging
# from typing import Dict, Tuple, Any
# import numpy as np
# import pickle
# import yaml
# import os
# import warnings
# import mlflow
# from mlflow.tracking import MlflowClient
# import shap
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, recall_score, precision_score
# from pathlib import Path

# logger = logging.getLogger(__name__)

# def model_train(
#         X_train_scaled: pd.DataFrame,
#         X_test_scaled: pd.DataFrame,
#         y_train_encoded: pd.DataFrame,
#         y_test_encoded: pd.DataFrame,
#         parameters_model_train: Dict[str, Any],
#         final_selected_features: list,
# ):

#     # Setup MLflow URI
#     with open("conf/local/mlflow.yml", "r") as f:
#         mlflow_config = yaml.load(f, Loader=yaml.SafeLoader)
#     mlflow.set_tracking_uri(mlflow_config["server"]["mlflow_tracking_uri"])

#     # Experiment and Run Names
#     experiment_name = parameters_model_train["experiment_name_train"]
#     experiment = mlflow.get_experiment_by_name(experiment_name)
#     if experiment is None:
#         experiment_id = mlflow.create_experiment(experiment_name)
#     else:
#         experiment_id = experiment.experiment_id

#     run_name = parameters_model_train["run_name_train"]

#     # Enable autologging
#     mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

#     logger.info(f"Starting MLFlow run {run_name}")
#     with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

#         # Load champion model
#         try:
#             with open(os.path.join(os.getcwd(), "data", "06_models", "champion_model.pkl"), "rb") as f:
#                 champ_model = pickle.load(f)
#         except FileNotFoundError:
#             logger.error("Champion model not fetched correctly.")
#             raise

#         results = {}
#         logger.info("Starting Training, with Selected features")
#         X_train_scaled = X_train_scaled[final_selected_features]
#         X_test_scaled = X_test_scaled[final_selected_features]

#         assert isinstance(y_train_encoded, pd.DataFrame)
#         assert isinstance(y_test_encoded, pd.DataFrame)

#         y_train_encoded = np.ravel(y_train_encoded)
#         y_test_encoded = np.ravel(y_test_encoded)

#         model = champ_model.fit(X_train_scaled, y_train_encoded)

#         # Predictions
#         y_train_pred = model.predict(X_train_scaled)
#         y_test_pred = model.predict(X_test_scaled)

#         # Metrics
#         f1_macro_train = f1_score(y_train_encoded, y_train_pred, average="macro")
#         f1_macro_test = f1_score(y_test_encoded, y_test_pred, average="macro")
#         f1_diff = f1_macro_train - f1_macro_test

#         mlflow.log_metric("F1_Macro_Train", f1_macro_train)
#         mlflow.log_metric("F1_Macro_Test", f1_macro_test)
#         mlflow.log_metric("F1_DIFF", f1_diff)

#         recall_train = recall_score(y_train_encoded, y_train_pred, average="macro")
#         recall_test = recall_score(y_test_encoded, y_test_pred, average="macro")
#         mlflow.log_metric("Recall-train", recall_train)
#         mlflow.log_metric("Recall-test", recall_test)

#         precision_train = precision_score(y_train_encoded, y_train_pred)
#         precision_test = precision_score(y_test_encoded, y_test_pred)
#         mlflow.log_metric("Precision-Train", precision_train)
#         mlflow.log_metric("Precision-Test", precision_test)

#         # Save results
#         results["classifier"] = champ_model.__class__.__name__
#         results["F1_Macro-Train"] = f1_macro_train
#         results["F1_Macro-Test"] = f1_macro_test
#         results["F1_DIFF"] = f1_diff
#         results["Recall-Train"] = recall_train
#         results["Recall-Test"] = recall_test
#         results["Precision-Train"] = precision_train
#         results["Precision-Test"] = precision_test

#         # Register model in MLflow
#         run_id = mlflow.active_run().info.run_id
#         model_uri = f"runs:/{run_id}/model"
#         registered_model_name = parameters_model_train["model_registry_name_train"]

#         mlflow.register_model(model_uri=model_uri, name=registered_model_name)
#         logger.info(f"Model registered in MLflow as '{registered_model_name}' from run {run_id}")
#         client = MlflowClient()
#         model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version

#         # Tags for registry
#         tags = {
#             "author": "Pedro",
#             "model_type": "Champ_RF",
#             "training_run": run_id,
#             "F1_Macro-Train": round(f1_macro_train, 3),
#             "stage": "Champion_Pre-Deployment"
#         }
#         for key, value in tags.items():
#             client.set_model_version_tag(
#                 name=registered_model_name,
#                 version=model_version,
#                 key=key,
#                 value=value
#             )

#         # SHAP - Explainability
#         explainer_shap = shap.TreeExplainer(model)
#         shap_values = explainer_shap(X_train_scaled)

#         # Visualize SHAP results
#         shap.initjs()

#         if isinstance(shap_values, list) and len(shap_values) == 2:
#             # Binary classification: class 1 and class 0
#             shap.summary_plot(shap_values[1], X_train_scaled, feature_names=X_train_scaled.columns, show=False)
#             plt.savefig("data/08_reporting/shap_summary_class_1.png")
#             plt.close()

#             shap.summary_plot(shap_values[0], X_train_scaled, feature_names=X_train_scaled.columns, show=False)
#             plt.savefig("data/08_reporting/shap_summary_class_0.png")
#             plt.close()

#             mlflow.log_artifact("data/08_reporting/shap_summary_class_1.png")
#             mlflow.log_artifact("data/08_reporting/shap_summary_class_0.png")
#         else:
#             # Fallback for other models or single-output cases
#             shap.summary_plot(shap_values, X_train_scaled, feature_names=X_train_scaled.columns, show=False)
#             plt.savefig("data/08_reporting/shap_summary.png")
#             plt.close()

#             mlflow.log_artifact("data/08_reporting/shap_summary.png")

#         # Feature importance
#         importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
#         importances.sort_values().plot(kind="barh")
#         plt.title("Tree-Based F.I")
#         plt.tight_layout()
#         plt.savefig("data/08_reporting/feature_importance.png")
#         plt.close()

#         mlflow.log_artifact("data/08_reporting/feature_importance.png")

#     return model, results

# """
# This is a boilerplate pipeline 'model_train'
# generated using Kedro 0.19.5
# """
# import pandas as pd
# import logging
# from typing import Dict, Tuple, Any
# import numpy as np
# import pickle
# import yaml
# import os
# import warnings
# import mlflow
# from mlflow.tracking import MlflowClient
# import shap
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, recall_score, precision_score
# from pathlib import Path

# logger = logging.getLogger(__name__)

# def model_train(
#         X_train_scaled: pd.DataFrame,
#         X_test_scaled: pd.DataFrame,
#         y_train_encoded: pd.DataFrame,
#         y_test_encoded: pd.DataFrame,
#         parameters_model_train: Dict[str, Any],
#         final_selected_features: list,
# ):
#     # Setup MLflow URI
#     with open("conf/local/mlflow.yml", "r") as f:
#         mlflow_config = yaml.load(f, Loader=yaml.SafeLoader)
#     mlflow.set_tracking_uri(mlflow_config["server"]["mlflow_tracking_uri"])

#     # Experiment and Run Names
#     experiment_name = parameters_model_train["experiment_name_train"]
#     experiment = mlflow.get_experiment_by_name(experiment_name)
#     experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

#     run_name = parameters_model_train["run_name_train"]

#     # ⚠️ Temporariamente desativado o autologging
#     # mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

#     logger.info(f"Starting MLFlow run {run_name}")
#     with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
#         # Load champion model
#         try:
#             with open(os.path.join(os.getcwd(), "data", "06_models", "champion_model.pkl"), "rb") as f:
#                 champ_model = pickle.load(f)
#         except FileNotFoundError:
#             logger.error("Champion model not fetched correctly.")
#             raise

#         results = {}
#         logger.info("Starting Training, with Selected features")
#         X_train_scaled = X_train_scaled[final_selected_features]
#         X_test_scaled = X_test_scaled[final_selected_features]

#         assert isinstance(y_train_encoded, pd.DataFrame)
#         assert isinstance(y_test_encoded, pd.DataFrame)

#         y_train_encoded = np.ravel(y_train_encoded)
#         y_test_encoded = np.ravel(y_test_encoded)

#         model = champ_model.fit(X_train_scaled, y_train_encoded)

#         # Predictions
#         y_train_pred = model.predict(X_train_scaled)
#         y_test_pred = model.predict(X_test_scaled)

#         # Metrics
#         f1_macro_train = f1_score(y_train_encoded, y_train_pred, average="macro")
#         f1_macro_test = f1_score(y_test_encoded, y_test_pred, average="macro")
#         f1_diff = f1_macro_train - f1_macro_test

#         mlflow.log_metric("F1_Macro_Train", f1_macro_train)
#         mlflow.log_metric("F1_Macro_Test", f1_macro_test)
#         mlflow.log_metric("F1_DIFF", f1_diff)

#         recall_train = recall_score(y_train_encoded, y_train_pred, average="macro")
#         recall_test = recall_score(y_test_encoded, y_test_pred, average="macro")
#         mlflow.log_metric("Recall-train", recall_train)
#         mlflow.log_metric("Recall-test", recall_test)

#         precision_train = precision_score(y_train_encoded, y_train_pred)
#         precision_test = precision_score(y_test_encoded, y_test_pred)
#         mlflow.log_metric("Precision-Train", precision_train)
#         mlflow.log_metric("Precision-Test", precision_test)

#         results = {
#             "classifier": champ_model.__class__.__name__,
#             "F1_Macro-Train": f1_macro_train,
#             "F1_Macro-Test": f1_macro_test,
#             "F1_DIFF": f1_diff,
#             "Recall-Train": recall_train,
#             "Recall-Test": recall_test,
#             "Precision-Train": precision_train,
#             "Precision-Test": precision_test,
#         }

#         # Register model in MLflow
#         run_id = mlflow.active_run().info.run_id
#         model_uri = f"runs:/{run_id}/model"
#         registered_model_name = parameters_model_train["model_registry_name_train"]

#         mlflow.register_model(model_uri=model_uri, name=registered_model_name)
#         logger.info(f"Model registered in MLflow as '{registered_model_name}' from run {run_id}")

#         client = MlflowClient()
#         model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version

#         tags = {
#             "author": "Pedro",
#             "model_type": "Champ_RF",
#             "training_run": run_id,
#             "F1_Macro-Train": round(f1_macro_train, 3),
#             "stage": "Champion_Pre-Deployment"
#         }
#         for key, value in tags.items():
#             client.set_model_version_tag(
#                 name=registered_model_name,
#                 version=model_version,
#                 key=key,
#                 value=value
#             )

#         # 🔒 Comentado: SHAP plots (podem bloquear)
#         # explainer_shap = shap.TreeExplainer(model)
#         # shap_values = explainer_shap(X_train_scaled)
#         # shap.summary_plot(...)

#         # 🔒 Desactivado por agora: logging de artefactos visuais
#         Path("data/08_reporting").mkdir(parents=True, exist_ok=True)

#         fi_path = "data/08_reporting/feature_importance.png"
#         try:
#             importances = pd.Series(model.feature_importances_, index=X_train_scaled.columns)
#             importances.sort_values().plot(kind="barh")
#             plt.title("Tree-Based F.I")
#             plt.tight_layout()
#             plt.savefig(fi_path)
#             plt.close()

#             if os.path.exists(fi_path):
#                 mlflow.log_artifact(fi_path)
#         except Exception as e:
#             logger.warning(f"Feature importance plot skipped: {e}")

#     # ✅ Garantir que a run termina mesmo que algo acima falhe silenciosamente
#     if mlflow.active_run():
#         mlflow.end_run()

#     return model, results


"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.19.5
"""
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

logger = logging.getLogger(__name__)

def model_train(
        X_train_scaled: pd.DataFrame,
        X_test_scaled: pd.DataFrame,
        y_train_encoded: pd.DataFrame,
        y_test_encoded: pd.DataFrame,
        parameters_model_train: Dict[str, Any],
        final_selected_features: list,
):
    # ---------------- SETUP EXPERIMENT ----------------
    with open("conf/local/mlflow.yml", "r") as f:
        mlflow_config = yaml.load(f, Loader=yaml.SafeLoader)
    mlflow.set_tracking_uri(mlflow_config["server"]["mlflow_tracking_uri"])

    experiment_name = parameters_model_train["experiment_name_train"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    run_name = parameters_model_train["run_name_train"]

    # ---------------- AUTOLOGGING ----------------
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    logger.info(f"Starting MLFlow run {run_name}")
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):

        # ---------------- LOAD MODEL ----------------
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

        # ---------------- TRAIN ----------------
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

        # ---------------- RESULTS DICT ----------------
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

        # ---------------- MODEL REGISTRY ----------------
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

        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_sample)
        shap.initjs()

        if isinstance(shap_values, list):
            for i, class_values in enumerate(shap_values):
                fig = plt.figure()
                shap.summary_plot(class_values, X_sample, feature_names=X_sample.columns, show=False)
                path = f"data/08_reporting/shap_summary_class_{i}.png"
                plt.savefig(path)
                mlflow.log_artifact(path)
                plt.close()
        else:
            fig = plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False)
            path = "data/08_reporting/shap_summary.png"
            plt.savefig(path)
            mlflow.log_artifact(path)
            plt.close()

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
