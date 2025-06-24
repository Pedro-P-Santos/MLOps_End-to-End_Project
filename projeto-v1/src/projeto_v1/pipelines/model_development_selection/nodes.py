"""
This is a boilerplate pipeline 'model_development_selection'
generated using Kedro 0.19.12
"""

import logging
import mlflow
import numpy as np
import pandas as pd
import random
import optuna
from optuna.samplers import TPESampler

from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
from datetime import datetime

logger = logging.getLogger(__name__)

# -----------------
# Phase 0: Auxiliaries
# -----------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def define_models(seed):
    return {
        "RandomForestClassifier": RandomForestClassifier(random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "KNeighborsClassifier": KNeighborsClassifier()
    }

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# -----------------
# Phase 1: Base models (no tuning)
# -----------------
def train_base_models(models_dict, X_train, y_train, X_test, y_test, experiment_name):
    mlflow.set_experiment(experiment_name)
    results = {}
    for name, model in models_dict.items():
        with mlflow.start_run(run_name=f"{name}_base", nested=True):
            mlflow.sklearn.autolog(log_models=True)
            model.fit(X_train, np.ravel(y_train))
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")
            results[name] = (model, f1)
            logger.info(f"Model: {name} | F1_macro: {f1:.4f}")
    best_name = max(results, key=lambda k: results[k][1])
    return results[best_name][0], best_name, results[best_name][1]

# -----------------
# Phase 2: Hyperparameter tuning with Optuna (from YML)
# -----------------
def optuna_tuning(model_name, X_train, y_train, X_test, y_test, grid_params, seed, n_trials=10, experiment_name="optuna_tuning"):
    mlflow.set_experiment(experiment_name)
    def objective(trial):
        params = {}
        for param, values in grid_params.items():
            if isinstance(values, list):
                params[param] = trial.suggest_categorical(param, values)
            else:
                params[param] = values

        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(**params)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(**params)
        elif model_name == "XGBClassifier":
            model = XGBClassifier(**params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_train, np.ravel(y_train))
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average="macro")
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params

    # Retrain with best params for metrics logging
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**best_params)
    elif model_name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(**best_params)
    elif model_name == "XGBClassifier":
        model = XGBClassifier(**best_params)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**best_params)
    elif model_name == "KNeighborsClassifier":
        model = KNeighborsClassifier(**best_params)
    model.fit(X_train, np.ravel(y_train))
    preds = model.predict(X_test)
    
    metrics = {
        "optuna_f1_macro": f1_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "accuracy": accuracy_score(y_test, preds)
    }

    with mlflow.start_run(run_name=f"{model_name}_optuna", nested=True):
        mlflow.log_params(best_params)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        logger.info(f"[Optuna] {model_name} best params: {best_params} | Metrics: {metrics}")

    return model, metrics["optuna_f1_macro"], best_params, metrics

# -----------------
# Phase 3: Register champion model (with description and tags)
# -----------------
def register_champion(model, model_name, mlflow_name, metrics=None, description=None, tags=None):
    safe_mlflow_name = mlflow_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    mlflow.set_experiment("optuna_and_champion")
    with mlflow.start_run(run_name=f"{model_name}_champion", nested=True) as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        if metrics:
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=safe_mlflow_name
        )
        client = MlflowClient()
        if description:
            client.update_registered_model(
                name=safe_mlflow_name,
                description=description
            )
        if tags:
            for key, value in tags.items():
                client.set_registered_model_tag(name=safe_mlflow_name, key=key, value=value)

# -----------------
# Main orchestrator function (pipeline)
# -----------------
def model_selection(
    X_train_preprocessed: pd.DataFrame,
    X_test_preprocessed: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_test_encoded: pd.Series,
    final_selected_features: list,
    parameters_model_selection: Dict[str, Any],
    parameters_grid: Dict[str, Any]
):
    """
    PHASE 0: Preparation
    PHASE 1: Train base models (separate experiment)
    PHASE 2: Hyperparameter tuning of the best base model via Optuna (main experiment)
    PHASE 3: Champion registration and output (main experiment)
    """

    # PHASE 0: Preparation
    seed = parameters_model_selection.get("random_state", 42)
    experiment_name = parameters_model_selection["run_name"]
    mlflow_name = parameters_model_selection["model_registry_name"]

    seed_everything(seed)
    models_dict = define_models(seed)
    X_train = X_train_preprocessed[final_selected_features]
    X_test = X_test_preprocessed[final_selected_features]
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # PHASE 1: Baseline
    base_model, best_name, base_f1 = train_base_models(
        models_dict, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, "baseline_models"
    )

    mlflow.sklearn.autolog(disable=True)

    # PHASE 2: Optuna tuning + Champion (no mesmo experiment principal)
    tuned_model, tuned_f1, best_params, tuned_metrics = optuna_tuning(
        best_name, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded,
        parameters_grid["hyperparameters"][best_name], seed, n_trials=15, experiment_name="optuna_and_champion"
    )

    # PHASE 3: Champion registration (no experiment principal)
    if tuned_f1 > base_f1:
        champion = tuned_model
        champion_metrics = tuned_metrics
        description = "Champion model selected via Optuna hyperparameter tuning."
        phase = "champion"
        best_score = champion_metrics.get("optuna_f1_macro")
    else:
        champion = base_model
        preds = champion.predict(X_test_scaled)
        champion_metrics = {
            "f1_macro": f1_score(y_test_encoded, preds, average="macro"),
            "recall_macro": recall_score(y_test_encoded, preds, average="macro"),
            "precision_macro": precision_score(y_test_encoded, preds, average="macro"),
            "accuracy": accuracy_score(y_test_encoded, preds)
        }
        description = "Champion model is the best baseline (no tuning performed)."
        phase = "champion"
        best_score = champion_metrics.get("f1_macro")

        # AUTOMATIC TAGS
        algorithm = type(champion).__name__
        run_date = datetime.now().strftime("%Y-%m-%d")
        project = parameters_model_selection.get("run_name", "unknown_project")
        autor = "Rodrigo"
        data_version = parameters_model_selection.get("data_version", "unknown")

        tags = {
            "project": project,
            "phase": phase,
            "autor": autor,
            "algorithm": algorithm,
            "data_version": data_version,
            "best_score": round(best_score, 4),
            "run_date": run_date
        }

        register_champion(
            champion, best_name, mlflow_name,
            metrics=champion_metrics, description=description, tags=tags
        )
        # return champion
        return (
            champion,
            pd.DataFrame(X_train_scaled, columns=final_selected_features),
            pd.DataFrame(X_test_scaled, columns=final_selected_features)
        )


































# import logging
# import mlflow
# import numpy as np
# import pandas as pd
# import random
# import optuna
# from optuna.samplers import TPESampler

# from typing import Dict, Any

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
# from sklearn.preprocessing import StandardScaler

# logger = logging.getLogger(__name__)

# # -----------------
# # Phase 0: Auxiliaries
# # -----------------
# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)

# def define_models(seed):
#     return {
#         "RandomForestClassifier": RandomForestClassifier(random_state=seed),
#         "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
#         "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed),
#         "LogisticRegression": LogisticRegression(max_iter=200),
#         "KNeighborsClassifier": KNeighborsClassifier()
#     }

# def scale_features(X_train, X_test):
#     scaler = StandardScaler()
#     return scaler.fit_transform(X_train), scaler.transform(X_test)

# # -----------------
# # Phase 1: Base models (no tuning)
# # -----------------
# def train_base_models(models_dict, X_train, y_train, X_test, y_test, experiment_name):
#     mlflow.set_experiment(experiment_name)
#     results = {}
#     for name, model in models_dict.items():
#         with mlflow.start_run(run_name=f"{name}_base", nested=True):
#             mlflow.sklearn.autolog(log_models=True)
#             model.fit(X_train, np.ravel(y_train))
#             preds = model.predict(X_test)
#             f1 = f1_score(y_test, preds, average="macro")
#             results[name] = (model, f1)
#             logger.info(f"Model: {name} | F1_macro: {f1:.4f}")
#     best_name = max(results, key=lambda k: results[k][1])
#     return results[best_name][0], best_name, results[best_name][1]

# # -----------------
# # Phase 2: Hyperparameter tuning with Optuna (from YML)
# # -----------------
# def optuna_tuning(model_name, X_train, y_train, X_test, y_test, grid_params, seed, n_trials=10):
#     def objective(trial):
#         params = {}
#         for param, values in grid_params.items():
#             if isinstance(values, list):
#                 params[param] = trial.suggest_categorical(param, values)
#             else:
#                 params[param] = values

#         if model_name == "RandomForestClassifier":
#             model = RandomForestClassifier(**params)
#         elif model_name == "GradientBoostingClassifier":
#             model = GradientBoostingClassifier(**params)
#         elif model_name == "XGBClassifier":
#             model = XGBClassifier(**params)
#         elif model_name == "LogisticRegression":
#             model = LogisticRegression(**params)
#         elif model_name == "KNeighborsClassifier":
#             model = KNeighborsClassifier(**params)
#         else:
#             raise ValueError(f"Unsupported model: {model_name}")

#         model.fit(X_train, np.ravel(y_train))
#         preds = model.predict(X_test)
#         score = f1_score(y_test, preds, average="macro")
#         return score

#     study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
#     study.optimize(objective, n_trials=n_trials)
#     best_params = study.best_trial.params

#     # Retrain with best params for metrics logging
#     if model_name == "RandomForestClassifier":
#         model = RandomForestClassifier(**best_params)
#     elif model_name == "GradientBoostingClassifier":
#         model = GradientBoostingClassifier(**best_params)
#     elif model_name == "XGBClassifier":
#         model = XGBClassifier(**best_params)
#     elif model_name == "LogisticRegression":
#         model = LogisticRegression(**best_params)
#     elif model_name == "KNeighborsClassifier":
#         model = KNeighborsClassifier(**best_params)
#     model.fit(X_train, np.ravel(y_train))
#     preds = model.predict(X_test)
    
#     metrics = {
#         "optuna_f1_macro": f1_score(y_test, preds, average="macro"),
#         "recall_macro": recall_score(y_test, preds, average="macro"),
#         "precision_macro": precision_score(y_test, preds, average="macro"),
#         "accuracy": accuracy_score(y_test, preds)
#     }

#     with mlflow.start_run(run_name=f"{model_name}_optuna", nested=True):
#         mlflow.log_params(best_params)
#         for metric_name, value in metrics.items():
#             mlflow.log_metric(metric_name, value)
#         logger.info(f"[Optuna] {model_name} best params: {best_params} | Metrics: {metrics}")

#     return model, metrics["optuna_f1_macro"], best_params, metrics

# # -----------------
# # Phase 3: Register champion model
# # -----------------
# def register_champion(model, model_name, mlflow_name, metrics=None):
#     safe_mlflow_name = mlflow_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
#     with mlflow.start_run(run_name=f"{model_name}_champion", nested=True) as run:
#         mlflow.sklearn.log_model(model, artifact_path="model")
#         if metrics:
#             for metric_name, value in metrics.items():
#                 mlflow.log_metric(metric_name, value)
#         mlflow.register_model(
#             model_uri=f"runs:/{run.info.run_id}/model",
#             name=safe_mlflow_name
#         )

# # -----------------
# # Main orchestrator function (pipeline)
# # -----------------
# def model_selection(
#     X_train_preprocessed: pd.DataFrame,
#     X_test_preprocessed: pd.DataFrame,
#     y_train_encoded: pd.Series,
#     y_test_encoded: pd.Series,
#     final_selected_features: list,
#     parameters_model_selection: Dict[str, Any],
#     parameters_grid: Dict[str, Any]
# ):
#     """
#     PHASE 0: Preparation
#     PHASE 1: Train base models
#     PHASE 2: Hyperparameter tuning of the best base model via Optuna
#     PHASE 3: Champion registration and output
#     """

#     # PHASE 0: Preparation
#     seed = parameters_model_selection.get("random_state", 42)
#     experiment_name = parameters_model_selection["run_name"]
#     mlflow_name = parameters_model_selection["model_registry_name"]

#     seed_everything(seed)
#     models_dict = define_models(seed)
#     X_train = X_train_preprocessed[final_selected_features]
#     X_test = X_test_preprocessed[final_selected_features]
#     X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

#     # PHASE 1: Base models
#     base_model, best_name, base_f1 = train_base_models(
#         models_dict, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, experiment_name
#     )

#     mlflow.sklearn.autolog(disable=True)
    
#     # PHASE 2: Optuna tuning
#     grid_params = parameters_grid["hyperparameters"][best_name]
#     tuned_model, tuned_f1, best_params, tuned_metrics = optuna_tuning(
#         best_name, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, grid_params, seed, n_trials=15
#     )

#     # PHASE 3: Champion selection and registration
#     if tuned_f1 > base_f1:
#         champion = tuned_model
#         champion_metrics = tuned_metrics
#         logger.info(f"Champion: {best_name} Optuna (F1={tuned_f1:.4f}) > Base (F1={base_f1:.4f})")
#     else:
#         champion = base_model
#         preds = champion.predict(X_test_scaled)
#         champion_metrics = {
#             "f1_macro": f1_score(y_test_encoded, preds, average="macro"),
#             "recall_macro": recall_score(y_test_encoded, preds, average="macro"),
#             "precision_macro": precision_score(y_test_encoded, preds, average="macro"),
#             "accuracy": accuracy_score(y_test_encoded, preds)
#         }
#         logger.info(f"Champion: {best_name} Base (F1={base_f1:.4f}) >= Optuna (F1={tuned_f1:.4f})")

#     register_champion(champion, best_name, mlflow_name, metrics=champion_metrics)
#     return champion


