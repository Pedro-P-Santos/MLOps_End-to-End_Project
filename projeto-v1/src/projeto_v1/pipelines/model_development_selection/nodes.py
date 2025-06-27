# import pandas as pd
# import numpy as np
# import logging
# import mlflow
# import optuna
# import warnings
# from typing import Dict, Any, Tuple
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import StandardScaler
# from mlflow.tracking import MlflowClient

# warnings.filterwarnings("ignore", category=Warning)
# logger = logging.getLogger(__name__)


# def _get_or_create_experiment_id(experiment_name: str) -> str:
#     exp = mlflow.get_experiment_by_name(experiment_name)
#     if exp is None:
#         logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
#         return mlflow.create_experiment(experiment_name)
#     return exp.experiment_id


# def get_model_class(model_name):
#     return {
#         "RandomForestClassifier": RandomForestClassifier,
#         "GradientBoostingClassifier": GradientBoostingClassifier,
#         "LogisticRegression": LogisticRegression,
#         "KNeighborsClassifier": KNeighborsClassifier,
#         "XGBClassifier": XGBClassifier,
#     }[model_name]


# def get_optuna_search_space(trial, param_grid: dict):
#     params = {}
#     for param, values in param_grid.items():
#         if isinstance(values[0], bool):
#             params[param] = trial.suggest_categorical(param, values)
#         elif isinstance(values[0], int) or isinstance(values[0], float):
#             if len(values) == 1:
#                 params[param] = values[0]
#             else:
#                 params[param] = trial.suggest_categorical(param, values)
#         else:
#             params[param] = trial.suggest_categorical(param, values)
#     return params


# def model_selection(
#     X_train_preprocessed: pd.DataFrame,
#     X_test_preprocessed: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame,
#     parameters_model_selection: Dict[str, Any],
#     parameters_grid: Dict[str, Any],
#     final_selected_features: list,
# ) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    
#     # Filtrar apenas as colunas selecionadas
#     X_train_selected = X_train_preprocessed[final_selected_features].copy()
#     X_test_selected = X_test_preprocessed[final_selected_features].copy()

#     # Aplicar StandardScaler
#     scaler = StandardScaler()
#     X_train = pd.DataFrame(
#         scaler.fit_transform(X_train_selected),
#         columns=final_selected_features,
#         index=X_train_selected.index
#     )
#     X_test = pd.DataFrame(
#         scaler.transform(X_test_selected),
#         columns=final_selected_features,
#         index=X_test_selected.index
#     )

#     experiment_name = parameters_model_selection["mlflow_experiment_name"]
#     model_registry_name = parameters_model_selection["model_registry_name"]
#     experiment_id = _get_or_create_experiment_id(experiment_name)
#     logger.info(f"Experiment ID: {experiment_id}")

#     models_dict = {
#         'RandomForestClassifier': RandomForestClassifier(random_state=2021),
#         'GradientBoostingClassifier': GradientBoostingClassifier(random_state=2021),
#         'LogisticRegression': LogisticRegression(max_iter=200),
#         'KNeighborsClassifier': KNeighborsClassifier(),
#         'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=2021),
#     }
#     baseline_results = {}

#     for model_name, model in models_dict.items():
#         with mlflow.start_run(run_name=f"{model_name}_baseline", experiment_id=experiment_id, nested=True):
#             mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
#             logger.info(f"Treinando baseline: {model_name}")
#             y_train_flat = np.ravel(y_train)
#             model.fit(X_train, y_train_flat)
#             preds = model.predict(X_test)
#             f1 = f1_score(y_test, preds, average="macro")
#             baseline_results[model_name] = f1
#             mlflow.log_metric("f1_macro", f1)
#             print(f"Baseline {model_name}: F1 macro = {f1:.4f}")

#     best_model_name = max(baseline_results, key=baseline_results.get)
#     best_baseline_score = baseline_results[best_model_name]
#     logger.info(f"Melhor baseline: {best_model_name} (F1 macro = {best_baseline_score:.4f})")

#     param_grid = parameters_grid["hyperparameters"][best_model_name]
#     model_class = get_model_class(best_model_name)

#     # Desativar autologging para evitar lentidão nos trials
#     mlflow.sklearn.autolog(disable=True)

#     def objective(trial):
#         params = get_optuna_search_space(trial, param_grid)
#         model = model_class(**params)
#         model.fit(X_train, np.ravel(y_train))
#         preds = model.predict(X_test)
#         f1 = f1_score(y_test, preds, average="macro")
#         return f1

#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=15, show_progress_bar=True)

#     best_optuna_params = study.best_params
#     best_optuna_score = study.best_value

#     best_model = model_class(**best_optuna_params)
#     best_model.fit(X_train, np.ravel(y_train))

#     if best_optuna_score > best_baseline_score:
#         champion_model = best_model
#         champion_score = best_optuna_score
#         phase = "champion_optuna"
#     else:
#         champion_model = models_dict[best_model_name]
#         champion_score = best_baseline_score
#         phase = "champion_baseline"

#     with mlflow.start_run(run_name=f"Model_Selection: {phase.upper()}_champion", experiment_id=experiment_id, nested=True):
#         mlflow.sklearn.log_model(
#             champion_model,
#             artifact_path="model",
#             registered_model_name=model_registry_name,
#             input_example=X_test.iloc[:1],  # Podes remover esta linha se continuar lento
#         )
#         mlflow.set_tag("algorithm", best_model_name)
#         mlflow.set_tag("best_score", np.round(champion_score, 4))
#         mlflow.set_tag("phase", phase)
#         mlflow.set_tag("autor", "Rodrigo")

#         client = MlflowClient()
#         client.set_registered_model_tag(model_registry_name, "algorithm", best_model_name)
#         client.set_registered_model_tag(model_registry_name, "best_score", str(np.round(champion_score, 4)))
#         client.set_registered_model_tag(model_registry_name, "phase", phase)
#         client.set_registered_model_tag(model_registry_name, "autor", "Rodrigo")

#         print(f"Champion model: {best_model_name} - F1 macro: {champion_score:.4f} ({phase})")

#     return champion_model, X_train, X_test


import pandas as pd
import numpy as np
import logging
import mlflow
import optuna
import warnings
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
import matplotlib
matplotlib.use("Agg")


warnings.filterwarnings("ignore", category=Warning)
logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def get_model_class(model_name):
    return {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "XGBClassifier": XGBClassifier,
    }[model_name]


def get_optuna_search_space(trial, param_grid: dict):
    params = {}
    for param, values in param_grid.items():
        if isinstance(values[0], bool):
            params[param] = trial.suggest_categorical(param, values)
        elif isinstance(values[0], (int, float)):
            if len(values) == 1:
                params[param] = values[0]
            else:
                params[param] = trial.suggest_categorical(param, values)
        else:
            params[param] = trial.suggest_categorical(param, values)
    return params


def log_metrics_and_confusion(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    precision_macro = precision_score(y_test, preds, average="macro")
    recall_macro = recall_score(y_test, preds, average="macro")
    precision_weighted = precision_score(y_test, preds, average="weighted")
    recall_weighted = recall_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("precision_weighted", precision_weighted)
    mlflow.log_metric("recall_weighted", recall_weighted)

    try:
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, proba[:, 1])
        else:
            roc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        mlflow.log_metric("roc_auc", roc)
    except Exception as e:
        logger.warning(f"ROC AUC not computed: {e}")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    path = "confusion_matrix.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close(fig)
    os.remove(path)

    return f1_macro


def model_selection(
    X_train_preprocessed: pd.DataFrame,
    X_test_preprocessed: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters_model_selection: Dict[str, Any],
    parameters_grid: Dict[str, Any],
    final_selected_features: list,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:

    X_train_selected = X_train_preprocessed[final_selected_features].copy()
    X_test_selected = X_test_preprocessed[final_selected_features].copy()

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=final_selected_features,
        index=X_train_selected.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=final_selected_features,
        index=X_test_selected.index
    )

    experiment_name = parameters_model_selection["mlflow_experiment_name"]
    model_registry_name = parameters_model_selection["model_registry_name"]
    experiment_id = _get_or_create_experiment_id(experiment_name)
    logger.info(f"Experiment ID: {experiment_id}")

    models_dict = {
        'RandomForestClassifier': RandomForestClassifier(random_state=2021),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=2021),
        'LogisticRegression': LogisticRegression(max_iter=200),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=2021),
    }
    baseline_results = {}

    for model_name, model in models_dict.items():
        with mlflow.start_run(run_name=f"{model_name}_baseline", experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            logger.info(f"Training baseline: {model_name}")
            y_train_flat = np.ravel(y_train)
            model.fit(X_train, y_train_flat)
            f1 = log_metrics_and_confusion(model, X_test, y_test)
            baseline_results[model_name] = f1
            print(f"Baseline {model_name}: F1 macro = {f1:.4f}")

    best_model_name = max(baseline_results, key=baseline_results.get)
    best_baseline_score = baseline_results[best_model_name]
    logger.info(f"Best baseline: {best_model_name} (F1 macro = {best_baseline_score:.4f})")

    param_grid = parameters_grid["hyperparameters"][best_model_name]
    model_class = get_model_class(best_model_name)

    mlflow.sklearn.autolog(disable=True)

    def objective(trial):
        params = get_optuna_search_space(trial, param_grid)
        model = model_class(**params)
        model.fit(X_train, np.ravel(y_train))
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=True)

    best_optuna_params = study.best_params
    best_optuna_score = study.best_value

    best_model = model_class(**best_optuna_params)
    best_model.fit(X_train, np.ravel(y_train))

    if best_optuna_score > best_baseline_score:
        champion_model = best_model
        champion_score = best_optuna_score
        phase = "champion_optuna"
    else:
        champion_model = models_dict[best_model_name]
        champion_score = best_baseline_score
        phase = "champion_baseline"

    with mlflow.start_run(run_name=f"Model_Selection: {phase.upper()}_champion", experiment_id=experiment_id, nested=True):
        f1 = log_metrics_and_confusion(champion_model, X_test, y_test)

        mlflow.sklearn.log_model(
            champion_model,
            artifact_path="model",
            registered_model_name=model_registry_name,
            input_example=X_test.iloc[:1],
        )
        mlflow.set_tag("algorithm", best_model_name)
        mlflow.set_tag("best_score", np.round(champion_score, 4))
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("author", "Rodrigo")

        client = MlflowClient()
        client.set_registered_model_tag(model_registry_name, "algorithm", best_model_name)
        client.set_registered_model_tag(model_registry_name, "best_score", str(np.round(champion_score, 4)))
        client.set_registered_model_tag(model_registry_name, "phase", phase)
        client.set_registered_model_tag(model_registry_name, "author", "Rodrigo")

        print(f"Champion model: {best_model_name} - F1 macro: {champion_score:.4f} ({phase})")

    return champion_model, X_train, X_test
