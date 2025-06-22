"""
This is a boilerplate pipeline 'model_development_selection'
generated using Kedro 0.19.12
"""

import logging
import mlflow
import numpy as np
import pandas as pd
import random

from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import optuna



logger = logging.getLogger(__name__)

## MAIN WRAPPER FUNCTION ##

def model_selection(
    X_train_preprocessed: pd.DataFrame,
    X_test_preprocessed: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_test_encoded: pd.Series,
    final_selected_features: list,
    parameters_model_selection: Dict[str, Any],
    parameters_grid: Dict[str, Any]  # Commented for Round 1
):
    seed = parameters_model_selection.get("random_state", 42)
    experiment_name = parameters_model_selection["run_name"]
    model_name = parameters_model_selection["model_registry_name"]

    seed_everything(seed)
    
    models_dict = define_models(seed)
    X_train_scaled, X_test_scaled = scale_features(
        X_train_preprocessed[final_selected_features],
        X_test_preprocessed[final_selected_features]
    )
    
    best_model, best_name, best_f1, best_model_class = train_base_models(
    models_dict, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, experiment_name
    )
    
    champion_model = run_grid_search(
    best_model=best_model,
    best_model_class=best_model_class,
    best_name=best_name,
    best_f1_base=best_f1,
    X_train=X_train_scaled,
    y_train=y_train_encoded,
    X_test=X_test_scaled,
    y_test=y_test_encoded,
    parameters_grid=parameters_grid,
    seed=seed
    )

    register_model(champion_model, best_name, model_name)
    return champion_model




## SUPPORTING FUNCTIONS #
def seed_everything(seed: int):
    """
    Set global seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

def define_models(seed: int) -> Dict[str, Any]:
    return {
        "RandomForestClassifier": RandomForestClassifier(random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "KNeighborsClassifier": KNeighborsClassifier()
    }

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_base_models(models_dict, X_train, y_train, X_test, y_test, experiment_name):
    mlflow.set_experiment(experiment_name)
    logger.info("Phase 1: Training base models and logging to MLflow...")
    results = {}
    valid_models = {}

    for name, model in models_dict.items():
        with mlflow.start_run(run_name=f"{name}_V2", nested=True):
            mlflow.set_tag("stage", "Tuned")
            mlflow.set_tag("Round", "3")
            mlflow.set_tag("Model_type", name)

            mlflow.set_tag("mlflow.note.content", "Hyperparameter Tuning, with 10 perc. control overfitting check.")

            mlflow.sklearn.autolog(log_models=True)
            model.fit(X_train, np.ravel(y_train))
            train_preds = model.predict(X_train)
            train_f1 = f1_score(y_train, train_preds, average="macro")
            mlflow.log_metric("train_f1_macro", train_f1)

            ## VAL
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")
            recall = recall_score(y_test, preds, average="macro")
            precision = precision_score(y_test, preds, average="macro")

            f1_diff = train_f1 - f1
            # Track overfitting
            f1_diff = train_f1 - f1
            mlflow.log_metric("f1_diff", train_f1 - f1)

            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("precision_macro", precision)
            
            if f1_diff < 0.1:
                valid_models[name] = (model, f1)
                
    if valid_models:
        best_name = max(valid_models, key=lambda k: valid_models[k][1])
        best_model = valid_models[best_name][0]
        best_f1 = valid_models[best_name][1]
        logger.info(f"Best valid model: {best_name} | F1_macro: {best_f1:.4f}")
    else:
        best_model, best_name, best_f1 = None, None, None
        logger.warning(" No valid models found with f1_diff < 0.1")

    return best_model, best_name, best_f1, best_model.__class__


def register_model(model, name:str, model_name: str):
    with mlflow.start_run(run_name=f"{name}_registered", nested=True) as run:
        logger.info(f"Logging and registering model {model_name}.")

        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name
        )

# ## GRID
def run_grid_search(
    best_model,
    best_model_class,
    best_name,
    best_f1_base,
    X_train,
    y_train,
    X_test,
    y_test,
    parameters_grid,
    seed
):
    logger.info(f"🔍 Phase 2: Running GridSearchCV on {best_name}...")

    param_grid = parameters_grid["hyperparameters"][best_name]

    try:
        base_estimator = best_model_class(random_state=seed)
    except TypeError:
        base_estimator = best_model_class()

    grid_model = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        n_jobs=-1,
        cv=3
    )

    with mlflow.start_run(run_name=f"{best_name}_gridsearch", nested=True):
        mlflow.set_tag("stage", "Tuned")
        mlflow.set_tag("Round", "3")
        mlflow.set_tag("Model_type", best_name)
        mlflow.sklearn.autolog()
        

        grid_model.fit(X_train, np.ravel(y_train))
        tuned_model = grid_model.best_estimator_

        preds = tuned_model.predict(X_test)
        best_f1_tuned = f1_score(y_test, preds, average="macro")

        mlflow.log_params(grid_model.best_params_)
        logger.info(f"Tuned {best_name} | F1_macro: {best_f1_tuned:.4f}")

        if best_f1_tuned > best_f1_base:
            logger.info(f"🎯 Champion: Tuned {best_name} selected.")
            return tuned_model
        else:
            logger.info(f"⚖️ Champion: Base {best_name} retained.")
            return best_model



# ## OPTUNA ##
# def optuna_objective(trial: optuna.Trial, model_name: str, X_train, X_test, y_train, y_test) -> float:
#     if model_name == "RandomForestClassifier":
#         model = RandomForestClassifier(
#             n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
#             max_depth=trial.suggest_categorical("max_depth", [5, 6, 7, 8]),
#             random_state=2021,
#         )
#     elif model_name == "GradientBoostingClassifier":
#         model = GradientBoostingClassifier(
#             n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
#             max_depth=trial.suggest_categorical("max_depth", [4, 5, 6, 7]),
#             learning_rate=trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.01]),
#             random_state=2021,
#         )
#     elif model_name == "XGBClassifier":
#         model = XGBClassifier(
#             n_estimators=trial.suggest_categorical("n_estimators", [100, 200, 300]),
#             max_depth=trial.suggest_categorical("max_depth", [4, 5, 6, 7]),
#             learning_rate=trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.01]),
#             use_label_encoder=False,
#             eval_metric="logloss",
#             random_state=2021,
#         )
#     else:
#         raise ValueError("Unsupported model")

#     model.fit(X_train, np.ravel(y_train))
#     preds = model.predict(X_test)
#     f1 = f1_score(y_test, preds, average="macro")

#     mlflow.log_params(trial.params)
#     mlflow.log_metric("f1_macro", f1)

#     return f1


#  def run_optuna_optimization(X_train, X_test, y_train, y_test, model_name: str, n_trials: int = 25):
#     mlflow.set_experiment("Optuna_Tuning")
    
#     with mlflow.start_run(run_name=f"{model_name}_Optuna_R2"):
#         mlflow.set_tag("stage", "Tuned")
#         mlflow.set_tag("Round", "2")
#         mlflow.set_tag("Model_type", model_name)
        
#         study = optuna.create_study(direction="maximize")
#         study.optimize(
#             lambda trial: optuna_objective(trial, model_name, X_train, X_test, y_train, y_test),
#             n_trials=n_trials
#         )

#         best_params = study.best_params
#         best_score = study.best_value
#         mlflow.log_params(best_params)
#         mlflow.log_metric("best_f1_macro", best_score)

#         print(f"🏆 Best F1_macro: {best_score:.4f}")
#         print(f"Best Params: {best_params}")

#         return best_params, best_score
   



# def compare_models(base_model, tuned_model, base_f1, tuned_f1, name, register_model: bool = True, model_name: str = "_"):
#    if tuned_f1 > base_f1:
#         logger.info(f"Champion: Tuned {name} (F1_macro: {tuned_f1:.4f}) > Base (F1_macro: {base_f1:.4f})")
#         champion_model = tuned_model
#     else:
#         logger.info(f"Champion: Base {name} (F1_macro: {base_f1:.4f}) retained over Tuned (F1_macro: {tuned_f1:.4f})")
#         champion_model = base_model

#     if register_model:
#         with mlflow.start_run(run_name=f"{name}_register", nested=True) as run:
#             logger.info(f"📦 Logging and registering model {model_name}...")
#             mlflow.sklearn.log_model(champion_model, artifact_path="model")

#             mlflow.register_model(
#                 model_uri=f"runs:/{run.info.run_id}/model",
#                 name=model_name
#             )

#     return champion_model