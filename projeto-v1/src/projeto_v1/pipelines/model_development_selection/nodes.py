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
from optuna.samplers import TPESampler


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
    
    best_model, best_name, best_f1_base, best_model_class = train_base_models(
    models_dict, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, experiment_name
    )
    
    tuned_model, optuna_f1 = run_optuna_for_all_models(
        X_train=X_train_scaled,
        y_train=y_train_encoded,
        X_test=X_test_scaled,
        y_test=y_test_encoded,
        parameters_grid=parameters_grid,
        seed=seed
    )

    register_model(champion_model, best_name, model_name)

    champion_model = compare_models(
        base_model=best_model,
        final_model=tuned_model,
        base_f1=best_f1_base,
        tuned_f1=optuna_f1,
        name=best_name,
        register_model=True,
        model_name="ChampionModel"
    )

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
    # valid_models = {}

    for name, model in models_dict.items():
        with mlflow.start_run(run_name=f"{name}_V2", nested=True):
            mlflow.set_tag("stage", "Optuna")
            mlflow.set_tag("Round", "4")
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
            
            # if f1_diff < 0.1:
            #     valid_models[name] = (model, f1)
        results[name] =  (model,f1)

        best_name = max(results, key=lambda k: results[k][1])
        best_model = results[best_name][0]
        best_f1 = results[best_name][1]
        logger.info(f"Best valid model: {best_name} | F1_macro: {best_f1:.4f}")

    return best_model, best_name, best_f1, best_model.__class__


def register_model(model, name:str, model_name: str):
    with mlflow.start_run(run_name=f"{name}_registered", nested=True) as run:
        logger.info(f"Logging and registering model {model_name}.")

        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name
        )



## OPTUNA ##
def run_optuna_for_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    parameters_grid,
    seed,
    n_trials= 5
):
    candidates = ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "KNeighborsClassifier"]
    results = {}

    for name in candidates:
        model = run_optuna_search(
            best_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            parameters_grid=parameters_grid,
            seed=seed,
            n_trials=n_trials
        )
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        results[name] = (model, f1)

    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    logger.info(f"🏆 Best Optuna Model: {best_name} | F1_macro: {results[best_name][1]:.4f}")
    return best_model, best_name


def run_optuna_search(
        best_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        parameters_grid: Dict[str,Any],
        seed: int,
        n_trials: int = 5
):
    logger.info(f"Moved on from Classical GridSearch to Optuna -> Phase 4.")
    logger.info(f"Running Optuna on {best_name}")
    

    def objective(trial):
        if best_name == "RandomForestClassifier":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [150, 200, 250, 300, 350, 400]),
                "max_depth": trial.suggest_categorical("max_depth", [5, 6, 7, 8]),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            }
            model = RandomForestClassifier(**params)

        elif best_name == "GradientBoostingClassifier":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [150, 200, 250, 300, 350, 400]),
                "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.01]),
                "random_state": seed
            }
            model = GradientBoostingClassifier(**params)

        elif best_name == "XGBClassifier":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [150, 200, 250, 300, 350, 400]),
                "max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.01]),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": seed
            }
            model = XGBClassifier(**params)

        elif best_name == "KNeighborsClassifier":
            params = {
                "n_neighbors": trial.suggest_categorical("n_neighbors", [3, 4, 5 ,6 , 7, 8]),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski"])
            }
            model = KNeighborsClassifier(**params)

        else:
            raise ValueError(f"Optuna not implemented for model: {best_name}")

        model.fit(X_train, np.ravel(y_train))
        preds = model.predict(X_test)
        f1_macro = f1_score(y_test, preds, average="macro")
        return f1_macro
    
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    with mlflow.start_run(run_name=f"{best_name}_optuna", nested=True):

        mlflow.set_tag("stage", "Optuna")
        mlflow.set_tag("Round", "4")
        mlflow.set_tag("Model_type", best_name)
        mlflow.set_tag("mlflow.note.content", "OPTUNA RUN")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_trial.params
        mlflow.log_params(best_params)

        logger.info(f" Best Params via Optuna: {best_params}")

        # Final model training
        if best_name == "RandomForestClassifier":
            final_model = RandomForestClassifier(**best_params, random_state=seed)
        elif best_name == "GradientBoostingClassifier":
            final_model = GradientBoostingClassifier(**best_params, random_state=seed)
        elif best_name == "XGBClassifier":
            final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=seed)
        elif best_name == "KNeighborsClassifier":
            final_model = KNeighborsClassifier(**best_params)

        final_model.fit(X_train, np.ravel(y_train))
        preds = final_model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("optuna_f1_macro", f1)

    return final_model, f1




   



def compare_models(
    base_model,
    final_model,
    base_f1: float,
    tuned_f1: float,
    name: str,
    register_model: bool = True,
    model_name: str = "_"
):
    """
    Compare base and tuned models using F1 score and register the best (champion) to MLflow.
    """

    if tuned_f1 > base_f1:
        logger.info(f"Champion: Tuned {name} selected (F1_macro: {tuned_f1:.4f} > {base_f1:.4f})")
        champion_model = final_model
    else:
        logger.info(f"Champion: Base {name} retained (F1_macro: {base_f1:.4f} ≥ {tuned_f1:.4f})")
        champion_model = base_model

    if register_model:
        with mlflow.start_run(run_name=f"{name}_register", nested=True) as run:
            logger.info(f"Logging and registering model to MLflow: {model_name}")
            mlflow.sklearn.log_model(champion_model, artifact_path="model")

            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name
            )

    return champion_model
