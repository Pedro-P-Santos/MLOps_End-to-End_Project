"""
This is a boilerplate pipeline 'model_development_selection'
generated using Kedro 0.19.12
"""

import logging
import mlflow
import numpy as np
import pandas as pd
import random

from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    final_selected_features: list,
    parameters_model_selection: Dict[str, Any],
    parameters_grid: Dict[str, Any]
):
    """
    Phase 1 & 2 - Model Selection and Hyperparameter Tuning
    --------------------------------------------------------
    This function performs:
    - Phase 1: Training of base models (no tuning), MLflow logging, F1_macro scoring
    - Phase 2: Grid search on best model using predefined hyperparameters
    - Compares base vs tuned model and returns best one
    """

    # Reproducibility: set global seeds
    seed = parameters_model_selection.get("random_state", 42)
    random.seed(seed)
    np.random.seed(seed)

    # Phase 0 - Define models to evaluate (with random_state where applicable)
    models_dict = {
        "RandomForestClassifier": RandomForestClassifier(random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=200),  # random_state opcional, depende do solver
        "KNeighborsClassifier": KNeighborsClassifier()  # não usa random_state
    }

    # Phase 1 - Filter features
    X_train = X_train[final_selected_features]
    X_test = X_test[final_selected_features]

    results = {}

    mlflow.set_experiment(parameters_model_selection["run_name"])
    logger.info("Phase 1: Training base models and logging to MLflow...")

    for name, model in models_dict.items():
        with mlflow.start_run(run_name=f"{name}_base", nested=True):
            mlflow.sklearn.autolog()
            model.fit(X_train, np.ravel(y_train))
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")
            results[name] = (model, f1)
            logger.info(f"Model: {name} | F1_macro: {f1:.4f}")

    # Select best base model
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    best_f1_base = results[best_name][1]
    best_model_class = models_dict[best_name].__class__
    logger.info(f"Best base model: {best_name} with F1_macro: {best_f1_base:.4f}")

    # Phase 2 - Grid search on best model
    logger.info(f"Phase 2: Running GridSearchCV on {best_name}...")

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
        cv=5
    )

    with mlflow.start_run(run_name=f"{best_name}_gridsearch", nested=True):
        mlflow.sklearn.autolog()
        grid_model.fit(X_train, np.ravel(y_train))
        tuned_model = grid_model.best_estimator_
        preds = tuned_model.predict(X_test)
        best_f1_tuned = f1_score(y_test, preds, average="macro")
        logger.info(f"Tuned {best_name} | F1_macro: {best_f1_tuned:.4f}")

    # Compare and return final champion
    if best_f1_tuned > best_f1_base:
        logger.info(f"Champion: Tuned {best_name} (F1_macro: {best_f1_tuned:.4f}) > Base (F1_macro: {best_f1_base:.4f})")
        return tuned_model
    else:
        logger.info(f"Champion: Base {best_name} (F1_macro: {best_f1_base:.4f}) retained over Tuned (F1_macro: {best_f1_tuned:.4f})")
        return best_model
