

import pandas as pd
import numpy as np
import logging
import mlflow
import optuna
import warnings
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient

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
        elif isinstance(values[0], int):
            if len(values) == 1:
                params[param] = values[0]
            else:
                params[param] = trial.suggest_categorical(param, values)
        elif isinstance(values[0], float):
            if len(values) == 1:
                params[param] = values[0]
            else:
                params[param] = trial.suggest_categorical(param, values)
        else:
            params[param] = trial.suggest_categorical(param, values)
    return params

def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters_model_selection: Dict[str, Any],
    parameters_grid: Dict[str, Any],
):
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
            logger.info(f"Treinando baseline: {model_name}")
            y_train_flat = np.ravel(y_train)
            model.fit(X_train, y_train_flat)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")
            baseline_results[model_name] = f1
            mlflow.log_metric("f1_macro", f1)
            print(f"Baseline {model_name}: F1 macro = {f1:.4f}")

    best_model_name = max(baseline_results, key=baseline_results.get)
    best_baseline_score = baseline_results[best_model_name]
    logger.info(f"Melhor baseline: {best_model_name} (F1 macro = {best_baseline_score:.4f})")

    param_grid = parameters_grid["hyperparameters"][best_model_name]
    model_class = get_model_class(best_model_name)

    def objective(trial):
        params = get_optuna_search_space(trial, param_grid)
        model = model_class(**params)
        model.fit(X_train, np.ravel(y_train))
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        return f1

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

    with mlflow.start_run(run_name=f"Model_Selection: {phase.upper()}_champion", experiment_id=experiment_id, nested=True) as run:
        mlflow.sklearn.log_model(
            champion_model,
            artifact_path="model",
            registered_model_name=model_registry_name,
            input_example=X_test[:2],
        )
        mlflow.set_tag("algorithm", best_model_name)
        mlflow.set_tag("best_score", np.round(champion_score, 4))
        mlflow.set_tag("phase", phase)
        mlflow.set_tag("autor", "Rodrigo")

        # Setar tags no MODEL REGISTRY (não apenas no run)
        client = MlflowClient()
        client.set_registered_model_tag(model_registry_name, "algorithm", best_model_name)
        client.set_registered_model_tag(model_registry_name, "best_score", str(np.round(champion_score, 4)))
        client.set_registered_model_tag(model_registry_name, "phase", phase)
        client.set_registered_model_tag(model_registry_name, "autor", "Rodrigo")

        print(f"Champion model: {best_model_name} - F1 macro: {champion_score:.4f} ({phase})")

    return champion_model












# import pandas as pd
# import numpy as np
# import logging
# import mlflow
# import optuna
# import warnings
# from typing import Dict, Any

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score

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
#         elif isinstance(values[0], int):
#             if len(values) == 1:
#                 params[param] = values[0]
#             else:
#                 params[param] = trial.suggest_categorical(param, values)
#         elif isinstance(values[0], float):
#             if len(values) == 1:
#                 params[param] = values[0]
#             else:
#                 params[param] = trial.suggest_categorical(param, values)
#         else:
#             params[param] = trial.suggest_categorical(param, values)
#     return params

# def model_selection(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame,
#     parameters_model_selection: Dict[str, Any],
#     parameters_grid: Dict[str, Any],
# ):
#     experiment_name = parameters_model_selection["mlflow_experiment_name"]
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

#     # Treinar todos os modelos baseline
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

#     # Escolher o melhor modelo baseline
#     best_model_name = max(baseline_results, key=baseline_results.get)
#     best_baseline_score = baseline_results[best_model_name]
#     logger.info(f"Melhor baseline: {best_model_name} (F1 macro = {best_baseline_score:.4f})")

#     # Otimização com Optuna para o melhor modelo
#     param_grid = parameters_grid["hyperparameters"][best_model_name]
#     model_class = get_model_class(best_model_name)

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

#     # Treinar o melhor modelo com os melhores parâmetros
#     best_model = model_class(**best_optuna_params)
#     best_model.fit(X_train, np.ravel(y_train))

#     # Guardar o campeão: baseline ou optuna?
#     if best_optuna_score > best_baseline_score:
#         champion_model = best_model
#         champion_score = best_optuna_score
#         phase = "champion_optuna"
#     else:
#         champion_model = models_dict[best_model_name]
#         champion_score = best_baseline_score
#         phase = "champion_baseline"

#     # MLflow run para o campeão
#     with mlflow.start_run(run_name=f"Model_Selection: {phase.upper()}_champion", experiment_id=experiment_id, nested=True):
#         mlflow.sklearn.log_model(
#             champion_model,
#             artifact_path="model",
#             registered_model_name=parameters_model_selection["model_registry_name"],
#             input_example=X_test[:2],  # Opcional, só para registo
#         )
#         mlflow.set_tag("algorithm", best_model_name)
#         mlflow.set_tag("best_score", np.round(champion_score, 4))
#         mlflow.set_tag("phase", phase)
#         mlflow.set_tag("autor", "Rodrigo")
#         print(f"Champion model: {best_model_name} - F1 macro: {champion_score:.4f} ({phase})")

#     return champion_model






























# import logging
# import numpy as np
# import pandas as pd
# from typing import Dict, Any
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
# import optuna
# import mlflow
# import mlflow.sklearn

# logger = logging.getLogger(__name__)

# def _get_or_create_experiment_id(experiment_name: str) -> str:
#     exp = mlflow.get_experiment_by_name(experiment_name)
#     if exp is None:
#         logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
#         return mlflow.create_experiment(experiment_name)
#     return exp.experiment_id

# def define_models(seed):
#     return {
#         "RandomForestClassifier": RandomForestClassifier(random_state=seed),
#         "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
#         "XGBClassifier": XGBClassifier(eval_metric="logloss", random_state=seed, use_label_encoder=False),
#         "LogisticRegression": LogisticRegression(max_iter=200),
#         "KNeighborsClassifier": KNeighborsClassifier()
#     }

# def scale_features(X_train, X_test):
#     scaler = StandardScaler()
#     return scaler.fit_transform(X_train), scaler.transform(X_test)

# def train_base_models(models_dict, X_train, y_train, X_test, y_test, experiment_id):
#     results = {}
#     best_f1 = -np.inf
#     best_model = None
#     best_name = None
#     baseline_metrics = {}

#     for name, model in models_dict.items():
#         with mlflow.start_run(experiment_id=experiment_id, run_name=f"{name}_baseline", nested=True):
#             mlflow.sklearn.autolog()
#             model.fit(X_train, np.ravel(y_train))
#             preds = model.predict(X_test)
#             f1 = f1_score(y_test, preds, average="macro")
#             recall = recall_score(y_test, preds, average="macro")
#             precision = precision_score(y_test, preds, average="macro")
#             accuracy = accuracy_score(y_test, preds)
#             mlflow.log_metric("f1_macro", f1)
#             mlflow.log_metric("recall_macro", recall)
#             mlflow.log_metric("precision_macro", precision)
#             mlflow.log_metric("accuracy", accuracy)
#             results[name] = (model, f1)
#             baseline_metrics[name] = {
#                 "f1_macro": f1,
#                 "recall_macro": recall,
#                 "precision_macro": precision,
#                 "accuracy": accuracy
#             }
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_model = model
#                 best_name = name
#     return best_model, best_name, best_f1, baseline_metrics

# def optuna_tuning(model_name, X_train, y_train, X_test, y_test, grid_params, seed, experiment_id, n_trials=10):
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

#     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
#     with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_optuna", nested=True):
#         study.optimize(objective, n_trials=n_trials)
#         best_params = study.best_trial.params

#         # Train final model with best params
#         if model_name == "RandomForestClassifier":
#             model = RandomForestClassifier(**best_params)
#         elif model_name == "GradientBoostingClassifier":
#             model = GradientBoostingClassifier(**best_params)
#         elif model_name == "XGBClassifier":
#             model = XGBClassifier(**best_params)
#         elif model_name == "LogisticRegression":
#             model = LogisticRegression(**best_params)
#         elif model_name == "KNeighborsClassifier":
#             model = KNeighborsClassifier(**best_params)

#         model.fit(X_train, np.ravel(y_train))
#         preds = model.predict(X_test)
#         f1 = f1_score(y_test, preds, average="macro")
#         recall = recall_score(y_test, preds, average="macro")
#         precision = precision_score(y_test, preds, average="macro")
#         accuracy = accuracy_score(y_test, preds)
#         mlflow.log_metric("f1_macro", f1)
#         mlflow.log_metric("recall_macro", recall)
#         mlflow.log_metric("precision_macro", precision)
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_params(best_params)
#         metrics = {
#             "f1_macro": f1,
#             "recall_macro": recall,
#             "precision_macro": precision,
#             "accuracy": accuracy
#         }
#     return model, f1, best_params, metrics

# def model_selection(
#     X_train_preprocessed: pd.DataFrame,
#     X_test_preprocessed: pd.DataFrame,
#     y_train_encoded: pd.Series,
#     y_test_encoded: pd.Series,
#     parameters_model_selection: Dict[str, Any],
#     parameters_grid: Dict[str, Any],
# ):
#     # PARAMS
#     seed = parameters_model_selection.get("random_state", 42)
#     experiment_name = parameters_model_selection["mlflow_experiment_name"]
#     model_registry_name = parameters_model_selection["model_registry_name"]
#     run_name = parameters_model_selection.get("run_name", "Model_Selection")

#     experiment_id = _get_or_create_experiment_id(experiment_name)

#     models_dict = define_models(seed)
#     X_train = X_train_preprocessed
#     X_test = X_test_preprocessed
#     X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

#     # Baseline runs
#     base_model, best_name, base_f1, baseline_metrics = train_base_models(
#         models_dict, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, experiment_id
#     )

#     # Tuning run (Optuna)
#     tuned_model, tuned_f1, best_params, tuned_metrics = optuna_tuning(
#         best_name, X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded,
#         parameters_grid["hyperparameters"][best_name], seed, experiment_id, n_trials=15
#     )

#     # Champion selection
#     if tuned_f1 > base_f1:
#         champion = tuned_model
#         champion_metrics = tuned_metrics
#         description = "Champion model selected via Optuna tuning."
#     else:
#         champion = base_model
#         champion_metrics = baseline_metrics[best_name]
#         description = "Champion model is the best baseline."

#     # Registar modelo campeão no registry do MLflow (fora dos nested runs)
#     with mlflow.start_run(experiment_id=experiment_id, run_name=f"{run_name}_champion", nested=True):
#         mlflow.sklearn.log_model(champion, artifact_path="model")
#         for k, v in champion_metrics.items():
#             mlflow.log_metric(k, v)
#         mlflow.set_tags({
#             "champion": True,
#             "model_registry_name": model_registry_name,
#             "description": description
#         })
#         run = mlflow.active_run()
#         model_uri = f"runs:/{run.info.run_id}/model"
#         # Regista no model registry
#         mlflow.register_model(model_uri, model_registry_name)

#     # Guardar outputs para o Kedro catalog (modelo campeão, metrics, etc.)
#     return champion, pd.DataFrame([champion_metrics])

