# """
# This is a boilerplate pipeline 'model_selec'
# generated using Kedro 0.19.12
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, Tuple, Any
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from xgboost import XGBClassifier
# import mlflow
# import mlflow.sklearn
# import tempfile
# import os

# def model_selection(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame
# ) -> Tuple[Any, Dict[str, float]]:
#     """Treina vários modelos, seleciona o melhor e regista tudo no MLflow."""

#     models = {
#         "RandomForest": RandomForestClassifier(),
#         "LogisticRegression": LogisticRegression(max_iter=1000),
#         "GradientBoosting": GradientBoostingClassifier(),
#         "KNN": KNeighborsClassifier(),
#         "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
#     }

#     results = {}
#     best_model = None
#     best_score = -1
#     best_name = ""

#     with mlflow.start_run(nested=True):
#         mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

#         for name, model in models.items():
#             model.fit(X_train, y_train.values.ravel())
#             preds = model.predict(X_test)
#             acc = accuracy_score(y_test, preds)

#             results[name] = acc
#             mlflow.log_metric(f"{name}_accuracy", acc)

#             if acc > best_score:
#                 best_model = model
#                 best_score = acc
#                 best_name = name
#                 best_preds = preds

#         # Log do melhor modelo explicitamente
#         mlflow.set_tag("best_model", best_name)
#         mlflow.log_metric("best_accuracy", best_score)
#         mlflow.sklearn.log_model(best_model, artifact_path="best_model")

#         # Confusion matrix
#         cm = confusion_matrix(y_test, best_preds)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         disp.plot()

#         # Guardar imagem temporária e logar como artefacto
#         with tempfile.TemporaryDirectory() as tmpdir:
#             plot_path = os.path.join(tmpdir, "confusion_matrix.png")
#             plt.savefig(plot_path)
#             mlflow.log_artifact(plot_path, artifact_path="plots")
#             plt.close()

#     return best_model, results



# """
# This is a boilerplate pipeline 'model_selec'
# generated using Kedro 0.19.12
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, Tuple, Any
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from xgboost import XGBClassifier
# import mlflow
# import mlflow.sklearn
# import tempfile
# import os

# # 🔁 Ativar tracking automático ANTES do run
# mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

# def model_selection(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_train: pd.DataFrame,
#     y_test: pd.DataFrame
# ) -> Tuple[Any, Dict[str, float]]:
#     """Treina vários modelos, seleciona o melhor e regista tudo no MLflow."""

#     models = {
#         "RandomForest": RandomForestClassifier(),
#         "LogisticRegression": LogisticRegression(max_iter=1000),
#         "GradientBoosting": GradientBoostingClassifier(),
#         "KNN": KNeighborsClassifier(),
#         "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
#     }

#     results = {}
#     best_model = None
#     best_score = -1
#     best_name = ""

#     with mlflow.start_run(nested=True):
#         for name, model in models.items():
#             model.fit(X_train, y_train.values.ravel())
#             preds = model.predict(X_test)
#             acc = accuracy_score(y_test, preds)

#             results[name] = acc
#             mlflow.log_metric(f"{name}_accuracy", acc)

#             if acc > best_score:
#                 best_model = model
#                 best_score = acc
#                 best_name = name
#                 best_preds = preds

#         # Log do melhor modelo
#         mlflow.set_tag("best_model", best_name)
#         mlflow.log_metric("best_accuracy", best_score)
#         mlflow.sklearn.log_model(best_model, artifact_path="best_model")

#         # Confusion matrix
#         cm = confusion_matrix(y_test, best_preds)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         disp.plot()

#         # Guardar imagem e fazer log como artefacto
#         with tempfile.TemporaryDirectory() as tmpdir:
#             plot_path = os.path.join(tmpdir, "confusion_matrix.png")
#             plt.savefig(plot_path)
#             mlflow.log_artifact(plot_path, artifact_path="plots")
#             plt.close()

#     return best_model, results





import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[Any, Dict[str, float]]:
    """Treina 1 modelo simples e faz log mínimo no MLflow."""

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)

    return model, {"RandomForest": acc}
