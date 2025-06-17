"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.12
"""


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def preprocess_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Colunas numéricas e categóricas
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pipelines de transformação
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer aplica o pré-processamento às colunas corretas
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Fit + transformação
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Nomes das colunas depois da codificação
    encoded_cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols)
    final_cols = num_cols + list(encoded_cat_cols)

    # Criar DataFrames com os mesmos índices e colunas corretas
    X_train_df = pd.DataFrame(X_train_proc, columns=final_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=final_cols, index=X_test.index)

    return X_train_df, X_test_df



def encode_targets(y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_train_encoded = y_train.replace({"yes": 1, "no": 0})
    y_test_encoded = y_test.replace({"yes": 1, "no": 0})
    return y_train_encoded, y_test_encoded
