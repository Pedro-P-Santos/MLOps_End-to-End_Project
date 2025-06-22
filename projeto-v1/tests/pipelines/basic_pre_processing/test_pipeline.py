"""
This is a boilerplate test file for pipeline 'basic_pre_processing'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
"""
Testes para pipeline 'basic_pre_processing'
"""

import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.basic_pre_processing.nodes import preprocess_data


X_train = pd.read_csv("data/05_model_input/X_train.csv")
X_test = pd.read_csv("data/05_model_input/X_test.csv")

def test_preprocess_data_returns_two_dataframes():
    X_train_proc, X_test_proc = preprocess_data(X_train, X_test)
    assert isinstance(X_train_proc, pd.DataFrame)
    assert isinstance(X_test_proc, pd.DataFrame)

def test_preprocessed_data_same_shape():
    X_train_proc, X_test_proc = preprocess_data(X_train, X_test)
    assert X_train_proc.shape[1] == X_test_proc.shape[1], "X_train and X_test processed should have the same number of columns"

def test_no_nulls_in_preprocessed_data():
    X_train_proc, X_test_proc = preprocess_data(X_train, X_test)
    assert not X_train_proc.isnull().values.any(), "X_train has missing values"
    assert not X_test_proc.isnull().values.any(), "X_test has missing values"
