"""
This is a boilerplate test file for pipeline 'split_data'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.split_data.nodes import split_data

X_train, X_test, y_train, y_test, columns = split_data(pd.read_csv(Path("03_primary/data_engineered.csv")))

def test_split_data_shapes_match():
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train rows"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in X_test and y_test rows"

def test_split_data_no_nulls():
    for name, df in [("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)]:
        assert not df.isnull().values.any(), f"{name} contains nulls"

def test_target_is_binary():
    assert y_train.nunique().values[0] == 2, "y_train is not binary"
    assert y_test.nunique().values[0] == 2, "y_test is not binary"
