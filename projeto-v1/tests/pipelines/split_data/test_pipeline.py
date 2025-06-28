"""
This is a boilerplate test file for pipeline 'split_data'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
from src.projeto_v1.pipelines.split_data.nodes import split_data  
from src.projeto_v1.pipelines.split_data.pipeline import create_pipeline

def test_split_data():
    # Create dummy DataFrame with features and target column 'y'
    df = pd.DataFrame({
        "age": [23, 45, 61, 30, 40, 50, 35, 28, 55, 60],
        "campaign": [1, 2, 3, 4, 5, 6, 2, 3, 1, 4],
        "previous": [0, 1, 2, 0, 1, 3, 0, 1, 2, 0],
        "cons.price.idx": [92.893, 93.200, 93.994, 92.5, 93.0, 94.0, 92.8, 93.3, 93.5, 92.6],
        "default": [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        "loan": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    # Call the split_data function
    X_train, X_test, y_train, y_test, columns = split_data(df)

    # Check columns do not contain the target 'y'
    assert "y" not in columns, "'y' should not be in feature columns"
    assert set(columns) == set(df.columns) - {"y"}, "Returned columns do not match expected features"

    # Check split sizes (80% train, 20% test)
    total_len = len(df)
    expected_train_len = int(total_len * 0.8)
    expected_test_len = total_len - expected_train_len
    assert len(X_train) == expected_train_len, "X_train size is incorrect"
    assert len(X_test) == expected_test_len, "X_test size is incorrect"
    assert len(y_train) == expected_train_len, "y_train size is incorrect"
    assert len(y_test) == expected_test_len, "y_test size is incorrect"

    # Check indices alignment
    assert all(X_train.index == y_train.index), "X_train and y_train indices do not match"
    assert all(X_test.index == y_test.index), "X_test and y_test indices do not match"

    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "split"