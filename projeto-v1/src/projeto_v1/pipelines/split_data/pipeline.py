"""
This is a boilerplate pipeline 'split_data'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["data_engineered"],
                outputs=["X_train_data", "X_test_data", "y_train_data", "y_test_data", "best_columns"],
                name="split",
            )
        ]
    )
