"""
This is a boilerplate pipeline 'basic_pre_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_train_test, encode_targets

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_train_test,
                inputs=["X_train_data", "X_test_data"],
                outputs=["X_train_preprocessed", "X_test_preprocessed"],
                name="preprocess_data_node",
            ),
            node(
                func=encode_targets,
                inputs=["y_train_data", "y_test_data"],
                outputs=["y_train_encoded", "y_test_encoded"],
                name="encode_targets_node"
            )
        ]
    )
