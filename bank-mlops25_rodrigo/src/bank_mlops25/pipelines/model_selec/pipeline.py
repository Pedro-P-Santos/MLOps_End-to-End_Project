"""
This is a boilerplate pipeline 'model_selec'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_preprocessed", "X_test_preprocessed", "y_train_encoded", "y_test_encoded"],
                outputs=["best_model", "model_scores"],
                name="model_selection_node",
            ),
        ]
    )


