"""
This is a boilerplate pipeline 'model_development_selection'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_selection,
            inputs=[
                "X_train_preprocessed",
                "X_test_preprocessed",
                "y_train_encoded",
                "y_test_encoded",
                "final_selected_features",
                "parameters_model_selection",
                "parameters_grid"
            ],
            outputs=[
                "champion_model",
                "X_train_scaled",
                "X_test_scaled"
            ],
            name="model_selection_node"
        )
    ])

