"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=model_train,
            inputs=[
                "X_train_scaled",
                "X_test_scaled",
                "y_train_encoded",
                "y_test_encoded",
                "parameters_model_train",
                "final_selected_features"
            ],
            outputs=["champion_model", "champion_results"],
            name="train_champion_model"
        )
    ])

