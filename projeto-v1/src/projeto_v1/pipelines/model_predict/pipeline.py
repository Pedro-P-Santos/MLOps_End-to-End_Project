"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=["X_test_scaled", "trained_model", "final_selected_features"],
                outputs=["predictions_df", "predictions_stats"],
                name="train",
            ),
        ]
    )
