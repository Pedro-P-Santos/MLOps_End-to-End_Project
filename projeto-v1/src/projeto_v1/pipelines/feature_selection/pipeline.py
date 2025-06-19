"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import run_all_feature_selection_methods

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_all_feature_selection_methods,
            inputs=["X_train_preprocessed", "y_train_encoded", "params:parameters_feature_selection"],
            outputs={
                "rfe_selected_features": "rfe_selected_features",
                "chi2_selected_features": "chi2_selected_features",
                "boruta_selected_features": "boruta_selected_features",
                "variance_threshold_selected_features": "variance_threshold_selected_features",
            },
            name="run_all_feature_selection_methods_node",
        ),
    ])
