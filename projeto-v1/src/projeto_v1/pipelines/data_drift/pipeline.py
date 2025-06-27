"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import run_all_drift_checks

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_all_drift_checks,
                inputs=dict(
                    train_df="X_train_scaled",
                    serving_df="X_test_scaled",
                    y_train="y_train_encoded",
                    y_serving="y_test_encoded",
                    feature_columns="params:feature_columns",
                    psi_report_path_serving="params:psi_report_path_serving",
                    psi_report_path_drifted="params:psi_report_path_drifted",
                ),
                outputs="results_drift",
                name="drift_detection_node",
            )
        ]
    )