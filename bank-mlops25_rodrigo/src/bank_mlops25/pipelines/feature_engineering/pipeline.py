"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_engineering  

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_engineering,
                inputs="bank_data",
                outputs="bank_data_plus_features",
                name="create_features",
            )
        ]
    )

