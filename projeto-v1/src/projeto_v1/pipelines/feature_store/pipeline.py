"""
This is a boilerplate pipeline 'feature_store'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import upload_to_feature_store

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=upload_to_feature_store,
            inputs="data_engineered",
            outputs=None,
            name="upload_features_to_store",
        )
    ])