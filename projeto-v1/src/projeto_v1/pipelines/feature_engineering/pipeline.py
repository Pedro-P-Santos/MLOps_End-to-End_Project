"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.5
"""
from .nodes import run_feature_engineering
from kedro.pipeline import node, Pipeline, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_feature_engineering,
                inputs="cleaned_data",
                outputs="data_engineered",  
                name="data_engineering_node"
            ),
        ]
    )