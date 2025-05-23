"""
This is a boilerplate pipeline 'ingestion'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import validate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            inputs="bank_data",
            outputs="validated_data",
            func=validate_data,
            name="validate_data_node"

        )
    ])
