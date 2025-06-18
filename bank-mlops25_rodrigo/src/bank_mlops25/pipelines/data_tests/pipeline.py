"""
This is a boilerplate pipeline 'data_tests'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import validate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validate_data,
            inputs="bank_data",
            outputs="reporting_tests",
            name="validate_data_node"
        )
    ])
