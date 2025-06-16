from kedro.pipeline import Pipeline, node
from .nodes import build_custom_expectations_for_rawdata

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_custom_expectations_for_rawdata,
                inputs="bank_data",
                outputs="validated_data",  
                name="data_expectations_raw_node"
            ),
        ]
    )