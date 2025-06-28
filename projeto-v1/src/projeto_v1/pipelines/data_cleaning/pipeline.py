from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_bank_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_bank_data,
            inputs=["ingested_data", "params:data_cleaning_config"],
            outputs="cleaned_data",  
            name="clean_bank_data_node"       
        )
    ])
