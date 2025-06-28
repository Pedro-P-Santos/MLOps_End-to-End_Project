from kedro.pipeline import Pipeline, node
from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ingestion,
                inputs="bank_data",        
                outputs="ingested_data",        
                name="ingestion_node"
            )
        ]
    )