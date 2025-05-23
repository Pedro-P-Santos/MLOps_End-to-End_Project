from kedro.pipeline import Pipeline, node
from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ingestion,
                inputs="bank_data",        # This comes from catalog.yml
                outputs="ingested_data",        # What you save next
                name="ingestion_node"
            )
        ]
    )