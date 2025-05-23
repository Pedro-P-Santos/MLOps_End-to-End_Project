"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline, pipeline
from typing import Dict
from bank_mlops25.pipelines import (
    ingestion as data_ingestion,
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion_pipeline = data_ingestion.create_pipeline()
    
    return {
        "ingestion": ingestion_pipeline,

    }