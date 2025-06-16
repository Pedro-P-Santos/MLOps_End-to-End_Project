"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from projeto_v1.pipelines import ingestion
from projeto_v1.pipelines import data_expectations


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    ingestion_pipeline = ingestion.create_pipeline()
    data_expectations_pipeline = data_expectations.create_pipeline()

    return {
        # "__default__": ingestion_pipeline,
        "ingestion": ingestion_pipeline,
        "data_expectations": data_expectations_pipeline
    }

