"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline, pipeline
from typing import Dict
from bank_mlops25.pipelines import (
    data_tests,
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_tests_pipeline = data_tests.create_pipeline()
    
    return {
        "data_tests": data_tests_pipeline,

    }