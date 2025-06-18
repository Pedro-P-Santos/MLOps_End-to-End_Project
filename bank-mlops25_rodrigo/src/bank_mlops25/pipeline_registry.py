"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline, pipeline
from typing import Dict
from bank_mlops25.pipelines import (
    data_tests,
    feature_engineering,
    feature_store,
    split_data,
    preprocess,
    model_selec
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_tests_pipeline = data_tests.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    feature_store_pipeline = feature_store.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_pipeline=preprocess.create_pipeline()
    model_selec_pipeliene = model_selec.create_pipeline()


    return {
        "data_tests": data_tests_pipeline,
        "split_data":split_data_pipeline,
        "preprocess": preprocess_pipeline,
        "model_selec": model_selec_pipeliene,
        "feature_engineering": feature_engineering_pipeline,
        "feature_store": feature_store_pipeline,

    }