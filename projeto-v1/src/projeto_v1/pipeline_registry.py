"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


from projeto_v1.pipelines import (
    ingestion,
    data_expectations,
    feature_engineering,
    # feature_store,
    split_data,
    basic_pre_processing,
    feature_selection,
    model_development_selection,
    model_train
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    ingestion_pipeline = ingestion.create_pipeline()
    data_expectations_pipeline = data_expectations.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    # feature_store_pipeline = feature_store.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    basic_pre_processing_pipeline = basic_pre_processing.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_development_selection_pipeline=model_development_selection.create_pipeline()
    model_train_pipeline = model_train.create_pipeline()

    return {
        "__default__": (
            ingestion_pipeline
            + data_expectations_pipeline
            + feature_engineering_pipeline
            + split_data_pipeline
            + basic_pre_processing_pipeline
            + feature_selection_pipeline
            + model_development_selection_pipeline
        ),
        "ingestion": ingestion_pipeline,
        "data_expectations": data_expectations_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        # "feature_store": feature_store_pipeline,
        "split_data": split_data_pipeline,
        "basic_pre_processing": basic_pre_processing_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_development_selection":model_development_selection_pipeline,
        "model_train": model_train_pipeline
    }

