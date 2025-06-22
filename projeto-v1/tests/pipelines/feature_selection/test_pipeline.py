"""
This is a test file for the pipeline 'feature_selection'.
It checks the functionality of run_all_feature_selection_methods using project parameters.
"""

import pandas as pd
import pytest
import yaml
from pathlib import Path
from kedro.config import TemplatedConfigLoader

from src.projeto_v1.pipelines.feature_selection.nodes import run_all_feature_selection_methods

# Load parameters from the Kedro YAML configuration
config_path = Path("conf/base")
config_loader = TemplatedConfigLoader(str(config_path))
params = config_loader.get("parameters_feature_selection")

# Load input data
X_train = pd.read_csv(Path("data/03_primary/X_train_preprocessed.csv"))
y_train = pd.read_csv(Path("data/03_primary/y_train_encoded.csv"))


def test_output_structure():
    """Test that the output is a tuple with four lists"""
    result = run_all_feature_selection_methods(X_train, y_train, params)
    assert isinstance(result, tuple), "Output should be a tuple"
    assert len(result) == 4, "Output tuple must have four elements"
    for feature_list in result:
        assert isinstance(feature_list, list), "Each output element should be a list"


def test_features_exist_in_input():
    """Test that all selected features exist in the original DataFrame"""
    result = run_all_feature_selection_methods(X_train, y_train, params)
    for method_features in result:
        for feature in method_features:
            assert feature in X_train.columns, f"Feature '{feature}' not found in input data"


def test_each_method_selects_features():
    """Test that each method returns at least one selected feature"""
    result = run_all_feature_selection_methods(X_train, y_train, params)
    method_names = ["RFE", "Chi2", "Boruta", "VarianceThreshold"]
    for method, features in zip(method_names, result):
        assert len(features) > 0, f"{method} did not return any features"
