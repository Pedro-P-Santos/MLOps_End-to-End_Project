"""
This is a boilerplate test file for pipeline 'feature_selection'
generated using Kedro 0.19.5.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
"""
This is a boilerplate test file for pipeline 'feature_selection'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.projeto_v1.pipelines.feature_selection.nodes import run_all_feature_selection_methods
from src.projeto_v1.pipelines.feature_selection.pipeline import create_pipeline


def test_run_all_feature_selection_methods_with_project_config():
    # Load feature selection config from parameters.yml
    config_path = Path("conf/base/parameters.yml")
    with config_path.open("r") as file:
        parameters = yaml.safe_load(file)
    config = parameters["parameters_feature_selection"]

    # Construct dummy DataFrame to cover all feature selection cases
    df = pd.DataFrame({
        "feature1": np.random.randint(0, 10, 100),        # high variance
        "feature2": np.ones(100),                         # low variance
        "feature3": np.random.randint(0, 5, 100),
        "feature4": np.random.randint(0, 2, 100),         # binary
        "feature5": np.linspace(1, 100, 100),             # increasing
        "feature6": np.abs(np.random.randn(100)),         # non-negative
        "feature7": np.random.randint(0, 3, 100),
        "contact": np.random.choice(["cellular", "telephone"], 100),
        "month": np.random.choice(["may", "jun"], 100),
        "duration": np.random.randint(50, 1000, 100),
        "contact_day_priority": np.random.randint(1, 4, 100),
        "contact_month_quarter": np.random.randint(1, 5, 100),
    })

    y = pd.DataFrame({"target": np.random.choice([0, 1], 100)})

    # Run the feature selection node
    rfe, chi2, boruta, var_thresh, final = run_all_feature_selection_methods(df, y, config)

    # --- RFE ---
    assert isinstance(rfe, list), "RFE output should be a list"
    for feature in rfe:
        assert isinstance(feature, str), "RFE features must be strings"

    # --- Chi² ---
    assert isinstance(chi2, list), "Chi2 output should be a list"
    for feature in chi2:
        assert isinstance(feature, str), "Chi2 features must be strings"
        assert df[feature].min() >= 0, f"Chi2 feature {feature} must have non-negative values"

    # --- Boruta ---
    assert isinstance(boruta, list), "Boruta output should be a list"
    for feature in boruta:
        assert isinstance(feature, str), "Boruta features must be strings"

    # --- Variance Threshold ---
    assert isinstance(var_thresh, list), "Variance Threshold output should be a list"
    assert "feature2" not in var_thresh, "'feature2' should be removed due to low variance"

    # --- Final Features ---
    assert isinstance(final, list), "Final selected features must be a list"
    assert len(final) > 0, "Final feature list should not be empty"

    # --- Benchmark Check ---
    for benchmark_col in ["contact", "month", "duration", "contact_day_priority", "contact_month_quarter"]:
        all_selected = rfe + chi2 + boruta + var_thresh
        assert benchmark_col not in all_selected, f"{benchmark_col} should be removed as a benchmark feature"

    # --- Pipeline Metadata ---
    pipeline = create_pipeline()
    assert pipeline.nodes, "Pipeline should contain at least one node"
    assert isinstance(pipeline.nodes[0].name, str)
