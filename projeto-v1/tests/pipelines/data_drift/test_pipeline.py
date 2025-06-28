"""
This is a boilerplate test file for the 'data_drift' pipeline
generated using Kedro 0.19.5.

Kedro recommends using the `pytest` framework. For more information, visit:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import numpy as np
import pytest
import sys
from unittest import mock

# Mock evidently package to avoid import errors during test collection
sys.modules['evidently'] = mock.MagicMock()
sys.modules['evidently.report'] = mock.MagicMock()
sys.modules['evidently.metric_preset'] = mock.MagicMock()

from src.projeto_v1.pipelines.data_drift.nodes import run_all_drift_checks
from src.projeto_v1.pipelines.data_drift.pipeline import create_pipeline


def test_run_all_drift_checks():
    np.random.seed(42)
    n_samples = 100

    # Generate synthetic train and serving datasets
    train_df = pd.DataFrame({
        "age": np.random.normal(40, 5, n_samples),
        "emp.var.rate": np.random.normal(-1.5, 0.5, n_samples),
        "poutcome_success": np.random.rand(n_samples),
        "nr.employed": np.random.normal(5000, 100, n_samples)
    })
    serving_df = train_df.copy()
    y_train = pd.Series(np.random.choice([0, 1], size=n_samples))
    y_serving = pd.Series(np.random.choice([0, 1], size=n_samples))

    feature_columns = ["age", "emp.var.rate", "poutcome_success", "nr.employed"]

    with mock.patch("evidently.report.Report.save_html") as mock_save_html:
        results = run_all_drift_checks(
            train_df=train_df,
            serving_df=serving_df,
            y_train=y_train,
            y_serving=y_serving,
            feature_columns=feature_columns,
            psi_report_path_serving="dummy_serving_path.html",
            psi_report_path_drifted="dummy_drifted_path.html"
        )

    # Basic result structure check
    assert isinstance(results, dict)
    assert "serving" in results
    assert "drifted" in results

    # Check KS and PCA results only include "age"
    for section_key in ["serving", "drifted"]:
        section = results[section_key]

        # KS
        ks_result = section.get("ks", {})
        assert isinstance(ks_result, dict)
        assert list(ks_result.keys()) == ["age"]
        assert "statistic" in ks_result["age"]
        assert "p_value" in ks_result["age"]
        assert "interpretation" in ks_result["age"]

        # PCA
        pca_result = section.get("pca", {})
        assert isinstance(pca_result, dict)
        assert "ks_p_value" in pca_result
        assert "ks_statistic" in pca_result
        assert "interpretation" in pca_result

        # PSI
        psi_result = section.get("psi", {})
        assert isinstance(psi_result, dict)
        assert psi_result.get("status") == "Report saved"
        assert psi_result.get("path") is not None

    # Check Fisher test
    fisher_result = results["serving"].get("fisher_target", {})
    assert isinstance(fisher_result, dict)
    assert "p_value" in fisher_result
    assert "odds_ratio" in fisher_result
    assert isinstance(fisher_result.get("contingency_table"), pd.DataFrame)

    # Check that Evidently's HTML report was saved twice
    assert mock_save_html.call_count == 2


def test_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline.nodes
    assert pipeline.nodes[0].name == "drift_detection_node"
