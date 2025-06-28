"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.5
"""

## NOTE: very important remark that we do not have a true "serving" dataset.
## This is obviously not an ideal scenario, but we will use this pipeline to simulate a data drift scenario!
## X_train_preprocessed will act as our reference dataset vs X_test_preprocessed as "serving"
## Naturally, we dont expect drift, since these are from same original distribution
from scipy.stats import ks_2samp
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging
from scipy.stats import chi2_contingency, fisher_exact
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from .utils import (
    ks_test,
    psi_report,
    fisher_test_target,
    pca_reconstruction_error,
    create_drifted_df,
)



logger = logging.getLogger(__name__)

### WRAPPER:
def run_all_drift_checks(
    train_df,
    serving_df,
    y_train,
    y_serving,
    feature_columns,
    psi_report_path_serving="data/08_reporting/psi_report_serving.html",
    psi_report_path_drifted="data/08_reporting/psi_report_drifted.html"
):
####
    logger.info("Running drift detection: Train vs. Serving")

    ks_serving = ks_test(train_df, serving_df, feature_columns)
    psi_serving = psi_report(train_df, serving_df, report_path=psi_report_path_serving)
    fisher_target_serving = fisher_test_target(y_train, y_serving)
    pca_serving = pca_reconstruction_error(train_df, serving_df)
#####
    logger.info("Generating drifted dataset")
    drifted_df = create_drifted_df(train_df)
####
    logger.info("Running drift detection: Train vs. Drifted")

    ks_drifted = ks_test(train_df, drifted_df, feature_columns)
    psi_drifted = psi_report(train_df, drifted_df, report_path=psi_report_path_drifted)
    pca_drifted = pca_reconstruction_error(train_df, drifted_df)

    results = {
        "serving": {
            "ks": ks_serving,
            "psi": psi_serving,
            "fisher_target": fisher_target_serving,
            "pca": pca_serving
        },
        "drifted": {
            "ks": ks_drifted,
            "psi": psi_drifted,
            "pca": pca_drifted
        }
    }

    logger.info("Drift analysis complete")
    return results