from scipy.stats import ks_2samp, fisher_exact
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import logging

logger = logging.getLogger(__name__)


####### UTILS


def ks_test(train_df, serving_df, columns):
    columns = ["age"]
    ks_results = {}
    for col in columns:
        stat, p_value = ks_2samp(train_df[col], serving_df[col])

        logger.info("KS Test for feature '%s':", col)
        logger.info("Statistic: %.4f", stat)
        logger.info("p-value: %.4f", p_value)

        if p_value > 0.05:
            interpretation = "Fail to reject null hypothesis: No significant drift detected."
        else:
            interpretation = "Reject null hypothesis: Significant drift detected."

        logger.info("Interpretation: %s", interpretation)

        ks_results[col] = {
            "statistic": stat,
            "p_value": p_value,
            "interpretation": interpretation,
        }

    return ks_results


def psi_report(train_df, serving_df, report_path="data/08_reporting/psi_report.html"):
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=train_df, current_data=serving_df)
    report.save_html(report_path)
    
    return {"status": "Report saved", "path": report_path}


def fisher_test_binary_feature(train_df, serving_df, column_name):
    train_counts = train_df[column_name].value_counts().sort_index()
    serving_counts = serving_df[column_name].value_counts().sort_index()

    contingency_table = pd.DataFrame({
        "Train": train_counts,
        "Serving": serving_counts
    })

    table = contingency_table.values
    oddsratio, p_value = fisher_exact(table)

    logger.info(f"Fisher's Exact Test for '{column_name}'")
    logger.info("\n%s", contingency_table)
    logger.info(f"Odds Ratio: {oddsratio:.4f}")
    logger.info(f"p-value: {p_value:.4f}")

    if p_value > 0.05:
        interpretation = "Fail to reject null hypothesis: No significant drift detected."
    else:
        interpretation = "Reject null hypothesis: Significant drift detected."

    logger.info(f"Interpretation: {interpretation}")

    return {
        "p_value": p_value,
        "odds_ratio": oddsratio,
        "contingency_table": contingency_table
    }


def fisher_test_target(y_train,y_test):
    y_train_counts = y_train.value_counts().sort_index()
    y_test_counts = y_test.value_counts().sort_index()

    contingency_table = pd.DataFrame({
        "Train": y_train_counts,
        "Test": y_test_counts
    })

    table = contingency_table.values
    oddsratio, p_value = fisher_exact(table)

    logger.info(f"Fisher's Exact Test for Target'")
    logger.info("\n%s", contingency_table)
    logger.info(f"Odds Ratio: {oddsratio:.4f}")
    logger.info(f"p-value: {p_value:.4f}")

    if p_value > 0.05:
        interpretation = "Fail to reject null hypothesis: No significant drift detected."
    else:
        interpretation = "Reject null hypothesis: Significant drift detected."

    logger.info(f"Interpretation: {interpretation}")

    return {
        "p_value": p_value,
        "odds_ratio": oddsratio,
        "contingency_table": contingency_table
    }

def pca_reconstruction_error(train_df, serving_df, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(train_df)

    train_reconstructed = pca.inverse_transform(pca.transform(train_df))
    serving_reconstructed = pca.inverse_transform(pca.transform(serving_df))

    train_errors = np.mean((train_df - train_reconstructed) ** 2, axis=1)
    serving_errors = np.mean((serving_df - serving_reconstructed) ** 2, axis=1)

    ks_stat, p_value = ks_2samp(train_errors, serving_errors)

    logger.info("PCA Reconstruction Error KS Test:")
    logger.info(f"KS statistic: {ks_stat:.4f}")
    logger.info(f"p-value: {p_value:.4f}")

    if p_value > 0.05:
        interpretation = "Fail to reject null hypothesis: No significant drift detected."
    else:
        interpretation = "Reject null hypothesis: Significant drift detected."

    logger.info(f"Interpretation: {interpretation}")


    return {
        "ks_p_value": p_value,
        "ks_statistic": ks_stat,
        "interpretation": interpretation,
        "train_errors": train_errors,
        "serving_errors": serving_errors,
    }


def create_drifted_df(train):
    drifted = train.copy()

    drifted["age"] += train["age"].std()
    drifted["emp.var.rate"] += train["emp.var.rate"].std()

    # binary feature but consider feature values scaled
    drifted["poutcome_success"] = np.random.choice([ -0.184965, 5.406418], size=len(drifted))

    less_common_values = np.random.choice(train["nr.employed"].unique(), size=len(drifted))
    drifted["nr.employed"] = less_common_values 

    return drifted