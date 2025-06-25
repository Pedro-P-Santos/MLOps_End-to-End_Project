"""
This is a boilerplate pipeline 'feature_selection'
generated using Kedro 0.19.5
"""
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is not installed. Please install it using 'pip install xgboost'.")


try: 
    from boruta import BorutaPy
except ImportError:
    raise ImportError("BorutaPy is not installed. Please install it using 'pip install boruta'.")

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_all_feature_selection_methods(X_train_preprocessed: pd.DataFrame, y_train_encoded: pd.DataFrame, params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Orchestrates the execution of multiple feature selection techniques and returns selected feature names.
    """
    assert isinstance(X_train_preprocessed, pd.DataFrame), "X_train_preprocessed must be a pandas DataFrame"
    assert isinstance(y_train_encoded, pd.DataFrame), "y_train_encoded must be a pandas DataFrame"

    # Step 1: Remove known benchmark features before any selection
    X_cleaned = remove_benchmark_features(X_train_preprocessed)

    # Run each method individually and return in order
    rfe_selected_features = feature_selection_rfe(X_cleaned, y_train_encoded, params) \
        if "model_params_xgb" in params else []
    
    chi2_selected_features = feature_selection_chi2(X_cleaned, y_train_encoded, params) \
        if "k_features_chi2" in params else []
    
    boruta_selected_features = feature_selection_boruta(X_cleaned, y_train_encoded, params) \
        if "model_params_boruta" in params else []
    
    variance_threshold_selected_features = feature_selection_variance_threshold(X_cleaned, y_train_encoded, params) \
        if "variance_threshold" in params else []

    ### FROM NOTEBOOK ANALYSIS!!
    final_selected_features = ['is_student_or_retired',
        'cpi_top_value_92.893',
        'poutcome_success',
        'cci_top_value',
        'marital_single',
        'previous_bin_1',
        'age',
        'age_binned_quantile_0.0',
        'education_mapped_basic_education',
        'poutcome_nonexistent',
        'middle_aged_housing_loan',
        'previous_bin_0',
        'previous',
        'euribor_bin_high',
        'emp.var.rate',
        'age_binned_quantile_4.0',
        'cpi_top_value_93.994',
        'cpi_top_value_93.918',
        'euribor_bin_very_high',
        'loan_risk_score',
        'nr.employed',
        'education_mapped_higher_education',
        'cons.conf.idx',
        'pdays',
        'successful_prev_contact',
        'senior_loan',
        'age_binned_quantile_2.0',
        'senior_housing_loan',
        'cci_above_75th',
        'young_housing_loan',
        'euribor_bin_very_low',
        'young_loan',
        'middle_aged_loan',
        'contacted_before',
        'default',
        'euribor_bin_low',
        'age_binned_quantile_3.0',
        'campaign',
        'cpi_above_75th',
        'previous_bin_3',
        'poutcome_failure',
        'marital_married',
        'euribor3m',
        'economic_pressure_index',
        'previous_bin_2',
        'cons.price.idx',
        'cpi_top_value_other']

    return (
        rfe_selected_features,
        chi2_selected_features,
        boruta_selected_features,
        variance_threshold_selected_features,
        final_selected_features
    )




def feature_selection_rfe(X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]) -> List[str]:
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting RFE with XGBoost over all feature sizes")

    model_params = params["model_params_xgb"]
    y = np.ravel(y)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    total_features = X.shape[1]
    feature_names = X.columns

    best_score = 0.0
    best_features = []
    best_n_features = 0

    for n in range(1, total_features + 1):
        logger.info(f"Testing RFE with {n} features")

        model = XGBClassifier(**model_params, verbosity=0)
        rfe = RFE(estimator=model, n_features_to_select=n, step=0.2)
        rfe.fit(X_train_split, y_train_split)

        rfe_selected_features = feature_names[rfe.support_].tolist()
        assert isinstance(rfe_selected_features, list), "Selected features should be a list"

        model.fit(X_train_split[rfe_selected_features], y_train_split)
        y_train_pred = model.predict(X_train_split[rfe_selected_features])
        assert isinstance(y_train_pred, np.ndarray), "y_train_pred should be a numpy array"
        y_val_pred = model.predict(X_val_split[rfe_selected_features])
        assert isinstance(y_val_pred, np.ndarray), "y_val_pred should be a numpy array"

        train_f1 = f1_score(y_train_split, y_train_pred, average="macro")
        val_f1 = f1_score(y_val_split, y_val_pred, average="macro")
        gap = abs(train_f1 - val_f1)

        logger.info(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Gap: {gap:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            best_features_rfe = rfe_selected_features
            best_n_features = n
            logger.info(f"✅ New best model with {n} features")

    logger.info(f"🎯 Best validation F1: {best_score:.4f} using {best_n_features} features")
    logger.info(f"Selected features: {best_features_rfe}")
    assert isinstance(best_features_rfe, list), "Selected features should be a list"

    return best_features_rfe


def feature_selection_boruta(X: pd.DataFrame, y: pd.Series, params: Dict) -> List[str]:
    """
    Placeholder for Boruta feature selection implementation.
    """
    logger.info("Starting Boruta feature selection")

    model_params = params["model_params_boruta"]
    max_iter = params["max_iter_boruta"]

    xgb = XGBClassifier(**model_params)

    boruta_selector = BorutaPy(
        estimator=xgb,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=42
    )

    boruta_selector.fit(X.values, y.values.ravel())

    # Log support and ranking
    logger.info("------  Support and Ranking for Each Feature ------")
    for i in range(len(boruta_selector.support_)):
        feature = X.columns[i]
        support = boruta_selector.support_[i]
        rank = boruta_selector.ranking_[i]
        if support:
            logger.info(f" Passes the test: {feature} - Ranking: {rank}")
        else:
            logger.info(f" Doesn't pass the test: {feature} - Ranking: {rank}")

    selected_features_boruta = X.columns[boruta_selector.support_].tolist()
    logger.info(f" Boruta selected {len(selected_features_boruta)} features: {selected_features_boruta}")

    assert isinstance(selected_features_boruta, list), "Selected features should be a list"
    return selected_features_boruta


def feature_selection_chi2(X: pd.DataFrame, y: pd.Series, params: Dict) -> List[str]:
    """
    Chi-squared feature selection using significance threshold instead of top-k.
    Selects all features with p-value below threshold.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Chi-squared feature selection")

    significance_threshold = params["chi2_p_value_threshold"]
    logger.info(f"Using Chi-squared with significance threshold: {significance_threshold}")

    # categorical_columns = [
    #     "job", "marital", "education", "contact", "month", "day_of_week",
    #     "poutcome", "age_binned_quantile", "campaign_bin", "previous_bin",
    #     "cpi_top_value", "cci_top_value", "euribor_bin", "education_mapped",
    #     "contact_day_priority", "contact_month_quarter", "marital_edu_combo"
    # ]

    # # valid_cols = [col for col in categorical_columns if col in X.columns]
    # X = X[valid_cols].copy()
    # assert not X.empty, "No matching categorical columns found in the dataset"

    # Chi2 requires non-negative features
    X = X.loc[:, (X.min() >= 0)]

    chi2_stats, p_values = chi2(X, y)
    chi2_results = pd.DataFrame({
        "feature": X.columns,
        "chi2_stat": chi2_stats,
        "p_value": p_values
    })

    selected = chi2_results[chi2_results["p_value"] < significance_threshold]
    selected_features_chi2 = selected["feature"].tolist()

    logger.info(f"✅ Chi-squared selected {len(selected_features_chi2)} features with p < {significance_threshold}: {selected_features_chi2}")

    assert isinstance(selected_features_chi2, list), "Selected features should be a list"
    return selected_features_chi2

def feature_selection_variance_threshold(X: pd.DataFrame, y: pd.Series, params: Dict) -> List[str]:
    """
    Placeholder for Variance Threshold feature selection implementation.
    """
    logger.info("Starting Variance Threshold feature selection")

    threshold = params["variance_threshold"]
    
    # --- Variance Threshold Feature Selection ---
    selector = VarianceThreshold(threshold=threshold)
    logger.info(f"Using VarianceThreshold with threshold: {threshold}")
    selector.fit(X)

    selected_features_var = X.columns[selector.get_support()].tolist()
    removed_features_var = X.columns[~selector.get_support()].tolist()

    logger.info(f" Selected {len(selected_features_var)} features with variance above {threshold}")
    if removed_features_var:
        logger.info(f" Removed {len(removed_features_var)} low-variance features: {removed_features_var}")

    return selected_features_var

def remove_benchmark_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes benchmark features from the DataFrame. And those derived from them.
    This is useful to avoid data leakage in the model training process.
    """
    benchmark_features = [
        "contact", "month", "day_of_week", "duration",
        "contact_efficiency", "contact_month_num", "is_summer_contact",
        "contact_day_priority", "contact_month_quarter"
    ]

    logger.info(f"Removing benchmark features: {benchmark_features}")
    X_cleaned = X.drop(columns=benchmark_features, errors='ignore')
    assert all(col not in X_cleaned.columns for col in benchmark_features), "Some benchmark features were not removed correctly"


    logger.info(f"Remaining features after removal: {X_cleaned.columns.tolist()}")

    return X_cleaned