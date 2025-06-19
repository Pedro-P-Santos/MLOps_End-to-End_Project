"""
This is a boilerplate pipeline 'basic_pre_processing'
generated using Kedro 0.19.12
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

def preprocess_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses training and test feature dataframes:
    - Handles categorical, binary, and numerical columns according to lists defined below.
    - Binary columns: Converts 'yes'/'no' to 1/0, replaces 'unknown' with np.nan for imputation.
    - Numerical columns: Imputes missing values with the mean.
    - Categorical columns: Imputes missing values with the mode, then applies OneHotEncoder.
    Returns processed train and test DataFrames with consistent columns.
    """

    categorical_features = [
        "job", "marital", "education", "contact", "month", "day_of_week",
        "poutcome", "age_binned_quantile", "campaign_bin", "previous_bin",
        "cpi_top_value", "cci_top_value", "euribor_bin", "education_mapped",
        "contact_day_priority", "contact_month_quarter", "marital_edu_combo"
    ]

    binary_features = [
        "default", "housing", "loan", "cpi_above_75th", "cci_above_75th",
        "young_housing_loan", "middle_aged_housing_loan", "senior_housing_loan",
        "young_loan", "middle_aged_loan", "senior_loan", "contacted_before",
        "is_summer_contact", "is_student_or_retired", "successful_prev_contact",
        "has_any_loan"
    ]

    numeric_features = [
        "age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
        "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
        "emp_rate_x_employed", "loan_risk_score", "contact_month_num",
        "contact_efficiency", "economic_pressure_index"
    ]

    # Sanity check: make sure every column is mapped, and nothing is missing or extra
    all_cols = set(categorical_features + binary_features + numeric_features)
    missing_cols = set(X_train.columns) - all_cols
    extra_cols = all_cols - set(X_train.columns)

    logging.info(f"Columns present in X_train: {list(X_train.columns)}")
    logging.info(f"All mapped columns: {list(all_cols)}")
    logging.info(f"Missing columns (should be empty): {missing_cols}")
    logging.info(f"Extra columns (should be empty or just in mapping): {extra_cols}")

    assert not missing_cols, f"Columns not mapped: {missing_cols}"
    if extra_cols:
        logging.warning(f"These columns are mapped but not present in X_train: {extra_cols}")

    # Treat 'unknown' in binary columns as np.nan, then map 'yes'/'no' to 1/0 and convert to float
    for col in binary_features:
        for df, name in zip([X_train, X_test], ["X_train", "X_test"]):
            n_unknown = (df[col] == "unknown").sum() if "unknown" in df[col].values else 0
            logging.info(f"[{name}] Replacing {n_unknown} 'unknown' values in '{col}' with np.nan")
            df[col] = df[col].replace("unknown", np.nan)
            if df[col].dtype == 'object':
                yes_count = (df[col] == 'yes').sum()
                no_count = (df[col] == 'no').sum()
                logging.info(f"[{name}] Replacing '{col}': {yes_count} 'yes', {no_count} 'no'")
                df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(float)
            else:
                df[col] = df[col].astype(float)
            nulls = df[col].isna().sum()
            logging.info(f"[{name}] '{col}' null count after replacements: {nulls}")

    # Define processing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features + binary_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # Fit on training, transform both train and test
    logging.info("Fitting and transforming pipelines...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Recreate final column names after encoding
    encoded_cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
    final_cols = numeric_features + binary_features + list(encoded_cat_cols)

    # Build processed DataFrames with correct columns and index
    X_train_df = pd.DataFrame(X_train_proc, columns=final_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=final_cols, index=X_test.index)

    logging.info(f"Processed X_train shape: {X_train_df.shape}")
    logging.info(f"Processed X_test shape: {X_test_df.shape}")
    logging.info(f"Columns in processed data: {list(X_train_df.columns)}")

    # Quick preview of first 3 rows
    logging.info("First 3 rows of processed X_train:\n%s", X_train_df.head(3))
    logging.info("First 3 rows of processed X_test:\n%s", X_test_df.head(3))

    return X_train_df, X_test_df

def encode_targets(y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mapping = {"yes": 1, "no": 0}
    logging.info(f"Encoding target values using mapping: {mapping}")
    y_train_enc = y_train.replace(mapping)
    y_test_enc = y_test.replace(mapping)

    # Convert DataFrame to Series if necessary
    if isinstance(y_train_enc, pd.DataFrame):
        y_train_enc = y_train_enc.squeeze()
    if isinstance(y_test_enc, pd.DataFrame):
        y_test_enc = y_test_enc.squeeze()

    logging.info("y_train unique values after encoding: %s", y_train_enc.unique())
    logging.info("y_test unique values after encoding: %s", y_test_enc.unique())
    return y_train_enc, y_test_enc
