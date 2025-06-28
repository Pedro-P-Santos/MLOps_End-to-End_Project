from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freqs_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freqs_[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_enc = X.copy()
        for col in X.columns:
            X_enc[col] = X[col].map(self.freqs_[col]).fillna(0)
        return X_enc.values  

def preprocess_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_high_cardinality = [
        "job", "education", "campaign_bin", "cci_top_value", "marital_edu_combo" #  "month",
    ]   
    categorical_low_cardinality = [
        "marital", "poutcome", "age_binned_quantile", "previous_bin",
        "cpi_top_value", "euribor_bin", "education_mapped" # "contact_day_priority" # "contact_month_quarter", "contact", "day_of_week"
    ]
    binary_features = [
        "default", "housing", "loan", "cpi_above_75th", "cci_above_75th",
        "young_housing_loan", "middle_aged_housing_loan", "senior_housing_loan",
        "young_loan", "middle_aged_loan", "senior_loan", "contacted_before",
        "is_student_or_retired", "successful_prev_contact",
        "has_any_loan" # "is_summer_contact", 
    ]
    numeric_features = [
        "age", "campaign", "pdays", "previous", "emp.var.rate",
        "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
        "emp_rate_x_employed", "loan_risk_score",
        "economic_pressure_index" # "duration", "contact_efficiency", "contact_month_num",
    ]

    features_to_drop = ["duration", "contact_efficiency", "contact", "month", "day_of_week"]
    features_to_drop = [f for f in features_to_drop if f in X_train.columns]
    X_train = X_train.drop(columns=features_to_drop)
    X_test = X_test.drop(columns=features_to_drop)

    # Binary columns: treat 'unknown', 'yes', 'no'
    for col in binary_features:
        for df in [X_train, X_test]:
            df[col] = df[col].replace("unknown", np.nan)
            if df[col].dtype == 'object':
                df[col] = df[col].replace({'yes': 1, 'no': 0}).astype(float)
            else:
                df[col] = df[col].astype(float)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq_encoder", FrequencyEncoder())
    ])
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features + binary_features),
        ("high_card", high_card_pipeline, categorical_high_cardinality),
        ("low_card", low_card_pipeline, categorical_low_cardinality)
    ])

    # Fit/transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Column names
    low_card_onehot = preprocessor.named_transformers_['low_card']['onehot']
    low_card_colnames = low_card_onehot.get_feature_names_out(categorical_low_cardinality)
    high_card_colnames = categorical_high_cardinality
    final_cols = numeric_features + binary_features + list(high_card_colnames) + list(low_card_colnames)

    X_train_df = pd.DataFrame(X_train_proc, columns=final_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=final_cols, index=X_test.index)
    return X_train_df, X_test_df

def encode_targets(y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mapping = {"yes": 1, "no": 0}
    y_train_enc = y_train.replace(mapping)
    y_test_enc = y_test.replace(mapping)
    if isinstance(y_train_enc, pd.DataFrame):
        y_train_enc = y_train_enc.squeeze()
    if isinstance(y_test_enc, pd.DataFrame):
        y_test_enc = y_test_enc.squeeze()
    return y_train_enc, y_test_enc
