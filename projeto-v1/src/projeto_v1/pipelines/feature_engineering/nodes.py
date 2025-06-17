"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.5
"""
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import logging
logger = logging.getLogger(__name__)

## DOES NOT INCLUDE CONTACT, MONTH, DAY OF WEEK, AND DURATION.
## We will not include these features in prediction, as they are only relevant for benchmarking.We want to exclude for realistic predictions.

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run feature engineering on the input DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data to be processed.
    
    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    logger.info("Starting feature engineering...")
    df_engineered = df.copy()
    df_engineered = quantile_bin_age(df_engineered)
    df_engineered = bin_campaign(df_engineered)
    df_engineered = bin_previous(df_engineered)
    df_engineered = flagging_cons_price_idx(df_engineered)
    df_engineered = flagging_cons_conf_idx(df_engineered)
    df_engineered = binning_eurbor(df_engineered)
    df_engineered = educ_mapping(df_engineered)
    df_engineered = clean_default(df_engineered)
    df_engineered = age_housing_interaction(df_engineered)
    df_engineered = age_loan_interaction(df_engineered)
    df_engineered = contacted_before(df_engineered)
    df_engineered = employment_rate_interaction(df_engineered)
    
    logger.info("Feature engineering completed.")
    return df_engineered




def quantile_bin_age(df: pd.DataFrame) -> pd.DataFrame:
    # quantile binning on "age" column
    assert "age" in df.columns, "The 'age' column is missing from the DataFrame."
    ## -- ##
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df["age_binned_quantile"] = kbd.fit_transform(df[["age"]])
    return df

def bin_campaign(df: pd.DataFrame) -> pd.DataFrame:
    # binning on "campaign" column
    df["campaign_bin"] = df["campaign"].apply(
        lambda x: 1 if x == 1 else 2 if x == 2 else 3 if x == 3 else 4 if x == 4 else 5 if x == 5 else 6
    )
    assert "campaign_bin" in df.columns, "'campaign_bin' was not created"
    assert df["campaign_bin"].nunique() == 6, "The 'campaign_bin' column should have 6 unique values."
    return df

def bin_previous(df: pd.DataFrame) -> pd.DataFrame:
    # binning on "previous" column
    df["previous_bin"] = df["previous"].apply(
        lambda x: 0 if x == 0 else 1 if x==1 else 2 if x==2 else 3)
    
    assert "previous_bin" in df.columns, "'previous_bin' was not created"
    assert df["previous_bin"].nunique() == 4, "The 'previous_bin' column should have 4 unique values."
    return df


### pdays -> cleaned as -1 depois ver o que fazer com esta

def flagging_cons_price_idx(df: pd.DataFrame) -> pd.DataFrame:
    top_cpi = df["cons.price.idx"].value_counts().nlargest(3).index.tolist()
    df["cpi_top_value"] = df["cons.price.idx"].apply(lambda x: x if x in top_cpi else "other")

    q75 = df["cons.price.idx"].quantile(0.75)
    df["cpi_above_75th"] = (df["cons.price.idx"] > q75).astype(int)
    return df

def flagging_cons_conf_idx(df: pd.DataFrame) -> pd.DataFrame:
    mean = df["cons.conf.idx"].mean()
    df["cci_top_value"] = df["cons.conf.idx"].apply(lambda x: x if x > mean else "below_mean")

    q75 = df["cons.conf.idx"].quantile(0.75)
    df["cci_above_75th"] = (df["cons.conf.idx"] > q75).astype(int)
    return df

def binning_eurbor(df: pd.DataFrame) -> pd.DataFrame:
    # binning on "euribor3m" column
    df["euribor_bin"] = pd.qcut(df["euribor3m"], q=4, labels=["very_low", "low", "high", "very_high"])
    assert df["euribor_bin"].dtype == "category", "The 'euribor_bin' column should be categorical."
    return df


#### CATEGORICAL VARIABLES
def educ_mapping(df: pd.DataFrame) -> pd.DataFrame:
    education_mapping = {
        "university.degree": "higher_education",
        "professional.course": "higher_education",
        "high.school": "high_school",
        "basic.9y": "basic_education",
        "basic.6y": "basic_education",
        "basic.4y": "basic_education",
        "illiterate": "low_or_unknown",
        "unknown": "low_or_unknown"
    }
    df["education_mapped"] = df["education"].map(education_mapping)

    assert df["education_mapped"].nunique() == 4, "The 'education_mapped' column should have 4 unique values."
    return df


def clean_default(df: pd.DataFrame) -> pd.DataFrame:
    # clean "default" column
    df["default"] = df["default"].replace({"no": 0, "unknown": 1, "yes": 1})
    return df


#### NEW FEATURES ###
# Age and Housing Loan Interaction
def age_housing_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["young_housing_loan"] = ((df["age"] <= 25) & (df["housing"] == "yes")).astype(int)
    df["middle_aged_housing_loan"] = ((df["age"] > 25) & (df["age"] <= 50) & (df["housing"] == "yes")).astype(int)
    df["senior_housing_loan"] = ((df["age"] > 50) & (df["housing"] == "yes")).astype(int)
    return df

# Age and Loan Interaction
def age_loan_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["young_loan"] = ((df["age"] <= 25) & (df["loan"] == "yes")).astype(int)
    df["middle_aged_loan"] = ((df["age"] > 25) & (df["age"] <= 50) & (df["loan"] == "yes")).astype(int)
    df["senior_loan"] = ((df["age"] > 50) & (df["loan"] == "yes")).astype(int)
    return df

# A AVALIAR CORRECTAMENTE APOS MUDANCS NA PIPELINE DATA CLEANING
def contacted_before(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature indicating if the client has been contacted before.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data to be processed.
    
    Returns:
    pd.DataFrame: DataFrame with the new feature added.
    """
    df["contacted_before"] = df["pdays"].apply(lambda x: 1 if x == 999 else 0)
    assert 999 in df["pdays"].values
    return df


def employment_rate_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature that is the interaction of employment rate and number of employed.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data to be processed.
    
    Returns:
    pd.DataFrame: DataFrame with the new feature added.
    """
    # Gives us momentum of employment rate
    assert df["emp.var.rate"].dtype in [float, int], "The 'emp.var.rate' column should be numeric."

    df["emp_rate_x_employed"] = df["emp.var.rate"] * df["nr.employed"]
    return df

