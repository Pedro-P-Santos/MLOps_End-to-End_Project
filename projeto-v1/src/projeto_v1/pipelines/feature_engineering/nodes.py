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


'''
Rodrigo - Adicionar as variaveis relevantes não repetidas
Lista para adicionar:
1.
    -     # 2. Histórico de crédito / risco
    df["has_any_loan"] = df[["loan", "housing"]].isin(["yes"]).any(axis=1).astype(int)  # Tem algum tipo de empréstimo
    df["has_default_flag"] = (df["default"] == "yes").astype(int)  # Histórico de incumprimento - Redundante com clean_default() que já transforma default em 0/1.

    # Interpretação de loan_risk_score:
    # Soma o número de sinais de risco financeiro (crédito pessoal, crédito habitação e incumprimento prévio)
    df["loan_risk_score"] = (
        (df["loan"] == "yes").astype(int) +  # Empréstimo pessoal
        (df["housing"] == "yes").astype(int) +  # Crédito habitação
        df["has_default_flag"]  # Histórico de incumprimento
    )
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

2.
    -     month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df["contact_month_num"] = df["month"].map(month_map)
    df["is_summer_contact"] = df["month"].isin(["jun", "jul", "aug"]).astype(int)  # Pode influenciar a disponibilidade e o humor dos clientes
    df["contact_day_priority"] = (df["day_of_week"] == "fri").astype(int)  # Sexta-feira pode ser mais ou menos propícia a conversas

    df["contact_month_quarter"] = pd.cut(
    df["contact_month_num"],
    bins=[0, 3, 6, 9, 12],
    labels=["Q1", "Q2", "Q3", "Q4"]
).astype(str)  # Agrupamento sazonal por trimestre

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

3.
    df["is_senior"] = (df["age"] >= 60).astype(int)  # Pessoas acima dos 60 podem ter comportamentos distintos em relação a investimentos
    is_senior: ⚠️ Redundante com quantile_bin_age + age > 60
    df["is_student_or_retired"] = df["job"].isin(["student", "retired"]).astype(int)  # Pode refletir populações com menos rendimento

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

4.
    df["previous_contacts_flag"] = (df["previous"] > 0).astype(int)  # Houve contacto prévio
    df["successful_prev_contact"] = (df["poutcome"] == "success").astype(int)  # Campanha anterior teve sucesso
    df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1e-6)  # Duração média por contacto

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

5.
    df["economic_pressure_index"] = -df["cons.conf.idx"] + df["cons.price.idx"]  # Índice sintético que reflete pessimismo e inflação

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

6.
    df["marital_edu_combo"] = df["marital"] + "_" + df["education"]  # Pode capturar padrões entre escolaridade e estado civil




✅ Resumo final
Feature	Adicionar?	Comentário
has_any_loan	✅	Nova e útil
has_default_flag	⚠️	Já criada por clean_default, redundante
loan_risk_score	✅	Composta, interpretável e útil
contact_month_num	✅	Nova
is_summer_contact	✅	Nova
contact_day_priority	✅	Nova
contact_month_quarter	✅	Nova
is_senior	⚠️	Redundante se quantile_bin_age for usado
is_student_or_retired	✅	Nova
previous_contacts_flag	⚠️	Redundante com previous_bin
successful_prev_contact	✅	Nova
contact_efficiency	✅	Nova e com potencial explicativo
economic_pressure_index	✅	Nova derivada sintética
marital_edu_combo	✅	Nova combinação útil para detetar padrões latentes

    '''