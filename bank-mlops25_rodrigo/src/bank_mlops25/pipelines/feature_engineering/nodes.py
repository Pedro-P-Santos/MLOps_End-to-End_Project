"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Cliente bancário (demográficos)
    df["is_senior"] = (df["age"] >= 60).astype(int)  # Pessoas acima dos 60 podem ter comportamentos distintos em relação a investimentos
    df["is_young"] = (df["age"] < 30).astype(int)  # Jovens podem estar mais recetivos a campanhas mas ter menor estabilidade financeira
    df["is_student_or_retired"] = df["job"].isin(["student", "retired"]).astype(int)  # Pode refletir populações com menos rendimento

    education_map = {
        "illiterate": 0,
        "basic.4y": 1,
        "basic.6y": 2,
        "basic.9y": 3,
        "high.school": 4,
        "professional.course": 5,
        "university.degree": 6,
        "unknown": 3  # valor neutro em vez de -1 (evita distorções em modelos lineares)
    }
    df["education_level"] = df["education"].map(education_map)
    df["marital_is_single"] = (df["marital"] == "single").astype(int)  # Solteiros podem ter menos responsabilidades financeiras

    df["is_employed"] = ~df["job"].isin(["unemployed", "student", "retired", "unknown"])  # Pessoas em idade ativa e empregadas

    # 2. Histórico de crédito / risco
    df["has_any_loan"] = df[["loan", "housing"]].isin(["yes"]).any(axis=1).astype(int)  # Tem algum tipo de empréstimo
    df["has_default_flag"] = (df["default"] == "yes").astype(int)  # Histórico de incumprimento

    # Interpretação de loan_risk_score:
    # Soma o número de sinais de risco financeiro (crédito pessoal, crédito habitação e incumprimento prévio)
    df["loan_risk_score"] = (
        (df["loan"] == "yes").astype(int) +  # Empréstimo pessoal
        (df["housing"] == "yes").astype(int) +  # Crédito habitação
        df["has_default_flag"]  # Histórico de incumprimento
    )

    # df["balance_flag"] = (df["balance"] > 0).astype(int)  # Saldo médio positivo (proxy de estabilidade financeira)

    # 3. Campanha atual de marketing
    month_map = {
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

    # 4. Histórico de contacto
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)  # Corrigido: -1 significa que nunca foi contactado
    df["previous_contacts_flag"] = (df["previous"] > 0).astype(int)  # Houve contacto prévio
    df["successful_prev_contact"] = (df["poutcome"] == "success").astype(int)  # Campanha anterior teve sucesso

    df["contact_efficiency"] = df["duration"] / (df["campaign"] + 1e-6)  # Duração média por contacto
    df["campaign_intensity"] = df["campaign"] / (df["duration"] + 1e-6)  # Número de contactos por segundo (efeito inverso)

    df["log_duration"] = np.log1p(df["duration"])  # Reduz assimetria em variáveis contínuas
    df["log_campaign"] = np.log1p(df["campaign"])

    # 5. Variáveis económicas (deixadas para escalar no pipeline posterior)
    df["economic_pressure_index"] = -df["cons.conf.idx"] + df["cons.price.idx"]  # Índice sintético que reflete pessimismo e inflação

    # Combinação extra
    df["marital_edu_combo"] = df["marital"] + "_" + df["education"]  # Pode capturar padrões entre escolaridade e estado civil

    return df
