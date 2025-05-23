"""
This is a boilerplate pipeline 'ingestion'
generated using Kedro 0.19.12
"""

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    df_ge = gx.from_pandas(df)

    # verifying if 'age' is between 18 and 100 (it's not, the minimum is 17)
    result_age = df_ge.expect_column_values_to_be_between("age", min_value=18, max_value=100)
    assert result_age.success, "Coluna 'age' falhou a expectativa de intervalo!"

    # check for marital
    result_marital = df_ge.expect_column_distinct_values_to_be_in_set(
        "marital", ["married", "single", "divorced"]
    )
    assert result_marital.success, "Coluna 'marital' tem valores inesperados!"

    return df  #returns the df if tests are ok
