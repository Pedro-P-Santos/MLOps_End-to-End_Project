"""
This is a boilerplate pipeline 'data_tests'
generated using Kedro 0.19.12
"""


""""
Lets divide in 3 groups the expectations:

1. Numerical features

2. Categorical features

3. Target variable
------------------



"""
import logging
import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

logger = logging.getLogger(__name__)

def get_validation_results(checkpoint_result):
    validation_result_data = list(checkpoint_result["run_results"].values())[0]["validation_result"]
    results = validation_result_data["results"]
    
    df_validation = pd.DataFrame(columns=[
        "Success", "Expectation Type", "Column", "Column Pair", "Max Value", "Min Value",
        "Element Count", "Unexpected Count", "Unexpected Percent",
        "Value Set", "Unexpected Value", "Observed Value"
    ])
    
    for result in results:
        expectation = result.get("expectation_config", {})
        kwargs = expectation.get("kwargs", {})

        row = {
            "Success": result.get("success"),
            "Expectation Type": expectation.get("expectation_type"),
            "Column": kwargs.get("column", ""),
            "Column Pair": (kwargs.get("column_A", ""), kwargs.get("column_B", "")),
            "Max Value": kwargs.get("max_value", ""),
            "Min Value": kwargs.get("min_value", ""),
            "Element Count": result.get("result", {}).get("element_count", ""),
            "Unexpected Count": result.get("result", {}).get("unexpected_count", ""),
            "Unexpected Percent": result.get("result", {}).get("unexpected_percent", ""),
            "Value Set": kwargs.get("value_set", ""),
            "Unexpected Value": result.get("result", {}).get("unexpected_values", []),
            "Observed Value": result.get("result", {}).get("observed_value", "")
        }

        df_validation = pd.concat([df_validation, pd.DataFrame([row])], ignore_index=True)

    return df_validation


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    context = gx.get_context()
    datasource_name = "bank_datasource"
    data_asset_name = "bank_data"

    try:
        datasource = context.sources.add_pandas(datasource_name)
    except Exception:
        datasource = context.datasources[datasource_name]

    # Create expectation suite
    suite = context.add_or_update_expectation_suite("Bank")

    # Add expectations
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "age", "min_value": 18, "max_value": 100}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "age", "type_": "int64"}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={"column": "marital", "value_set": ["single", "married", "divorced"]}
    ))
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_of_type",
        kwargs={"column": "marital", "type_": "str"}
    ))

    # Register suite
    context.add_or_update_expectation_suite(expectation_suite=suite)

    # Create batch request
    data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Run checkpoint
    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="bank_data_checkpoint",
        data_context=context,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": "Bank"
        }]
    )
    checkpoint_result = checkpoint.run()

    # Get structured results
    df_validation = get_validation_results(checkpoint_result)
    logger.info("Validação concluída com sucesso.")

    return df_validation
