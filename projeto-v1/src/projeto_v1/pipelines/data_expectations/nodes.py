import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from great_expectations.data_context import get_context
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_custom_expectations_for_rawdata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply custom expectations to the ingested dataset.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        pd.DataFrame: The validated DataFrame.
    """
    context = gx.get_context(context_root_dir="gx")

    datasource_name = "bank_source"
    try:
        datasource = context.sources.add_pandas(name=datasource_name)
    except Exception:
        logger.info(f"Datasource {datasource_name} already exists. Skipping creation.")
        datasource = context.datasources[datasource_name]

    ### CREATE EXPECTATION SUITE ###
    suite_name = "bank_suite"
    main_suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    add_target_and_null_expectations(df, main_suite)
    add_type_and_numeric_expectations(df, main_suite)
    add_macro_feature_expectations(main_suite)
    add_categorical_expectations(df, main_suite)

    context.save_expectation_suite(expectation_suite=main_suite)

    ### VALIDATE DATAFRAME ###
    data_asset_name = "bank_source_asset"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except Exception:
        logger.info("The data asset already exists. Loading it.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)


    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_bankraw_suite",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    logger.info("Checkpoint run results:")
    logger.info(checkpoint_result)
    validation_df = get_validation_results(checkpoint_result, df)

    logger.info("Validation results:")
    logger.info(validation_df)
    assert checkpoint_result["success"], "Data expectations failed!"

    return df






def add_target_and_null_expectations(df, suite):
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={"column": "y", "value_set": ["no", "yes"]}
    ))

    null_columns = df.columns.tolist()
    for col in null_columns:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": col}
        ))

def add_type_and_numeric_expectations(df, suite):
    numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous',
                       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                       'euribor3m', 'nr.employed']

    for col in numeric_columns:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_type_list",
            kwargs={"column": col, "type_list": ["int64", "float64"]}
        ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "age", "min_value": 17, "max_value": 98}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_quantile_values_to_be_between",
        kwargs={"column": "age", "quantile_ranges": {"quantiles": [0.25, 0.75], "value_ranges": [[30, 35], [45, 50]]}}
    ))

    # suite.add_expectation(ExpectationConfiguration(
    #     expectation_type="expect_column_most_common_value_to_equal",
    #     kwargs={"column": "age", "value": 31}
    # ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "duration", "min_value": 0, "max_value": 5000, "mostly": 0.95}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_mean_to_be_between",
        kwargs={"column": "campaign", "min_value": 1, "max_value": 3}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "pdays", "min_value": 0, "max_value": 999}
    ))

    # suite.add_expectation(ExpectationConfiguration(
    #     expectation_type="expect_column_values_to_be_between",
    #     kwargs={"column": "pdays", "value": 999, "min_value": 20000, "max_value": 30000}
    # ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_unique_value_count_to_be_between",
        kwargs={"column": "previous", "min_value": 0, "max_value": 8}
    ))

def add_macro_feature_expectations(suite):
    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_distinct_values_to_be_in_set",
        kwargs={"column": "emp.var.rate", "value_set": [1.4, -1.8, 1.1, -0.1, -2.9, -3.4, -1.7, -1.1, -3.0, -0.2]}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "cons.price.idx", "min_value": 92.0, "max_value": 95.0}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_unique_value_count_to_be_between",
        kwargs={"column": "cons.price.idx", "min_value": 20, "max_value": 40}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "cons.conf.idx", "min_value": -51, "max_value": -25}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_unique_value_count_to_be_between",
        kwargs={"column": "cons.conf.idx", "min_value": 20, "max_value": 40}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "euribor3m", "min_value": 0.0, "max_value": 5.5}
    ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_quantile_values_to_be_between",
        kwargs={"column": "euribor3m", "quantile_ranges": {"quantiles": [0.25, 0.75],
                                                              "value_ranges": [[1.0, 2.5], [4.5, 5.3]]}}
    ))

def add_categorical_expectations(df, suite):
    categorical_values = {
        "job": ["admin.", "blue-collar", "technician", "services", "management",
                 "retired", "entrepreneur", "self-employed", "housemaid",
                 "unemployed", "student", "unknown"],
        "marital": ["married", "single", "divorced", "unknown"],
        "education": ["university.degree", "high.school", "basic.9y", "professional.course",
                      "basic.4y", "basic.6y", "unknown", "illiterate"],
        "default": ["no", "yes", "unknown"],
        "housing": ["yes", "no", "unknown"],
        "loan": ["yes", "no", "unknown"],
        "contact": ["cellular", "telephone"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"]
    }

    for col, values in categorical_values.items():
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": col, "value_set": values}
        ))

    suite.add_expectation(ExpectationConfiguration(
        expectation_type="expect_column_unique_value_count_to_be_between",
        kwargs={"column": "day_of_week", "min_value": 5, "max_value": 5}
    ))




def get_validation_results(checkpoint_result, df: pd.DataFrame):
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    
    df_validation = pd.DataFrame(columns=[
        "Success", "Expectation Type", "Column", "Column Pair", "Max Value",
        "Min Value", "Element Count", "Unexpected Count", "Unexpected Percent",
        "Value Set", "Unexpected Value", "Observed Value"
    ])

    for result in results:
        success = result.get('success', '')
        ec = result.get('expectation_config', {})
        rk = ec.get('kwargs', {})
        res = result.get('result', {})

        observed_value = res.get("observed_value", "")
        unexpected_value = [item for item in observed_value if item not in rk.get("value_set", [])] if isinstance(observed_value, list) else []

        df_validation = pd.concat([df_validation, pd.DataFrame([{
            "Success": success,
            "Expectation Type": ec.get("expectation_type", ""),
            "Column": rk.get("column", ""),
            "Column Pair": (rk.get("column_A", ""), rk.get("column_B", "")),
            "Max Value": rk.get("max_value", ""),
            "Min Value": rk.get("min_value", ""),
            "Element Count": res.get("element_count", ""),
            "Unexpected Count": res.get("unexpected_count", ""),
            "Unexpected Percent": res.get("unexpected_percent", ""),
            "Value Set": rk.get("value_set", ""),
            "Unexpected Value": unexpected_value,
            "Observed Value": observed_value
        }])], ignore_index=True
        )

    # Custom assertions on raw df
    most_common_pdays = df["pdays"].mode().iloc[0]
    assert most_common_pdays == 999, f"Expected 999 to be the most common value in pdays, but got {most_common_pdays}"

    most_common_age = df["age"].mode().iloc[0]
    assert 25 <= most_common_age <= 35, f"Expected most common 'age' to be between 25 and 35, but got {most_common_age}"

    return df_validation
