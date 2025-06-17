import pandas as pd
import pytest
from projeto_v1.pipelines.data_cleaning.nodes import clean_bank_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [25, 25, 40],
        "pdays": [999, 3, 999],
        "duration": [150, 250, 350],
        "emp.var.rate": [1.1, -1.8, 1.4],
        "cons.price.idx": [93.2, 92.3, 94.1],
        "cons.conf.idx": [-36.4, -41.8, -29.4],
        "nr.employed": [5191.0, 5099.1, 5228.1],
    })

def test_cleaning_removes_duplicates(sample_data):
    df = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    cleaned = clean_bank_data(df, {"drop_duplicates": True, "pdays_missing_value": -1})
    assert cleaned.shape[0] == sample_data.shape[0]

def test_pdays_replacement(sample_data):
    cleaned = clean_bank_data(sample_data, {"drop_duplicates": False, "pdays_missing_value": -1})
    assert all(cleaned["pdays"].isin([3, -1]))

def test_downcast_types(sample_data):
    cleaned = clean_bank_data(sample_data, {"drop_duplicates": False, "pdays_missing_value": -1})
    assert cleaned["age"].dtype in [pd.Int8Dtype(), pd.Int16Dtype(), "int8", "int16"]
