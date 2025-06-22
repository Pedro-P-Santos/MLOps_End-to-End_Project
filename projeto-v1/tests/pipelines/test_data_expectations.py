import pandas as pd
from pathlib import Path
from src.projeto_v1.pipelines.data_expectations.nodes import build_custom_expectations_for_rawdata

df_sample = pd.read_csv(Path("data/02_intermediate/validated_data.csv")).sample(frac=0.1, random_state=42)

def test_expectations_pass():
    """Test that expectations function returns a DataFrame."""
    result = build_custom_expectations_for_rawdata(df_sample)
    assert isinstance(result, pd.DataFrame), "Expectations did not return a DataFrame"

def test_common_pdays_value():
    """Test that the most common value of pdays is 999."""
    mode_value = df_sample["pdays"].mode().iloc[0]
    assert mode_value == 999, "Most common pdays value is not 999"

def test_common_age_range():
    """Test that the most common age falls within the expected range."""
    mode_age = df_sample["age"].mode().iloc[0]
    assert 25 <= mode_age <= 35, "Most common age is not in the expected range"
