# IMPORTS #
import pandas as pd
import numpy as np
from typing import Dict, Any   
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from pathlib import Path
import logging

### LOCATING config path ###
# conf_path = str(Path('') / settings.CONF_SOURCE)
conf_path = str(Path(__file__).resolve().parents[4] / settings.CONF_SOURCE)


### Loading config file ###
conf_loader = OmegaConfigLoader(conf_source=conf_path)
### Acess config file ###
credentials = conf_loader["credentials"]
logger = logging.getLogger(__name__)


# INGESTION OF DATA #
def ingestion(bank_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingests data from a DataFrame and performs some basic assertions.

    Args:
        bank_df (pd.DataFrame): The input DataFrame containing bank data.

    Returns:
        pd.DataFrame: The same DataFrame, if all checks pass.
    """
    assert isinstance(bank_df, pd.DataFrame), "Input must be a pandas DataFrame"

    actual_columns = bank_df.shape[1]
    assert actual_columns == 21, f"Expected 21 columns, got {actual_columns}"

    assert bank_df["y"].nunique() == 2, "Target variable 'y' must have 2 unique values"

    logger.info(f"✅ Dataset loaded with {bank_df.shape[0]} rows and {bank_df.shape[1]} columns")

    return bank_df