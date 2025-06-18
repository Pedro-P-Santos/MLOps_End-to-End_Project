"""
This is a boilerplate pipeline 'feature_store'
generated using Kedro 0.19.12
"""

#AINDA NAO HA INTEGRAÇÃO COM AS EXPECTATIONS!!


from great_expectations.core import ExpectationSuite
from typing import Any, Dict
import pandas as pd
import hopsworks

def upload_to_feature_store(
    data: pd.DataFrame,
    credentials: Dict[str, Any]
) -> None:
    project = hopsworks.login(
        api_key_value=credentials["FS_API_KEY"],
        project=credentials["FS_PROJECT_NAME"]
    )

    feature_store = project.get_feature_store()

    fg = feature_store.get_or_create_feature_group(
        name="bank_features",
        version=1,
        description="Engineered bank marketing features",
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False
    )

    fg.insert(
        features=data,
        overwrite=True,
        write_options={"wait_for_job": True}
    )

    # Opcional: update de estatísticas
    fg.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    fg.update_statistics_config()
    fg.compute_statistics()
