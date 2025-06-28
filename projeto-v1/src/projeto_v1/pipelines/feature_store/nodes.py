"""
This is a boilerplate pipeline 'feature_store'
generated using Kedro 0.19.12
"""

import os
import pandas as pd
import hopsworks

def upload_to_feature_store(data: pd.DataFrame) -> None:
    # Garante que a coluna 'index' e 'datetime' existem
    data = data.reset_index()
    if "datetime" not in data.columns:
        data["datetime"] = pd.to_datetime("now")

    # Limpa os nomes das colunas para serem válidos no Hopsworks
    data.columns = (
        data.columns
            .str.strip()
            .str.lower()
            .str.replace('.', '_', regex=False)
            .str.replace('-', '_', regex=False)
    )

    # Lê as credenciais do ambiente
    api_key = os.environ.get("FS_API_KEY")
    project_name = os.environ.get("FS_PROJECT_NAME")

    if not api_key or not project_name:
        raise ValueError("FS_API_KEY or FS_PROJECT_NAME not set in environment.")

    # Liga ao Hopsworks
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name
    )

    feature_store = project.get_feature_store()

    # Cria ou vai buscar o feature group
    fg = feature_store.get_or_create_feature_group(
        name="bank_features_v2",
        version=1,
        description="Engineered bank marketing features",
        primary_key=["index"],
        event_time="datetime",
        online_enabled=False
    )


    # Insere os dados
    fg.insert(
        features=data,
        overwrite=True,
        write_options={"wait_for_job": True}
    )
