"""
This is a boilerplate test file for pipeline 'feature_store'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import os
import pandas as pd
from unittest import mock
from src.projeto_v1.pipelines.feature_store.nodes import upload_to_feature_store
from src.projeto_v1.pipelines.feature_store.pipeline import create_pipeline

@mock.patch.dict(os.environ, {
    "FS_API_KEY": "dummy_key",
    "FS_PROJECT_NAME": "dummy_project"
})
@mock.patch("src.projeto_v1.pipelines.feature_store.nodes.hopsworks.login")
def test_upload_to_feature_store(mock_login):
    # Arrange
    dummy_data = pd.DataFrame({
        "index": [0, 1],
        "feature-1": [0.5, 0.7],
        "feature.2": [1.0, 2.0]
    })
    
    # Mock the project and feature store behavior
    mock_fg = mock.Mock()
    mock_project = mock.Mock()
    mock_project.get_feature_store.return_value.get_or_create_feature_group.return_value = mock_fg
    mock_login.return_value = mock_project

    # Act
    upload_to_feature_store(dummy_data)

    # Assert
    # Check if login was called with correct credentials
    mock_login.assert_called_once_with(
        api_key_value="dummy_key",
        project="dummy_project"
    )

    # Check if feature group insert was called
    assert mock_fg.insert.called

    # Check if column names were cleaned
    called_args = mock_fg.insert.call_args[1]["features"].columns
    assert "feature_1" in called_args
    assert "feature_2" in called_args
    assert "datetime" in called_args

def test_pipeline_has_correct_node():
    pipeline = create_pipeline()
    nodes = pipeline.nodes
    assert len(nodes) == 1
    assert nodes[0].name == "upload_features_to_store"
