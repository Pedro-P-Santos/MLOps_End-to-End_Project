"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

def test_kedro_run():
    bootstrap_project(Path.cwd())
    with KedroSession.create(project_path=Path.cwd()) as session:
        result = session.run()
        assert result, "Pipeline returned empty results"

