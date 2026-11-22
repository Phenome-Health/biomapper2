"""The API reports the installed release, not a hardcoded string.

``/api/v1/health`` once returned a constant that release-please never touched, so prod on 1.5.2
still answered with the original placeholder. These tests pin the version to the package metadata
and the metadata to ``pyproject.toml``, so a stale environment or a new hardcoded value fails here.
"""

import re
from importlib.metadata import version
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from biomapper2.api.constants import API_VERSION
from biomapper2.api.main import app

pytestmark = pytest.mark.unit

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "pyproject.toml has no [project] version line"
    return match.group(1)


def test_api_version_is_the_installed_package_version() -> None:
    assert API_VERSION == version("biomapper2")


def test_installed_version_matches_pyproject() -> None:
    assert version("biomapper2") == _pyproject_version(), "environment is stale: run `uv sync`"


def test_health_and_openapi_report_the_package_version() -> None:
    client = TestClient(app)  # no lifespan: health answers without building the mapper
    assert client.get("/api/v1/health").json()["version"] == _pyproject_version()
    assert app.openapi()["info"]["version"] == _pyproject_version()
