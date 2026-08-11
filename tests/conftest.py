from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from semantic_cache.config import Settings
from semantic_cache.main import create_app
from tests.helpers import make_settings


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
