import pytest

from shelfsense.config import get_settings
from shelfsense.seed_db import build


@pytest.fixture(scope="session", autouse=True)
def database():
    settings = get_settings()
    if not settings.db_path.exists():
        build(settings.db_path)
    return settings.db_path


@pytest.fixture(scope="session")
def copilot(database):
    from shelfsense.agents.orchestrator import Copilot
    from shelfsense.observability.tracing import Tracer
    return Copilot(llm=None, tracer=Tracer())
