import pytest

from biomapper2.mapper import Mapper


@pytest.fixture(scope="session")
def shared_mapper():
    """
    Creates a session-scoped instantiation of Mapper that is created once per test run and shared across all
    pytest files.
    """
    mapper = Mapper()
    yield mapper
