import pytest

from biomapper2.mapper import Mapper
from biomapper2.utils import setup_logging


# Setup logging once for all tests
setup_logging()


@pytest.fixture(scope="session")
def shared_mapper():
    """
    Creates a session-scoped instantiation of Mapper that is created once per test run and shared across all
    pytest files.
    """
    mapper = Mapper()
    yield mapper
