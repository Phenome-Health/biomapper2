"""API constants."""

from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    """The version of the installed biomapper2 package, as release-please last set it.

    Read from package metadata rather than written here, so ``/api/v1/health`` and the OpenAPI
    document report the release that is actually deployed. The metadata
    comes from ``pyproject.toml`` when the environment was last synced: a deploy that bumps the
    version must run ``uv sync`` before restarting the service (``deploy-api.yml`` does), or health
    keeps reporting the previous release.
    """
    try:
        return version("biomapper2")
    except PackageNotFoundError:
        return "unknown"


API_VERSION = _installed_version()
