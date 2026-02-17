"""API authentication via API key."""

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key() -> str | None:
    """Get the configured API key from environment."""
    return os.getenv("BIOMAPPER_API_KEY")


async def validate_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """
    Validate the API key from the request header.

    If BIOMAPPER_API_KEY is not set, authentication is disabled (open access).
    If BIOMAPPER_API_KEY is set, the request must include a matching X-API-Key header.

    Returns:
        The validated API key or "open-access" if auth is disabled
    """
    expected_key = get_api_key()

    # If no API key is configured, allow open access
    if expected_key is None:
        return "open-access"

    # API key is required
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key
