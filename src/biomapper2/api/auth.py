"""API authentication via API key."""

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> set[str]:
    """
    Get configured API keys from environment.

    Supports multiple keys via comma-separated BIOMAPPER2_API_KEYS,
    with fallback to single BIOMAPPER_API_KEY for backward compatibility.

    Returns:
        Set of valid API keys (empty if none configured)
    """
    keys: set[str] = set()

    # Primary: comma-separated list (BIOMAPPER2_API_KEYS)
    if multi := os.getenv("BIOMAPPER2_API_KEYS"):
        keys.update(k.strip() for k in multi.split(",") if k.strip())

    # Fallback: single key for backward compatibility (BIOMAPPER_API_KEY)
    if single := os.getenv("BIOMAPPER_API_KEY"):
        keys.add(single)

    return keys


async def validate_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """
    Validate the API key from the request header.

    If no API keys are configured, authentication is disabled (open access).
    If API keys are configured, the request must include a valid X-API-Key header.

    Returns:
        The validated API key or "open-access" if auth is disabled
    """
    valid_keys = get_api_keys()

    # If no API keys are configured, allow open access
    if not valid_keys:
        return "open-access"

    # API key is required
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key
