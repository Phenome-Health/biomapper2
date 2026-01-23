"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""

import logging
from datetime import timedelta
from typing import Any, Literal, TypeGuard

import requests
import requests_cache

from .config import CACHE_DIR, KESTREL_API_KEY, KESTREL_API_URL, LOG_LEVEL
from .models import AssignedIDsDict as AssignedIDsDict  # Re-export for backward compatibility

# Type hint for annotation mode
AnnotationMode = Literal["all", "missing", "none"]

VALIDATOR_PROP = "validator"
CLEANER_PROP = "cleaner"
ALIASES_PROP = "aliases"


def setup_logging():
    """Configure logging based on LOG_LEVEL in config.py."""
    if not logging.getLogger().hasHandlers():  # Skip setup if it's already been done
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level = LOG_LEVEL.upper()

        if level not in valid_levels:
            print(f"Invalid log level '{LOG_LEVEL}', defaulting to INFO")
            level = "INFO"

        logging.basicConfig(
            level=getattr(logging, level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def text_is_not_empty(value: Any) -> TypeGuard[str]:
    """Check if a name/text field value is a valid non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def to_list(item: Any) -> list[Any]:
    if item is None:
        return []
    elif isinstance(item, list):
        return item
    elif isinstance(item, (str, int, float)):
        return [item]
    else:
        return list(item)


def to_set(item: Any) -> set[Any]:
    if item is None:
        return set()
    elif isinstance(item, set):
        return item
    elif isinstance(item, (str, int, float)):
        return {item}
    else:
        return set(item)


def safe_divide(numerator, denominator) -> float | None:
    """
    Divide two numbers, returning None if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value

    Returns:
        Result of division, or None if denominator is zero
    """
    # Cast to float to handle potential numpy types
    numerator = float(numerator)
    denominator = float(denominator)

    if denominator == 0.0:
        # Return None, which will be serialized as 'null' in JSON.
        # This is more accurate for metrics like 'accuracy'
        # where a 0 denominator means 'not applicable'.
        return None

    result = numerator / denominator
    return result


def kestrel_request(method: str, endpoint: str, session: requests.Session | None = None, **kwargs) -> Any:
    """
    Internal helper for making Kestrel API requests.

    Args:
        method: HTTP method ('GET' or 'POST')
        endpoint: API endpoint path
        session: Optional requests session (defaults to cached session)
        **kwargs: Additional arguments to pass to requests (json, params, etc.)

    Returns:
        JSON response from API

    Raises:
        requests.exceptions.HTTPError: If API returns error status
        requests.exceptions.RequestException: If request fails
    """
    # Sort search_text in json payload for consistent cache keys (if handling a batch)
    if "json" in kwargs and isinstance(kwargs["json"], dict):
        payload = kwargs["json"]
        if "search_text" in payload and isinstance(payload["search_text"], list):
            payload["search_text"].sort()

    if session is None:
        session = requests_cache.CachedSession(
            CACHE_DIR / "kestrel_http",
            expire_after=timedelta(hours=1),
            allowable_methods=["GET", "POST"],
        )

    try:
        response = session.request(
            method, f"{KESTREL_API_URL}/{endpoint}", headers={"X-API-Key": KESTREL_API_KEY}, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Kestrel API HTTP error ({endpoint}): {e}", exc_info=True)
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Kestrel API request failed ({endpoint}): {e}", exc_info=True)
        raise
