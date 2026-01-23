"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""

import logging
from collections.abc import Iterator
from datetime import timedelta
from typing import Any, Literal, TypeGuard

import pandas as pd
import requests
import requests_cache

from .config import CACHE_DIR, KESTREL_API_KEY, KESTREL_API_URL, KESTREL_BATCHING_ENABLED, LOG_LEVEL

# Type alias for annotation results structure
# Structure: {annotator: {vocabulary: {local_id: result_metadata_dict}}}
AssignedIDsDict = dict[str, dict[str, dict[str, dict[str, Any]]]]

# Type hint for annotation mode
AnnotationMode = Literal["all", "missing", "none"]

VALIDATOR_PROP = "validator"
CLEANER_PROP = "cleaner"
ALIASES_PROP = "aliases"


def chunk_list(items: list, chunk_size: int) -> Iterator[list]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Yields:
        List chunks of at most chunk_size items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


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


def bulk_kestrel_request(method: str, endpoint: str, session: requests.Session | None = None, **kwargs) -> Any:
    """
    Make a single Kestrel API request with the full payload.

    This is the low-level function that sends one request. For batching support,
    use kestrel_request() instead.

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


def kestrel_request(
    method: str,
    endpoint: str,
    batch_field: str,
    batch_items: list,
    batch_size: int,
    session: requests.Session | None = None,
    **kwargs,
) -> dict:
    """
    Make Kestrel API requests with automatic batching for large payloads.

    Splits batch_items into chunks, makes separate API calls for each chunk,
    and merges the results. Assumes API returns dict keyed by input items.

    When KESTREL_BATCHING_ENABLED is False, sends all items in a single request
    (useful for performance testing).

    Args:
        method: HTTP method ('GET' or 'POST')
        endpoint: API endpoint path
        batch_field: JSON field name for batch items (e.g., 'search_text', 'curies')
        batch_items: List of items to batch
        batch_size: Maximum items per request (ignored if batching disabled)
        session: Optional requests session
        **kwargs: Additional arguments (json, params, etc.)

    Returns:
        Merged dict of results from all batches
    """
    if not batch_items:
        return {}

    json_payload = kwargs.pop("json", {})

    # If batching is disabled, send all items in a single request
    if not KESTREL_BATCHING_ENABLED:
        full_payload = {**json_payload, batch_field: batch_items}
        result = bulk_kestrel_request(method, endpoint, session=session, json=full_payload, **kwargs)
        return result if isinstance(result, dict) else {}

    # Batch the request
    chunks = list(chunk_list(batch_items, batch_size))
    num_chunks = len(chunks)

    if num_chunks > 1:
        logging.info(f"Batching {len(batch_items)} items into {num_chunks} chunks of {batch_size} for {endpoint}")

    merged_results: dict = {}
    for chunk in chunks:
        chunk_payload = {**json_payload, batch_field: chunk}
        chunk_results = bulk_kestrel_request(method, endpoint, session=session, json=chunk_payload, **kwargs)

        if isinstance(chunk_results, dict):
            merged_results.update(chunk_results)

    return merged_results


def merge_into_entity(entity: pd.Series | dict[str, Any], series_to_merge: pd.Series) -> pd.Series | dict[str, Any]:
    """Merge fields from series_to_merge into entity, returning the updated entity."""
    if isinstance(entity, pd.Series):
        return pd.concat([entity, series_to_merge])
    else:  # Dict
        return {**entity, **series_to_merge.to_dict()}
