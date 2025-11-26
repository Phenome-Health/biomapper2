"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""
import logging
from typing import Optional, Dict, Any

import requests
import pandas as pd

from .config import LOG_LEVEL, KESTREL_API_URL, KESTREL_API_KEY


# Type alias for annotation results structure
# Structure: {annotator: {vocabulary: {local_id: result_metadata_dict}}}
AssignedIDsDict = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]


def setup_logging():
    """Configure logging based on LOG_LEVEL in config.py."""
    if not logging.getLogger().hasHandlers():  # Skip setup if it's already been done
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level = LOG_LEVEL.upper()

        if level not in valid_levels:
            print(f"Invalid log level '{LOG_LEVEL}', defaulting to INFO")
            level = 'INFO'

        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def safe_divide(numerator, denominator) -> Optional[float]:
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


def calculate_f1_score(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """
    Calculate F1 score from precision and recall.

    Args:
        precision: Precision value
        recall: Recall value

    Returns:
        F1 score, or None if either input is None
    """
    if precision is None or recall is None:
        return None
    else:
        return safe_divide(2 * (precision * recall), (precision + recall))


# Kestrel API functions
def kestrel_request(method: str, endpoint: str, **kwargs) -> Any:
    """
    Internal helper for making Kestrel API requests.

    Args:
        method: HTTP method ('GET' or 'POST')
        endpoint: API endpoint path
        **kwargs: Additional arguments to pass to requests (json, params, etc.)

    Returns:
        JSON response from API

    Raises:
        requests.exceptions.HTTPError: If API returns error status
        requests.exceptions.RequestException: If request fails
    """
    try:
        response = requests.request(
            method,
            f"{KESTREL_API_URL}/{endpoint}",
            headers={'X-API-Key': KESTREL_API_KEY},
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Kestrel API HTTP error ({endpoint}): {e}", exc_info=True)
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Kestrel API request failed ({endpoint}): {e}", exc_info=True)
        raise


def merge_into_entity(entity: pd.Series | Dict[str, Any], series_to_merge: pd.Series) -> pd.Series | Dict[str, Any]:
    """Merge fields from series_to_merge into entity, returning the updated entity."""
    if isinstance(entity, pd.Series):
        return pd.concat([entity, series_to_merge])
    else:  # Dict
        return {**entity, **series_to_merge.to_dict()}
