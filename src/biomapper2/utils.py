"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""

import logging
from collections.abc import Iterable
from typing import Any, Literal, TypeGuard, cast

import inflect
import pandas as pd
import requests
from bmt import Toolkit

from .config import BIOLINK_VERSION_DEFAULT, KESTREL_API_KEY, KESTREL_API_URL, LOG_LEVEL

# Type alias for annotation results structure
# Structure: {annotator: {vocabulary: {local_id: result_metadata_dict}}}
AssignedIDsDict = dict[str, dict[str, dict[str, dict[str, Any]]]]

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


def initialize_biolink_model_toolkit(biolink_version: str | None = None) -> Toolkit:
    version = biolink_version if biolink_version else BIOLINK_VERSION_DEFAULT
    url = f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{version}/biolink-model.yaml"
    logging.info("Initializing bmt (Biolink Model Toolkit)...")
    bmt = Toolkit(schema=url)
    return bmt


def standardize_entity_type(entity_type: str, bmt: Toolkit) -> str:
    # Map any aliases to their corresponding biolink category
    entity_type_cleaned = entity_type.removeprefix("biolink:").lower()
    entity_type_singular = singularize(entity_type_cleaned)
    aliases = {"metabolite": "SmallMolecule", "lipid": "SmallMolecule"}
    category_raw = aliases.get(entity_type_singular, entity_type_singular)

    if bmt.is_category(category_raw):
        category_element = bmt.get_element(category_raw)
        if category_element:
            return category_element["class_uri"]

    message = (
        f"Could not find valid Biolink category for entity type '{entity_type}'. "
        f"Valid entity types are: {bmt.get_descendants('NamedThing')}. Or accepted aliases are: {aliases}."
    )
    logging.error(message)
    raise ValueError(message)


def get_descendants(biolink_category: str, bmt: Toolkit) -> set[str]:
    # Get descendants of the given category (includes self)
    if bmt.is_category(biolink_category):
        return set(bmt.get_descendants(biolink_category, formatted=True, mixin=True, reflexive=True))
    else:
        message = (
            f"Category '{biolink_category}' is not a valid biolink category. Valid categories are: "
            f"{bmt.get_descendants('NamedThing')}."
        )
        logging.error(message)
        raise ValueError(message)


def singularize(phrase: str) -> str:
    """Singularize the last word of a phrase.

    Examples:
        "metabolites" -> "metabolite"
        "amino acids" -> "amino acid"
        "classes" -> "class"
    """
    _inflect_engine = inflect.engine()

    words = phrase.split()
    if not words:
        return phrase

    last_word = words[-1]
    singular = _inflect_engine.singular_noun(last_word)

    # singular_noun returns False if the word is already singular
    if singular:
        words[-1] = singular

    return " ".join(words)


def to_list(
    item: str | Iterable[str] | int | Iterable[int] | float | Iterable[float] | None,
) -> list[str | int | float]:
    if item is None:
        return []
    elif isinstance(item, list):
        return cast(list[str | int | float], item)
    elif isinstance(item, (str, int, float)):
        return [item]
    else:
        return list(item)


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


def merge_into_entity(entity: pd.Series | dict[str, Any], series_to_merge: pd.Series) -> pd.Series | dict[str, Any]:
    """Merge fields from series_to_merge into entity, returning the updated entity."""
    if isinstance(entity, pd.Series):
        return pd.concat([entity, series_to_merge])
    else:  # Dict
        return {**entity, **series_to_merge.to_dict()}
