"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""
import logging
from typing import Optional

from .config import LOG_LEVEL


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