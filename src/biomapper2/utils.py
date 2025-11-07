import logging

from .config import LOG_LEVEL


def setup_logging():
    """Setup logging with the level specified in config.py"""
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


def safe_divide(numerator, denominator):
    """
    Performs division, returning None if the denominator is zero.
    This avoids division-by-zero warnings and ensures JSON-compatible output.
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