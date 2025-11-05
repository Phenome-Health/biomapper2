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
