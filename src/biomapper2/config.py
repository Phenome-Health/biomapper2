"""
Configuration settings for biomapper2.

Customize these values to change API endpoints, model versions, and logging behavior.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load environmental variables (secrets)


# Set up our general cache directory (e.g., for requests cache, biolink)
PROJECT_ROOT = Path(__file__).parents[2]
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# KG API configuration
# Set to https://kestrel.nathanpricelab.com/api to use 'production' KESTREL hosting KRAKEN
# Set to http://localhost:8000/api to use a local KESTREL instance
KESTREL_API_URL = "https://kestrel.nathanpricelab.com/api"

# Biolink model version
BIOLINK_VERSION_DEFAULT = "4.2.5"

# Level of logging messages to display (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
LOG_LEVEL = "INFO"

# Secrets (from environment variables)
_kestrel_api_key: str | None = None


def get_kestrel_api_key() -> str:
    global _kestrel_api_key
    if _kestrel_api_key is None:
        _kestrel_api_key = os.getenv("KESTREL_API_KEY")
        if not _kestrel_api_key:
            raise ValueError("KESTREL_API_KEY environment variable is not set")
    return _kestrel_api_key


# Batching for Kestrel API requests (to prevent timeouts on large datasets)
KESTREL_BATCHING_ENABLED = True  # Set to False to disable batching (for performance testing)
KESTREL_BATCH_SIZE_SEARCH = 1000  # For text-search, vector-search, hybrid-search
KESTREL_BATCH_SIZE_CANONICALIZE = 2000  # For canonicalize endpoint
