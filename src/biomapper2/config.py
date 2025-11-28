"""
Configuration settings for biomapper2.

Customize these values to change API endpoints, model versions, and logging behavior.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # Load environmental variables (secrets)


# KG API configuration
# Set to https://kestrel.nathanpricelab.com/api to use 'production' KESTREL hosting KRAKEN
# Set to http://localhost:8000/api to use a local KESTREL instance
KESTREL_API_URL = "https://kestrel.nathanpricelab.com/api"

# Biolink model version
BIOLINK_VERSION_DEFAULT = "4.2.5"

# Level of logging messages to display (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
LOG_LEVEL = "INFO"

# Secrets (from environment variables)
KESTREL_API_KEY = os.getenv("KESTREL_API_KEY")
if not KESTREL_API_KEY:
    raise ValueError("KESTREL_API_KEY environment variable is not set")
