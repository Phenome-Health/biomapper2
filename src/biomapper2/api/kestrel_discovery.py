"""
Kestrel discovery: fetch categories and derive vocabulary prefix presets.

Provides functions to dynamically discover entity type categories from the
Kestrel knowledge graph and derive default vocabulary prefix presets by
sampling text-search results and ranking prefix frequency.
"""

import json
import logging
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..config import CACHE_DIR
from ..utils import bulk_kestrel_request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESET_CACHE_PATH = CACHE_DIR / "entity_type_presets.json"
SCHEMA_VERSION = 1
MAX_PREFIXES_PER_CATEGORY = 30
PER_REQUEST_TIMEOUT = 10  # seconds
TOTAL_DERIVE_TIMEOUT = 30  # seconds

# Human-friendly aliases mapping user-facing names to Biolink categories
ALIASES: dict[str, str] = {
    "metabolite": "biolink:SmallMolecule",
    "lipid": "biolink:SmallMolecule",
    "protein": "biolink:Protein",
    "gene": "biolink:Gene",
    "disease": "biolink:Disease",
    "phenotype": "biolink:PhenotypicFeature",
    "pathway": "biolink:Pathway",
    "drug": "biolink:Drug",
    "clinicallab": "biolink:ClinicalFinding",
    "lab": "biolink:ClinicalFinding",
    "microbiome": "biolink:OrganismTaxon",
    "taxonomy": "biolink:OrganismTaxon",
}

# Sample search terms per category for prefix derivation
CATEGORY_SAMPLE_TERMS: dict[str, list[str]] = {
    "biolink:SmallMolecule": ["glucose", "cholesterol", "aspirin"],
    "biolink:Protein": ["insulin", "hemoglobin", "albumin"],
    "biolink:Gene": ["BRCA1", "TP53", "EGFR"],
    "biolink:Disease": ["diabetes", "cancer", "asthma"],
    "biolink:PhenotypicFeature": ["fever", "headache", "fatigue"],
    "biolink:Pathway": ["glycolysis", "citric acid cycle", "fatty acid oxidation"],
    "biolink:Drug": ["metformin", "ibuprofen", "acetaminophen"],
    "biolink:ClinicalFinding": ["blood pressure", "heart rate", "glucose level"],
    "biolink:OrganismTaxon": ["E. coli", "Homo sapiens", "Staphylococcus"],
}

# Static fallback presets — used when both live derivation and disk cache fail
STATIC_FALLBACK: dict[str, list[str]] = {
    "biolink:SmallMolecule": ["CHEBI", "HMDB", "PUBCHEM.COMPOUND", "CHEMBL.COMPOUND", "KEGG.COMPOUND", "INCHIKEY"],
    "biolink:Protein": ["UniProtKB", "PR", "CHEMBL.TARGET", "DRUGBANK"],
    "biolink:Gene": ["ENSEMBL", "NCBIGene", "UniProtKB", "HGNC"],
    "biolink:Disease": ["SNOMEDCT", "MONDO", "DOID", "ICD10", "OMIM"],
    "biolink:PhenotypicFeature": ["SNOMEDCT", "MONDO", "DOID", "HP"],
    "biolink:Pathway": ["SMPDB", "PathWhiz", "REACT", "GO", "KEGG"],
    "biolink:Drug": ["RXNORM", "CHEMBL.COMPOUND", "PUBCHEM.COMPOUND", "DRUGBANK"],
    "biolink:ClinicalFinding": ["HP", "SNOMEDCT", "LOINC"],
    "biolink:OrganismTaxon": ["NCBITaxon"],
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def fetch_categories() -> list[str]:
    """Fetch all Kestrel knowledge graph categories."""
    result = bulk_kestrel_request("GET", "categories", auth_required=False, timeout=PER_REQUEST_TIMEOUT)
    if isinstance(result, list):
        return [str(c) for c in result]
    return []


def sample_prefixes_for_category(
    category: str,
    sample_terms: list[str],
    limit: int = 50,
) -> list[str]:
    """Sample text-search results for a category and rank prefix frequency.

    For each sample term, runs a text search against the Kestrel API, collects
    the ``prefixes`` array from each result node, counts frequency across all
    results, applies a >10% threshold, and returns the top 8 prefixes.

    Args:
        category: Biolink category string (e.g. ``"biolink:SmallMolecule"``).
        sample_terms: List of search terms to query.
        limit: Maximum results per search term.

    Returns:
        Ranked list of prefix strings (up to ``MAX_PREFIXES_PER_CATEGORY``).
    """
    prefix_counter: Counter[str] = Counter()
    total_nodes = 0

    for term in sample_terms:
        try:
            response = bulk_kestrel_request(
                "POST",
                "text-search",
                auth_required=True,
                json={
                    "search_text": [term],
                    "limit": limit,
                    "category": category,
                },
                timeout=PER_REQUEST_TIMEOUT,
            )
        except (requests.exceptions.RequestException, ValueError):
            logger.warning("Text search failed for term=%s category=%s", term, category)
            continue

        # Response is dict keyed by search term, each value is list of result nodes
        if not isinstance(response, dict):
            continue

        for _, nodes in response.items():
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                total_nodes += 1
                prefixes = node.get("prefixes")
                if isinstance(prefixes, list):
                    for p in prefixes:
                        if isinstance(p, str) and p:
                            prefix_counter[p] += 1

    if total_nodes == 0:
        return []

    return [prefix for prefix, _ in prefix_counter.most_common(MAX_PREFIXES_PER_CATEGORY)]


def derive_all_presets(aliases: dict[str, str] | None = None) -> tuple[dict[str, list[str]], bool]:
    """Fetch categories and derive prefix presets for aliased categories.

    Iterates sequentially over aliased categories (at most ~9), running
    sample text searches for each. Categories not in ``CATEGORY_SAMPLE_TERMS``
    use the category display name as a fallback search term.

    Args:
        aliases: Alias mapping (defaults to ``ALIASES``).

    Returns:
        Tuple of (presets dict, completed_fully flag). The flag is False if any
        category failed during sampling — partial results should not be
        persisted to disk.
    """
    if aliases is None:
        aliases = ALIASES

    categories = fetch_categories()
    if not categories:
        return {}, True

    sampled_categories = set(aliases.values())
    presets: dict[str, list[str]] = {}
    completed_fully = True

    # Only sample aliased categories; others get empty presets
    to_sample = [c for c in categories if c in sampled_categories]

    deadline = time.monotonic() + TOTAL_DERIVE_TIMEOUT

    for cat in to_sample:
        if time.monotonic() > deadline:
            logger.warning("Deadline exceeded; skipping remaining categories")
            completed_fully = False
            break

        try:
            terms = CATEGORY_SAMPLE_TERMS.get(cat)
            if terms is None:
                display_name = cat.removeprefix("biolink:")
                terms = [display_name]
            presets[cat] = sample_prefixes_for_category(cat, terms)
        except Exception as exc:
            logger.warning("Sampling failed for %s: %s", cat, exc)
            presets[cat] = []
            completed_fully = False

    # Include non-sampled categories with empty presets
    for cat in categories:
        if cat not in presets:
            presets[cat] = []

    return presets, completed_fully


# ---------------------------------------------------------------------------
# Disk caching
# ---------------------------------------------------------------------------


def save_to_disk(presets: dict[str, list[str]], path: Path | None = None) -> None:
    """Persist presets to disk via atomic write (write to .tmp, rename)."""
    if path is None:
        path = PRESET_CACHE_PATH

    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "presets": presets,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file in same directory, then rename
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path_str, str(path))
        logger.info("Saved entity type presets to %s", path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def load_from_disk(path: Path | None = None) -> dict[str, list[str]] | None:
    """Load presets from disk and validate structural integrity."""
    if path is None:
        path = PRESET_CACHE_PATH

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load presets from %s: %s", path, exc)
        return None

    # Validate schema version
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        logger.warning("Schema version mismatch in %s", path)
        return None

    presets = data.get("presets")
    if not isinstance(presets, dict):
        logger.warning("Invalid presets structure in %s", path)
        return None

    # Basic type checks
    for key, value in presets.items():
        if not isinstance(key, str):
            logger.warning("Invalid category key '%s' in %s", key, path)
            return None
        if not isinstance(value, list):
            logger.warning("Invalid prefix list for '%s' in %s", key, path)
            return None

    return presets


def derive_presets_with_fallback(aliases: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Derive presets with three-tier fallback: live API, disk cache, static.

    1. Try ``derive_all_presets()`` from live Kestrel API.
       - Save to disk ONLY if derivation completed fully.
    2. On failure, try ``load_from_disk()``.
    3. On failure, return ``STATIC_FALLBACK``.

    This function never raises -- it always returns valid preset data.

    Args:
        aliases: Alias mapping (defaults to ``ALIASES``).

    Returns:
        Category-to-prefixes dict.
    """
    if aliases is None:
        aliases = ALIASES

    # Tier 1: Live derivation
    try:
        presets, completed_fully = derive_all_presets(aliases)
        if presets:
            if completed_fully:
                save_to_disk(presets)
                logger.info("Successfully derived and cached %d category presets", len(presets))
            else:
                logger.warning("Partial derivation (%d categories) — not persisting to disk", len(presets))
            return presets
    except Exception as exc:
        logger.warning("Live preset derivation failed: %s", exc)

    # Tier 2: Disk cache
    disk_presets = load_from_disk()
    if disk_presets is not None:
        logger.info("Loaded %d category presets from disk cache", len(disk_presets))
        return disk_presets

    # Tier 3: Static fallback
    logger.warning("Using static fallback presets")
    return dict(STATIC_FALLBACK)
