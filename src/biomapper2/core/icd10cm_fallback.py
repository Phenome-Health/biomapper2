"""ICD-10-CM provided-id fallback (Phase 1a: same-code ``ICD10:`` twin)."""

from __future__ import annotations

import json
from pathlib import Path

# The Linker's per-row side-map column/key: {original ICD10CM curie: {route, mapping_relation, kg_id}}.
# Internal provenance consumed by the Mapper; never emitted as an output field.
SIDE_MAP_FIELD = "_icd10cm_fallback"

_KG_BUILD_CACHE: dict[str, str] = {}


def kg_build_stamp() -> str:
    """``<kg_version>@<git_commit>`` from Kestrel /health, fetched once per process and cached."""
    if "stamp" not in _KG_BUILD_CACHE:
        from ..config import get_kestrel_api_url
        from ..provenance import fetch_kg_build_info

        _, info = fetch_kg_build_info(get_kestrel_api_url())
        _KG_BUILD_CACHE["stamp"] = f"{info.kg_version}@{info.git_commit}"
    return _KG_BUILD_CACHE["stamp"]


def reset() -> None:
    """Clear process-level caches (tests)."""
    _KG_BUILD_CACHE.clear()


def load_twin_denylist(path: Path | str) -> frozenset[str]:
    """Codes whose twin hits must be emitted ``broad``. A missing or malformed file raises."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = payload.get("codes") if isinstance(payload, dict) else None
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        raise ValueError(f"malformed ICD10CM twin denylist at {path}: expected {{'codes': [str, ...]}}")
    return frozenset(codes)
