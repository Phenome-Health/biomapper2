"""ICD-10-CM provided-id fallback (Phase 1a: same-code ``ICD10:`` twin)."""

from __future__ import annotations

import json
from pathlib import Path


def load_twin_denylist(path: Path | str) -> frozenset[str]:
    """Codes whose twin hits must be emitted ``broad``. A missing or malformed file raises."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = payload.get("codes") if isinstance(payload, dict) else None
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        raise ValueError(f"malformed ICD10CM twin denylist at {path}: expected {{'codes': [str, ...]}}")
    return frozenset(codes)
