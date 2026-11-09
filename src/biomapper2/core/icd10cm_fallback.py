"""ICD-10-CM provided-id fallback (Phase 1a: the same-code ``ICD10:`` twin).

Why this exists
---------------
Since v1.3.0 a provided ``ICD10CM:<code>`` keeps its prefix. The pinned KRAKEN build has no ``ICD10CM``
nodes, but its ``ICD10:`` namespace already holds many CM-shaped ids, so codes that used to resolve
through the old accidental ``ICD10:`` rewrite came back unresolved. This module restores that, honestly
labelled:

* **Expansion (before the batch).** Every ``ICD10CM:X`` curie adds ``ICD10:X`` to the ONE batched
  /canonicalize call the Linker already makes. No extra sequential round trip per row.
* **Selection (after the batch, per row).** Tiers are direct > twin. If any of the row's own curies
  linked under its own prefix, the fallback is ignored (it can never outvote a direct curie). Otherwise
  each ``ICD10CM:X`` whose twin linked is written back under the ORIGINAL ``ICD10CM:X`` key through a
  per-row overlay, so ``kg_ids_provided`` and resolver voting are unchanged.
* **Labelling.** A twin hit carries ``mapping_relation`` = ``config.ICD10CM_TWIN_MAPPING_RELATION`` (set
  from the committed gate artifact), or ``broad`` for codes on the gate's disagreement denylist.
* **R7a.** A lookup that is not a definite null is an error. Kestrel answers a miss with an explicit
  null; when bisect-on-5xx isolates (drops) a curie the row depends on, the row raises
  :class:`Icd10cmFallbackLookupError` instead of reading the gap as "no node".
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from .. import config

logger = logging.getLogger(__name__)

ICD10CM_PREFIX = "ICD10CM:"
ICD10_PREFIX = "ICD10:"
ROUTE_TWIN = "twin"

# The Linker's per-row side-map column/key: {original ICD10CM curie: {route, mapping_relation, kg_id}}.
# Internal provenance consumed by the Mapper; never emitted as an output field.
SIDE_MAP_FIELD = "_icd10cm_fallback"

_PACKAGED_DENYLIST = ("data", "icd10cm", "twin_denylist.json")

_KG_BUILD_CACHE: dict[str, str] = {}
# A failed /health lookup is NOT cached for the process: it is retried once this many seconds have
# passed, so a brief outage on the first fallback row does not pin ``unknown@unknown`` forever.
_KG_BUILD_RETRY_SECONDS = 60.0
_KG_BUILD_FAILED: dict[str, tuple[float, str]] = {}  # {"last": (monotonic time of the failure, its stamp)}
_DENYLIST_CACHE: dict[str, frozenset[str]] = {}


class Icd10cmFallbackLookupError(RuntimeError):
    """A fallback candidate's lookup did not return a definite answer (R7a): never read as unresolved."""


def is_enabled() -> bool:
    """Read at call time so ops (env) and tests (monkeypatch) can flip it."""
    return bool(config.ICD10CM_FALLBACK_ENABLED)


def twin_of(curie: str) -> str | None:
    """``ICD10:X`` for an ``ICD10CM:X`` curie; None for any other curie."""
    if curie.startswith(ICD10CM_PREFIX):
        return ICD10_PREFIX + curie[len(ICD10CM_PREFIX) :]
    return None


def expand_candidates(curies: Iterable[str]) -> list[str]:
    """The twin curies to add to the batch, deduplicated and excluding curies already being asked."""
    asked = set(curies)
    return sorted({t for c in asked if (t := twin_of(c)) is not None and t not in asked})


def select_for_row(
    row_curies: list[str],
    canonical: Mapping[str, str | None],
    isolated: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Per-row selection against the UNMODIFIED batch result.

    Returns ``(overlay, side_map)``: ``overlay`` maps an original ``ICD10CM:X`` curie to the node its twin
    linked to (empty unless the fallback applies), ``side_map`` records route + relation per curie.
    """
    icd_curies = [c for c in row_curies if c.startswith(ICD10CM_PREFIX)]
    if not icd_curies:
        return {}, {}
    if any(canonical.get(c) for c in row_curies):
        return {}, {}  # direct tier present: the fallback never votes (R5)
    if isolated:
        unknown = {c for c in row_curies if c in isolated}
        for curie in icd_curies:
            twin = twin_of(curie)
            if twin is not None and twin in isolated:
                unknown.add(twin)
        if unknown:
            raise Icd10cmFallbackLookupError(
                f"/canonicalize gave no definite answer for {sorted(unknown)} (isolated by bisect-on-5xx); "
                f"refusing to read the gap as unresolved or to fall through to a weaker candidate"
            )
    denylist = twin_denylist()
    overlay: dict[str, str] = {}
    side_map: dict[str, dict[str, Any]] = {}
    for curie in icd_curies:
        twin = twin_of(curie)
        node = canonical.get(twin) if twin else None
        if not node:
            continue
        code = curie[len(ICD10CM_PREFIX) :]
        relation = "broad" if code in denylist else config.ICD10CM_TWIN_MAPPING_RELATION
        overlay[curie] = node
        side_map[curie] = {"route": ROUTE_TWIN, "mapping_relation": relation, "kg_id": node}
    return overlay, side_map


def kg_build_stamp() -> str:
    """``<kg_version>@<git_commit>`` from Kestrel /health, fetched once per process and cached.

    A lookup that degrades to ``unknown`` is not cached: it is reused for ``_KG_BUILD_RETRY_SECONDS``
    (so a down /health is not hit per row) and then retried. Warns once when a resolved build differs
    from the build the twin gate ran on (``config.ICD10CM_TWIN_GATE_KG_BUILD``).
    """
    if "stamp" in _KG_BUILD_CACHE:
        return _KG_BUILD_CACHE["stamp"]
    last_failure = _KG_BUILD_FAILED.get("last")
    if last_failure is not None and time.monotonic() - last_failure[0] < _KG_BUILD_RETRY_SECONDS:
        return last_failure[1]

    from ..config import get_kestrel_api_url
    from ..provenance import fetch_kg_build_info

    _, info = fetch_kg_build_info(get_kestrel_api_url())
    stamp = f"{info.kg_version}@{info.git_commit}"
    if "unknown" in (info.kg_version, info.git_commit):
        _KG_BUILD_FAILED["last"] = (time.monotonic(), stamp)
        return stamp
    _KG_BUILD_FAILED.clear()
    _KG_BUILD_CACHE["stamp"] = stamp
    if stamp != config.ICD10CM_TWIN_GATE_KG_BUILD:
        logger.warning(
            "ICD10CM twin gate ran on KG build %s but Kestrel serves %s; twin relations were not "
            "re-validated for this build (see studies/analysis/icd10cm_twin_control.py)",
            config.ICD10CM_TWIN_GATE_KG_BUILD,
            stamp,
        )
    return stamp


def load_twin_denylist(path: Path | str) -> frozenset[str]:
    """Codes whose twin hits must be emitted ``broad``. A missing or malformed file raises."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = payload.get("codes") if isinstance(payload, dict) else None
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        raise ValueError(f"malformed ICD10CM twin denylist at {path}: expected {{'codes': [str, ...]}}")
    return frozenset(codes)


def twin_denylist() -> frozenset[str]:
    """The configured (or packaged) twin denylist, loaded once per resolved path."""
    if config.ICD10CM_TWIN_DENYLIST_PATH:
        path = str(config.ICD10CM_TWIN_DENYLIST_PATH)
        if path not in _DENYLIST_CACHE:
            _DENYLIST_CACHE[path] = load_twin_denylist(path)
        return _DENYLIST_CACHE[path]
    key = "<packaged>"
    if key not in _DENYLIST_CACHE:
        resource = resources.files("biomapper2").joinpath(*_PACKAGED_DENYLIST)
        with resources.as_file(resource) as path:
            _DENYLIST_CACHE[key] = load_twin_denylist(path)
    return _DENYLIST_CACHE[key]


def reset() -> None:
    """Clear process-level caches (tests)."""
    _KG_BUILD_CACHE.clear()
    _KG_BUILD_FAILED.clear()
    _DENYLIST_CACHE.clear()
