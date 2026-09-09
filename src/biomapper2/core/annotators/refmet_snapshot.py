"""Pinned local RefMet freeze loader — the authoritative, deterministic RefMet source.

Why this exists
---------------
The RefMet vote used to come from a LIVE Metabolomics Workbench ``/match`` call gated by a
process-global circuit breaker. Arm-asymmetric availability of that endpoint changed the committed
node (retinol and short-chain-fatty-acid flips), so resolution was non-deterministic. This loader
reads a FROZEN ``/match`` corpus — a TSV recording exactly what ``/match`` returned per query name —
so the breaker leaves the default resolution path and the vote is reproducible.

The freeze is keyed by the EXACT query name. ``/match`` is a fuzzy/synonym layer, and the freeze
already encodes that resolution (this is why a BULK canonical-name export could not replace it — it
lost most of the live ``/match`` votes at the coverage gate). So no normalization is applied at
freeze time; ``lookup`` tries the exact name first and falls back to a light normalization only as a
convenience.

Format (tab-separated, with header)::

    query_name  status  refmet_id  refmet_name  inchi_key  formula  exactmass  pubchem_cid \
        super_class  main_class  sub_class  panels

``status`` is one of ``voted`` / ``no_match`` / ``unavailable``. ``voted`` rows carry a ``refmet_id``
like ``RM0041813``; the others leave it empty. The annotator consumes only ``refmet_id`` today
(``API_FIELDS = ["refmet_id"]``); the remaining columns are loaded and kept for future use.

Loading is done ONCE per resolved path and cached (immutable). The cache is keyed on the resolved
path so a test that repoints ``REFMET_SNAPSHOT_PATH`` picks up the new freeze; ``reset()`` clears it.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ... import config

logger = logging.getLogger(__name__)

# Columns beyond the two structural ones, loaded into ``FreezeHit.extra`` for future use. The
# annotator reads only ``refmet_id`` today; everything else is carried so a later consumer needs no
# reload.
_EXTRA_FIELDS = (
    "refmet_name",
    "inchi_key",
    "formula",
    "exactmass",
    "pubchem_cid",
    "super_class",
    "main_class",
    "sub_class",
    "panels",
)

_WS = re.compile(r"\s+")

# The only statuses a well-formed freeze row may carry (they mirror the AVAILABILITY_* values the
# builder writes). A row outside this set — or a `voted` row with no refmet_id — means the freeze is
# corrupt; `_parse` rejects the WHOLE freeze so it is not silently trusted as authoritative (a
# malformed row would otherwise resolve to a spurious no_match and drop a real vote). A rejected
# freeze reports NOT present, so resolution falls back to live RefMet with a loud warning.
_VALID_STATUSES = frozenset({"voted", "no_match", "unavailable"})


def _normalize(name: str) -> str:
    """Secondary lookup key: lower-case, stripped, internal whitespace collapsed to one space."""
    return _WS.sub(" ", name.strip().lower())


@dataclass(frozen=True)
class FreezeHit:
    """One row of the freeze: the frozen ``/match`` outcome for a single query name."""

    status: str
    refmet_id: str | None
    version: str | None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _LoadedSnapshot:
    """A parsed, immutable freeze: exact and normalized lookup tables plus the derived version."""

    version: str | None
    by_exact: dict[str, FreezeHit]
    by_normalized: dict[str, FreezeHit]

    def lookup(self, name: str) -> FreezeHit | None:
        hit = self.by_exact.get(name)
        if hit is not None:
            return hit
        return self.by_normalized.get(_normalize(name))


# Cache keyed on the resolved path string. A distinct sentinel value (a load that FAILED) is stored
# as None so a broken/unreadable freeze is not retried on every call yet still reports NOT present.
_CACHE: dict[str, _LoadedSnapshot | None] = {}


def reset() -> None:
    """Drop the load cache. Tests repointing ``REFMET_SNAPSHOT_PATH`` call this; production never."""
    _CACHE.clear()


def _parse(path: Path) -> _LoadedSnapshot:
    """Read and index the freeze TSV. Raises on an unreadable file, a missing header, or a malformed
    row (unknown status, or a ``voted`` row with no ``refmet_id``) — a corrupt freeze must be rejected,
    not trusted as authoritative."""
    version = config.derive_refmet_snapshot_version(path)
    by_exact: dict[str, FreezeHit] = {}
    by_normalized: dict[str, FreezeHit] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "query_name" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise ValueError(f"RefMet freeze {path} is missing the required 'query_name'/'status' header")
        for row in reader:
            query_name = (row.get("query_name") or "").strip()
            if not query_name:
                continue
            status = (row.get("status") or "").strip()
            refmet_id = (row.get("refmet_id") or "").strip() or None
            if status not in _VALID_STATUSES:
                raise ValueError(f"RefMet freeze {path} has an unknown status {status!r} for {query_name!r}")
            if status == "voted" and not refmet_id:
                raise ValueError(f"RefMet freeze {path} has a 'voted' row with no refmet_id for {query_name!r}")
            extra = {f: (row.get(f) or "").strip() for f in _EXTRA_FIELDS}
            hit = FreezeHit(status=status, refmet_id=refmet_id, version=version, extra=extra)
            # First occurrence wins on a duplicate key; the freeze is expected to be unique per name.
            by_exact.setdefault(query_name, hit)
            by_normalized.setdefault(_normalize(query_name), hit)
    return _LoadedSnapshot(version=version, by_exact=by_exact, by_normalized=by_normalized)


def _current() -> _LoadedSnapshot | None:
    """The loaded freeze for the currently configured path, or None when unset/unloadable."""
    path = config.get_refmet_snapshot_path()
    if path is None:
        return None
    key = str(path)
    if key not in _CACHE:
        try:
            _CACHE[key] = _parse(path)
        except (OSError, ValueError) as exc:
            logger.warning("RefMet freeze at %s could not be loaded (%s); falling back to live RefMet", path, exc)
            _CACHE[key] = None
    return _CACHE[key]


def is_present() -> bool:
    """True iff a freeze is configured AND loadable. The annotator's rollout switch."""
    return _current() is not None


def version() -> str | None:
    """Version of the loaded freeze (sidecar or filename stem), or None when not present."""
    snapshot = _current()
    return snapshot.version if snapshot is not None else None


def lookup(name: str) -> FreezeHit | None:
    """Frozen ``/match`` outcome for ``name`` (exact first, then normalized), or None on a miss.

    None means the name is NOT in the freeze (a miss). A row that exists but says ``no_match`` still
    returns a FreezeHit — the freeze positively recorded that ``/match`` had no match for it.
    """
    if not name:
        return None
    snapshot = _current()
    if snapshot is None:
        return None
    return snapshot.lookup(name)
