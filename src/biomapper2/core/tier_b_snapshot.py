"""Pinned local Tier B freeze loader: a frozen name to InChIKey corpus for independent structure
evidence.

Why this exists
---------------
Tier B resolves the QUERY NAME against an external registry (PubChem, Metabolomics Workbench) to
corroborate or refute the committed node. Making that live call per unique name across a whole run is
a real cost against rate-limited services, and it was the reason Tier B shipped opt-in. This loader
reads a FROZEN name to InChIKey corpus so the hot path is served from disk: a HIT resolves with no
network, and only a MISS falls back to the live path (behind a circuit breaker). That is what makes
Tier B SAFE to run on by default. It mirrors the RefMet freeze in core/annotators/refmet_snapshot.py.

Format (tab-separated, with a header)::

    name    inchikey    source

``name`` is the query string. ``inchikey`` is the frozen structure: a full InChIKey or a first block,
placed verbatim (upper-cased) into ``TierBResult.inchikey_block``; the live PubChem path emits a
first block, so a corpus built from it stores first blocks, and a full key is accepted too (it lets a
stereo-level agreement register, matching the lipid hop). A non-empty ``inchikey`` is VALIDATED
against first-block form (fourteen letters) or full-InChIKey form; a malformed value is SKIPPED at
load with a loud warning (the name falls back to the live path), never returned as evidence. An EMPTY
``inchikey`` positively records "this name has no independent structure" (a frozen unresolvable,
still no network) and needs no source.

``source`` names the registry the row came from and is REQUIRED on a resolved (non-empty inchikey)
row. It is normalized (strip + casefold) and must be one of the canonical sources the certificate
understands (``pubchem`` / ``metabolomics-workbench`` / ``lipidmaps``); a resolved row whose source is
empty or non-canonical is SKIPPED at load with a loud warning (it falls to the live path) rather than
silently defaulting -- pubchem provenance is never attributed to missing data. The only required
header columns are ``name`` and ``inchikey``.

Lookup is by NORMALIZED name (strip + casefold), so the corpus is case- and surrounding-whitespace
insensitive. Loading is done ONCE per resolved path and cached (immutable); the cache is keyed on the
resolved path so a test that repoints the env picks up a new freeze, and ``reset()`` clears it. A
missing/unreadable path, or a corrupt header, reports NOT present (never raises): the freeze is
inactive and every name falls to the live path.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .certificate import TIER_B_SOURCE_LIPIDMAPS, TIER_B_SOURCE_MW, TIER_B_SOURCE_PUBCHEM

logger = logging.getLogger(__name__)

# The canonical Tier B sources the certificate understands. A RESOLVED freeze row (non-empty
# inchikey) MUST carry one of these; the value is normalized (strip + casefold) and checked against
# this set. A resolved row whose source is EMPTY or NON-canonical is SKIPPED at load (it falls to the
# live path) rather than silently assigned a default provenance -- never attribute pubchem to missing
# data on a resolved row. These are the same source strings the live path emits (see certificate.py):
# pubchem and lipidmaps are independent of every annotator the resolver source-weights toward, and MW
# is handled by the certificate's per-row independence calculation.
_CANONICAL_SOURCES = frozenset({TIER_B_SOURCE_PUBCHEM, TIER_B_SOURCE_MW, TIER_B_SOURCE_LIPIDMAPS})

# InChIKey validation, mirroring the two forms the live path can emit: a 14-character first block
# (MW/PubChem emit first-block only) OR a full InChIKey (block1-block2-flag; the lipid hop emits one).
# A non-empty value that matches neither is a truncation/typo and is SKIPPED at load (falls to live);
# a malformed key must never be returned as RESOLVED evidence.
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}$|^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _normalize(name: str) -> str:
    """The single lookup key: leading/trailing whitespace stripped, then case-folded."""
    return name.strip().casefold()


@dataclass(frozen=True)
class FrozenStructureHit:
    """One row of the freeze: the frozen independent structure for a single query name.

    ``inchikey`` is None for a row that positively records "no structure for this name"; a caller
    treats that as a deterministic unresolvable, NOT as a miss (a miss returns None from ``lookup``).
    """

    inchikey: str | None
    source: str
    version: str | None


@dataclass(frozen=True)
class _LoadedSnapshot:
    """A parsed, immutable freeze: the normalized lookup table plus the derived version."""

    version: str | None
    by_normalized: dict[str, FrozenStructureHit]

    def lookup(self, name: str) -> FrozenStructureHit | None:
        return self.by_normalized.get(_normalize(name))


# Cache keyed on the resolved path string. A load that FAILED is stored as None so a broken/unreadable
# freeze is not retried on every call yet still reports NOT present.
_CACHE: dict[str, _LoadedSnapshot | None] = {}


def reset() -> None:
    """Drop the load cache. Tests repointing the env call this; production never."""
    _CACHE.clear()


def _parse(path: Path) -> _LoadedSnapshot:
    """Read and index the freeze TSV. Raises on an unreadable file or a missing 'name'/'inchikey'
    header: a corrupt freeze must be rejected (reported NOT present), not trusted as an authoritative
    empty corpus that silently turns every name into a miss."""
    version = config.derive_tier_b_snapshot_version(path)
    by_normalized: dict[str, FrozenStructureHit] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "name" not in reader.fieldnames or "inchikey" not in reader.fieldnames:
            raise ValueError(f"Tier B freeze {path} is missing the required 'name'/'inchikey' header")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            raw_inchikey = (row.get("inchikey") or "").strip()
            if not raw_inchikey:
                # A frozen-unresolvable row: positively records "no independent structure" for this
                # name. It needs no source and is served deterministically with no network call.
                by_normalized.setdefault(_normalize(name), FrozenStructureHit(None, "", version))
                continue
            inchikey = raw_inchikey.upper()
            if not _INCHIKEY_RE.fullmatch(inchikey):
                # Finding 3: a malformed inchikey must never become RESOLVED evidence. Skip the row so
                # the name falls back to the live path, and say so loudly.
                logger.warning(
                    "Tier B freeze %s: skipping %r, malformed inchikey %r (falls back to live)",
                    path,
                    name,
                    raw_inchikey,
                )
                continue
            source = (row.get("source") or "").strip().casefold()
            if source not in _CANONICAL_SOURCES:
                # Finding 2: never assign pubchem provenance to missing data on a resolved row. An empty
                # or non-canonical source is skipped so the name falls back to the live path.
                logger.warning(
                    "Tier B freeze %s: skipping %r, resolved row has empty/non-canonical source %r "
                    "(falls back to live; expected one of %s)",
                    path,
                    name,
                    row.get("source"),
                    sorted(_CANONICAL_SOURCES),
                )
                continue
            # First occurrence wins on a duplicate key; the freeze is expected to be unique per name.
            by_normalized.setdefault(_normalize(name), FrozenStructureHit(inchikey, source, version))
    return _LoadedSnapshot(version=version, by_normalized=by_normalized)


def _current() -> _LoadedSnapshot | None:
    """The loaded freeze for the currently configured path, or None when unset/unloadable."""
    path = config.get_tier_b_snapshot_path()
    if path is None:
        return None
    key = str(path)
    if key not in _CACHE:
        try:
            _CACHE[key] = _parse(path)
        except (OSError, ValueError) as exc:
            logger.warning("Tier B freeze at %s could not be loaded (%s); falling back to live lookups", path, exc)
            _CACHE[key] = None
    return _CACHE[key]


def is_present() -> bool:
    """True iff a freeze is configured, loadable, AND has at least one USABLE row.

    The freeze-first rollout switch. The usable-row requirement is a safety guard, not tidiness: a
    header-valid corpus whose rows were ALL rejected at load (e.g. an older corpus missing the
    now-required ``source``, or every inchikey malformed) reduces to an EMPTY lookup table. If such a
    corpus still read as present, the default posture would resolve to ``enabled_freeze``, every
    eligible name would miss the empty table, and it would fall back to live MW/PubChem, defeating the
    inert default-safety. Reporting NOT present sends the default posture to INERT instead. A corpus of
    only frozen-unresolvable rows (empty inchikeys, valid deterministic-unresolvable entries) is
    non-empty and stays present.
    """
    snapshot = _current()
    return snapshot is not None and bool(snapshot.by_normalized)


def version() -> str | None:
    """Version of the loaded freeze (sidecar or filename stem), or None when not present."""
    snapshot = _current()
    return snapshot.version if snapshot is not None else None


def lookup(name: str) -> FrozenStructureHit | None:
    """Frozen independent structure for ``name`` (normalized), or None on a miss.

    None means the name is NOT in the freeze (a miss, which falls back to live). A row that exists but
    carries an empty ``inchikey`` still returns a FrozenStructureHit (inchikey None): the freeze
    positively recorded that the name has no independent structure.
    """
    if not name:
        return None
    snapshot = _current()
    if snapshot is None:
        return None
    return snapshot.lookup(name)
