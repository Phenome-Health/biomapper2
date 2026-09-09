"""Tier B: independent structure evidence for a QUERY NAME. Opt-in, default-off.

What this is for
----------------
Tier A is a *self*-certificate: it reports what the graph asserts about the node the pipeline
committed. That is free and honest, but it cannot corroborate or refute the choice, because both
sides come from the same graph. Tier B resolves the **query name** -- the string the user handed in,
not the node's name -- against an external registry, so a verdict can be independent of the
selection.

Why it is off by default
------------------------
Turning it on moves external calls from a small conflict subset to every unique query name across
every benchmark arm. That is a real cost against rate-limited services, and it changes the meaning
of the emitted state, so it is a decision an operator makes deliberately per run rather than a
default the pipeline drifts into. ``config.TIER_B_ENABLED`` is the switch and a test asserts it is
False.

Independence is a per-row property, not a property of the tier (L26)
--------------------------------------------------------------------
The first hop is Metabolomics Workbench, which is the same registry the RefMet annotator queries to
produce the candidate the resolver source-weights toward. On rows where RefMet supplied the
committed node, Tier B via MW asks RefMet whether RefMet was right. The hop is kept for coverage,
but ``certificate.issue`` computes ``independent_of_selection`` and the published curve is
stratified by source, so independence is claimed only where it holds.

Guarded, throttled, and accounted
---------------------------------
The fetchers are called through this wrapper rather than directly: the swallow-everything
``try/except`` in ``StructureResolver.inchikey_block`` does not cover them, so a ``raise_for_status``
on a server error would propagate into the mapping loop. Failures degrade to ``lookup_failed``, which is kept
distinct from ``unresolvable`` -- a throttled service is a property of the network, and collapsing
the two would turn an operating curve into an artifact of the run. ``stats()`` reports the tier's own
resolution rate, which must accompany every operating point: the endpoints here are EXACT-name
lookups while the annotator uses a fuzzy match, so corroboration is otherwise computed on a biased
easy subset.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import requests_cache

from ..config import (
    CACHE_DIR,
    MW_INCHIKEY_URL,
    PUBCHEM_INCHIKEY_URL,
    STRUCTURE_LOOKUP_TIMEOUT_S,
    TIER_B_BACKOFF_BASE_S,
    TIER_B_MAX_ATTEMPTS,
    TIER_B_MIN_INTERVAL_S,
)
from .certificate import (
    STRUCTURE_CACHE_STORE,
    TIER_B_SOURCE_MW,
    TIER_B_SOURCE_PUBCHEM,
    TierBOutcome,
    TierBResult,
)

log = logging.getLogger(__name__)

# Statuses that mean "this registry does not know this name" rather than "this call went wrong".
# 404 is the normal answer for an unknown compound name at both services.
_NOT_FOUND_STATUSES = frozenset({400, 404})

_UNRESOLVED = TierBResult(source=None, inchikey_block=None, outcome=TierBOutcome.UNRESOLVABLE)


class IndependentStructureLookup:
    """Resolve a query name to an InChIKey first block via MW, then PubChem.

    Every collaborator (session, sleep, clock) is injectable so the whole class is testable against
    fakes. No test in this repo may exercise it against a live service.
    """

    def __init__(
        self,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        min_interval_s: float = TIER_B_MIN_INTERVAL_S,
        max_attempts: int = TIER_B_MAX_ATTEMPTS,
        backoff_base_s: float = TIER_B_BACKOFF_BASE_S,
        lipid_resolver: Any | None = None,
    ) -> None:
        self._session = (
            session if session is not None else requests_cache.CachedSession(str(CACHE_DIR / STRUCTURE_CACHE_STORE))
        )
        self._sleep = sleep
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        # The THIRD hop: Goslin -> LIPID MAPS, resolving lipid shorthand the exact-name registries
        # miss (the dominant REFUSED class). None keeps the tier at MW -> PubChem. Shared with the
        # candidate-side StructureResolver so query and candidate structures come from one lookup.
        self._lipid_resolver = lipid_resolver
        self._last_call_at: float | None = None
        self._memo: dict[str, TierBResult] = {}
        # Unique names attempted, and those whose LAST attempt failed. Tracked separately from the
        # memo because a failed lookup is intentionally not cached (see ``lookup``), so the memo is
        # no longer a complete record of what was attempted.
        self._seen: set[str] = set()
        self._failed: set[str] = set()

    # -- public API ---------------------------------------------------------------------------

    def lookup(self, query_name: str | None) -> TierBResult:
        """Resolve one query name. Never raises; failures come back as ``lookup_failed``."""
        name = (query_name or "").strip()
        if not name:
            return _UNRESOLVED
        if name in self._memo:
            memo = self._memo[name]
            return TierBResult(
                source=memo.source,
                inchikey_block=memo.inchikey_block,
                outcome=memo.outcome,
                cache_state="process_memo",
            )

        self._seen.add(name)
        result = self._resolve(name)
        if result.outcome is TierBOutcome.LOOKUP_FAILED:
            # Deliberately NOT memoized. ``lookup_failed`` is a statement about the network at one
            # instant -- a throttle, a timeout, a 5xx -- not about the name. Caching it would pin a
            # transient outage onto every later occurrence of that name for the rest of the process,
            # so a brief rate-limit would masquerade as a durable property of the data and depress
            # the resolution rate the figure's admissibility gate reads. Counted, not cached.
            self._failed.add(name)
            return result
        self._failed.discard(name)
        self._memo[name] = result
        return result

    def stats(self) -> dict[str, Any]:
        """Tier B's own resolution rate, to be emitted beside every operating point.

        Counted over UNIQUE query names, because a repeated name is one lookup and counting it twice
        would flatter the rate. The published curve must be refused below the floor in
        ``config.TIER_B_MIN_RESOLUTION_RATE``; that check lives with the figure, not here.
        """
        n_unique = len(self._seen)
        n_resolved = sum(1 for r in self._memo.values() if r.outcome is TierBOutcome.RESOLVED)
        n_failed = len(self._failed)
        return {
            "n_unique_query_names": n_unique,
            "n_tier_b_resolved": n_resolved,
            "n_tier_b_lookup_failed": n_failed,
            "resolution_rate": (n_resolved / n_unique) if n_unique else None,
        }

    # -- internals ----------------------------------------------------------------------------

    def _resolve(self, name: str) -> TierBResult:
        any_failure = False
        for source, fetch in ((TIER_B_SOURCE_MW, self._fetch_mw), (TIER_B_SOURCE_PUBCHEM, self._fetch_pubchem)):
            key, cache_state, failed = fetch(name)
            any_failure = any_failure or failed
            if key:
                return TierBResult(
                    source=source,
                    # Upper-cased here as well as at the comparison: the value goes straight
                    # into the emitted certificate column, and an external service's casing must
                    # not become part of the published artifact. MW/PubChem emit FIRST-BLOCK only.
                    inchikey_block=key.split("-")[0].upper(),
                    outcome=TierBOutcome.RESOLVED,
                    cache_state=cache_state,
                )
        # Third hop: the lipid independent-structure source. Only reached once MW and PubChem have
        # both missed, so it costs nothing on the common path. It returns a fully-formed TierBResult
        # (a FULL key + source="lipidmaps" on a hit); a hit flows through unchanged, a lookup failure
        # folds into the tier's own lookup_failed accounting.
        if self._lipid_resolver is not None:
            lipid = self._lipid_resolver.resolve(name)
            if lipid.outcome is TierBOutcome.RESOLVED:
                return lipid
            if lipid.outcome is TierBOutcome.LOOKUP_FAILED:
                any_failure = True
        if any_failure:
            # At least one hop went wrong rather than answering "unknown". Reporting this as
            # ``unresolvable`` would silently fold a service outage into the name-difficulty
            # statistic, which is the number the curve's admissibility rests on.
            return TierBResult(source=None, inchikey_block=None, outcome=TierBOutcome.LOOKUP_FAILED)
        return _UNRESOLVED

    def _fetch_mw(self, name: str) -> tuple[str | None, str | None, bool]:
        """Metabolomics Workbench exact-name endpoint: GET /rest/refmet/name/{name}/inchi_key.

        ``quote(..., safe="")`` is load-bearing, and the default is actively wrong here. ``quote``
        defaults to ``safe="/"``, so a slash inside a query name survives encoding and becomes an
        extra URL path segment; the request 404s and the row is recorded ``unresolvable`` rather
        than ``lookup_failed``. That biases the very ``resolution_rate`` that ``_curve_publishable``
        gates Figure 5 on, in the direction that makes the gate look satisfied. Lipid shorthand is
        full of slashes. Share of affected names, per arm: artifact field ``slash_bearing_name_rate``.
        """

        def parse(payload: Any) -> str | None:
            if isinstance(payload, dict):
                key = payload.get("inchi_key")
                return key if key and key != "-" else None
            return None

        return self._get(f"{MW_INCHIKEY_URL}/{quote(name, safe='')}/inchi_key", parse)

    def _fetch_pubchem(self, name: str) -> tuple[str | None, str | None, bool]:
        """PubChem PUG-REST: GET /rest/pug/compound/name/{name}/property/InChIKey/JSON."""

        def parse(payload: Any) -> str | None:
            props = (payload or {}).get("PropertyTable", {}).get("Properties", []) if isinstance(payload, dict) else []
            return props[0].get("InChIKey") if props else None

        return self._get(f"{PUBCHEM_INCHIKEY_URL}/{quote(name, safe='')}/property/InChIKey/JSON", parse)

    def _get(self, url: str, parse: Callable[[Any], str | None]) -> tuple[str | None, str | None, bool]:
        """One guarded, throttled, retried GET.

        Returns ``(key, cache_state, failed)``. ``failed`` is True only for outcomes that are the
        service's problem rather than the name's -- a not-found answer is a clean "unknown".
        """
        for attempt in range(self._max_attempts):
            self._throttle()
            try:
                response = self._session.get(url, timeout=STRUCTURE_LOOKUP_TIMEOUT_S)
                status = getattr(response, "status_code", 200)
                if status in _NOT_FOUND_STATUSES:
                    return None, self._cache_state(response), False
                response.raise_for_status()
                return parse(response.json()), self._cache_state(response), False
            except Exception as exc:  # noqa: BLE001 - a Tier B failure must never reach the mapping loop
                status = self._status_of(exc)
                if status in _NOT_FOUND_STATUSES:
                    return None, None, False
                if attempt == self._max_attempts - 1:
                    log.warning("Tier B lookup failed for %s; recording lookup_failed", url, exc_info=True)
                    return None, None, True
                self._sleep(self._backoff_base_s * (2**attempt))
        return None, None, True

    def _throttle(self) -> None:
        if self._min_interval_s <= 0:
            return
        now = self._clock()
        if self._last_call_at is not None:
            wait = self._min_interval_s - (now - self._last_call_at)
            if wait > 0:
                self._sleep(wait)
        self._last_call_at = self._clock()

    @staticmethod
    def _cache_state(response: Any) -> str | None:
        from_cache = getattr(response, "from_cache", None)
        if from_cache is None:
            return None
        return "hit" if from_cache else "miss"

    @staticmethod
    def _status_of(exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) if response is not None else None
