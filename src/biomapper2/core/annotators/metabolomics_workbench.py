"""Metabolomics Workbench RefMet API annotator for metabolite entities."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast
from urllib.parse import quote

import pandas as pd
import requests
import requests_cache
from circuitbreaker import CircuitBreakerError, circuit

from ...config import CACHE_DIR, CACHE_IGNORED_PARAMETERS
from ...utils import AssignedIDsDict
from . import refmet_snapshot
from .base import (
    AVAILABILITY_NO_MATCH,
    AVAILABILITY_NOT_QUERIED,
    AVAILABILITY_UNAVAILABLE,
    AVAILABILITY_VOTED,
    REFMET_SOURCE_LIVE,
    REFMET_SOURCE_LOCAL,
    REFMET_SOURCE_NOT_IN_SNAPSHOT,
    REFMET_SOURCE_NOT_QUERIED,
    REFMET_SOURCE_UNAVAILABLE,
    BaseAnnotator,
)


@dataclass(frozen=True)
class RefMetResult:
    """One RefMet lookup outcome: availability status, the payload when it VOTED, and provenance.

    ``status`` is one of the ``AVAILABILITY_*`` strings. ``data`` is the raw API dict only for a
    VOTED result; NO_MATCH and UNAVAILABLE carry ``None`` so an UNAVAILABLE outcome can never be
    mistaken for (or folded into) an empty vote.

    ``source`` is one of the ``REFMET_SOURCE_*`` strings — WHICH source served the row (the pinned
    local freeze, a freeze miss, the live endpoint, or an unreachable service) — on a parallel axis
    to ``status``. ``version`` is the freeze version when a snapshot served (or was consulted for)
    the row, else None.
    """

    status: str
    data: dict[str, Any] | None = None
    source: str = REFMET_SOURCE_NOT_QUERIED
    version: str | None = None


class MetabolomicsWorkbenchAnnotator(BaseAnnotator):
    """Annotator that queries the Metabolomics Workbench RefMet match API.

    Retrieves RefMet IDs for metabolite entities using fuzzy name matching.
    Returns raw API field names; the Normalizer handles mapping and ID cleaning.

    The /match endpoint handles non-standard names (e.g., "cholate" -> "Cholic acid").
    Only refmet_id is returned since KRAKEN has all RefMet equivalencies.

    API Endpoint: GET https://www.metabolomicsworkbench.org/rest/refmet/match/{metabolite_name}
    """

    slug = "metabolomics-workbench"
    BASE_URL = "https://www.metabolomicsworkbench.org/rest/refmet/match"

    # Only extract refmet_id - KRAKEN has all RefMet equivalencies
    API_FIELDS = ["refmet_id"]

    # Resilience tunables (D4). Class attributes so they are instance-overridable (tests set an
    # instance attribute) AND readable on an instance built via ``__new__`` in a test double.
    #   REQUEST_TIMEOUT_S : per-attempt HTTP timeout, raised from the old fast-fail value so a
    #                       merely-slow-but-alive endpoint is not counted as a failure (O3).
    #   MAX_RETRIES       : extra attempts INSIDE the circuit-decorated call. At most one, so a
    #                       one-off slow call is rescued without loosening the breaker.
    #   RETRY_BACKOFF_S   : short bounded pause between attempts (skipped after the last attempt).
    #   BATCH_DEADLINE_S  : hard per-batch wall-clock bound. Once elapsed, remaining names are
    #                       marked UNAVAILABLE with NO network call, so worst-case time is bounded
    #                       regardless of retry accounting and without any serial pacing.
    REQUEST_TIMEOUT_S: float = 8.0
    MAX_RETRIES: int = 1
    RETRY_BACKOFF_S: float = 0.5
    BATCH_DEADLINE_S: float = 120.0
    # Class-level default so an instance built via __new__ (some tests bypass __init__) still has a
    # safe unarmed value; the deadline branches then no-op without touching the injectable clock.
    _batch_deadline: float | None = None

    # Default-OFF. When a snapshot is present, a freeze MISS resolves deterministically to NO_MATCH
    # (source=not_in_snapshot) with NO network call. Only when this is True does a miss fall through
    # to the live /match + breaker path — deliberately opt-in so the breaker stays out of the
    # default resolution path.
    LIVE_API_FALLBACK: bool = False

    def __init__(
        self,
        *,
        request_timeout_s: float | None = None,
        max_retries: int | None = None,
        retry_backoff_s: float | None = None,
        batch_deadline_s: float | None = None,
        live_api_fallback: bool | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._session = requests_cache.CachedSession(
            CACHE_DIR / "metabolomics_workbench_http",
            expire_after=timedelta(days=7),
            ignored_parameters=CACHE_IGNORED_PARAMETERS,
            # allowable_codes stays default (200 only): timeouts/5xx are NOT cached, so a transient
            # UNAVAILABLE is never frozen into the on-disk cache. Only a genuine no-match (a 200 with
            # refmet_id == "-") is cached.
        )
        # Injectable so a test can override without a real sleep or wall clock; default to real ones.
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        if request_timeout_s is not None:
            self.REQUEST_TIMEOUT_S = request_timeout_s
        if max_retries is not None:
            self.MAX_RETRIES = max_retries
        if retry_backoff_s is not None:
            self.RETRY_BACKOFF_S = retry_backoff_s
        if batch_deadline_s is not None:
            self.BATCH_DEADLINE_S = batch_deadline_s
        if live_api_fallback is not None:
            self.LIVE_API_FALLBACK = live_api_fallback

    def arm_batch_deadline(self) -> bool:
        """Start the shared per-batch wall-clock bound; return True iff THIS call set it.
        Only the FIRST (outermost) arm sets the clock, so an API /batch loop that maps one entity at
        a time bounds the WHOLE loop. A nested caller (each entity re-enters ``_fetch_all``) gets
        False and must NOT disarm, or it would wipe the outer deadline. Pair with a guarded
        ``disarm_batch_deadline`` in a finally.
        """
        if self._batch_deadline is None:
            self._batch_deadline = self._clock() + self.BATCH_DEADLINE_S
            return True
        return False

    def disarm_batch_deadline(self) -> None:
        """Clear the shared per-batch deadline so the next batch (or a single lookup) is unbounded."""
        self._batch_deadline = None

    def _past_batch_deadline(self) -> bool:
        return self._batch_deadline is not None and self._clock() >= self._batch_deadline

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,  # accepted for interface parity; metabolites have no HGNC analogue
        preferred_prefixes: set[str] | None = None,  # parity; MW returns raw IDs, no namespace re-rank
        accepted_categories: set[str] | None = None,  # parity; MW returns raw IDs, no KG categories to check
        candidate_limit: int | None = None,  # parity; MW is not a Kestrel search, has no candidate window
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations.

        Emits the vote only. VOTED yields ``{slug: {refmet_id: {...}}}``; NO_MATCH and UNAVAILABLE
        both yield ``{slug: {}}`` (an empty vote). The NO_MATCH/UNAVAILABLE distinction lives in
        ``get_availability``, never here — folding UNAVAILABLE into a vote is the silent-blanking bug.
        """

        # Extract the entity name
        name = entity.get(name_field)

        if not name:
            # No name provided, cannot annotate
            return {}

        # Use cache if available, otherwise fetch from API
        if cache is not None:
            result = cache.get(name)
            api_data = result.data if result is not None else None
        else:
            api_data = self._fetch_refmet_data(name).data

        if not api_data:
            # No vote: genuine no-match OR unavailable service (status carried by get_availability)
            return {self.slug: {}}

        # Build the annotations structure using raw API field names
        annotations: dict[str, dict[str, dict[str, Any]]] = {}

        for api_field in self.API_FIELDS:
            value = api_data.get(api_field)
            if value:
                # Use raw field name and value - Normalizer handles mapping and cleaning
                annotations.setdefault(api_field, {})[value] = {}

        return {self.slug: annotations}

    def get_availability(self, entity: dict | pd.Series, name_field: str, cache: dict | None = None) -> dict[str, str]:
        """RefMet availability for one row: VOTED / NO_MATCH / UNAVAILABLE (or NOT_QUERIED).

        Reads the SAME cache ``get_annotations`` reads, so a degraded row is fetched (and
        breaker-counted) once. With no cache (the single-entity path builds none) it fetches inline.
        """
        name = entity.get(name_field)
        if not name:
            return {self.slug: AVAILABILITY_NOT_QUERIED}
        if cache is not None:
            result = cache.get(name)
            status = result.status if result is not None else AVAILABILITY_NOT_QUERIED
        else:
            status = self._fetch_refmet_data(name).status
        return {self.slug: status}

    def get_source(self, entity: dict | pd.Series, name_field: str, cache: dict | None = None) -> dict[str, str]:
        """WHICH RefMet source served this row: local_snapshot / not_in_snapshot / live_api / unavailable.

        Parallel to ``get_availability`` and reads the SAME cache, so a row is fetched (and, on the
        live path, breaker-counted) once. With no cache it fetches inline.
        """
        name = entity.get(name_field)
        if not name:
            return {self.slug: REFMET_SOURCE_NOT_QUERIED}
        if cache is not None:
            result = cache.get(name)
            source = result.source if result is not None else REFMET_SOURCE_NOT_QUERIED
        else:
            source = self._fetch_refmet_data(name).source
        return {self.slug: source}

    def build_availability_cache(
        self, items: dict | pd.Series | pd.DataFrame, name_field: str
    ) -> dict[str, RefMetResult]:
        """Fetch every unique name ONCE, honoring the hard per-batch deadline (D4)."""
        if isinstance(items, pd.DataFrame):
            names = items[name_field].dropna().unique().tolist()
        else:
            name = items.get(name_field)
            names = [name] if name else []
        return self._fetch_all(names)

    def get_annotations_bulk(
        self,
        entities: pd.DataFrame,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,  # accepted for interface parity; metabolites have no HGNC analogue
        preferred_prefixes: set[str] | None = None,  # parity; MW returns raw IDs, no namespace re-rank
        accepted_categories: set[str] | None = None,  # parity; MW returns raw IDs, no KG categories to check
        candidate_limit: int | None = None,  # parity; MW is not a Kestrel search, has no candidate window
        cache: dict | None = None,
    ) -> pd.Series:
        """Implements BaseAnnotator.get_annotations_bulk.

        ``cache`` is the pre-built RefMet cache the engine threads in so the vote and the
        availability signal share one fetch; a direct caller may omit it and one is built here.
        """

        # Build the RefMet cache (once) unless the engine already did.
        if cache is None:
            logging.info(f"Fetching RefMet data for {entities[name_field].nunique()} unique metabolite names")
            cache = self.build_availability_cache(entities, name_field)

        # Apply get_annotations to each row using the cache
        assigned_ids_col = entities.apply(
            self.get_annotations,
            axis=1,
            cache=cache,
            name_field=name_field,
            category=category,
            prefixes=prefixes,
            prefer_human=prefer_human,
        )

        return cast(pd.Series, assigned_ids_col)

    def _fetch_all(self, names: list[str]) -> dict[str, RefMetResult]:
        """Fetch each name into a RefMetResult, stopping network work at the hard batch deadline."""
        cache: dict[str, RefMetResult] = {}
        armed_here = self.arm_batch_deadline()
        try:
            for name in names:
                # _fetch_refmet_data marks names past the armed deadline UNAVAILABLE with no network.
                cache[name] = self._fetch_refmet_data(name)
        finally:
            # Only the outermost armer disarms; a route-armed batch deadline survives each entity's
            # nested _fetch_all so the whole /batch loop stays bounded (not re-armed per row).
            if armed_here:
                self.disarm_batch_deadline()
        return cache

    def _fetch_refmet_data(self, metabolite_name: str) -> RefMetResult:
        """Resolve one name to a RefMetResult, consulting the pinned freeze FIRST when present.

        SAFE ROLLOUT (this is the whole point of the change):
        - Snapshot ABSENT (not configured / not loadable): behave EXACTLY as before — live /match +
          breaker + retry + deadline, tagged source=live_api (or unavailable). Zero behavior change.
        - Snapshot PRESENT: consult the freeze first, so the circuit breaker leaves the default
          resolution path. A freeze hit (voted/no_match) is authoritative and deterministic; a MISS
          resolves to NO_MATCH (source=not_in_snapshot) with NO network call — UNLESS
          ``LIVE_API_FALLBACK`` is on, in which case a miss falls through to the live path.
        """
        if refmet_snapshot.is_present():
            return self._fetch_from_snapshot(metabolite_name)
        return self._fetch_live(metabolite_name)

    def _fetch_from_snapshot(self, metabolite_name: str) -> RefMetResult:
        """Resolve one name against the pinned freeze (breaker OUT of the path)."""
        snapshot_version = refmet_snapshot.version()
        hit = refmet_snapshot.lookup(metabolite_name)
        if hit is not None:
            if hit.status == AVAILABILITY_VOTED and hit.refmet_id:
                data = {"refmet_id": hit.refmet_id}
                return RefMetResult(AVAILABILITY_VOTED, data, source=REFMET_SOURCE_LOCAL, version=snapshot_version)
            if hit.status == AVAILABILITY_UNAVAILABLE:
                # The freeze positively recorded that /match did not answer for this name at freeze
                # time. Deterministic, still no network: surface UNAVAILABLE, source=local_snapshot.
                return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_LOCAL, version=snapshot_version)
            # NO_MATCH (or a malformed 'voted' row with no id): the freeze answered "no such
            # metabolite". Deterministic no-match from the local source.
            return RefMetResult(AVAILABILITY_NO_MATCH, source=REFMET_SOURCE_LOCAL, version=snapshot_version)
        # MISS: the name is not in the freeze.
        if not self.LIVE_API_FALLBACK:
            # Default deterministic path: NO_MATCH with NO network call, breaker untouched.
            return RefMetResult(AVAILABILITY_NO_MATCH, source=REFMET_SOURCE_NOT_IN_SNAPSHOT, version=snapshot_version)
        # Opt-in fallback: resolve the miss against the live endpoint (source=live_api / unavailable).
        return self._fetch_live(metabolite_name)

    def _fetch_live(self, metabolite_name: str) -> RefMetResult:
        """Live /match fetch, classifying the outcome as VOTED / NO_MATCH / UNAVAILABLE.

        A CircuitBreakerError (breaker open), a transport error, or a timeout is UNAVAILABLE — the
        service did not answer. A 200 whose refmet_id is "-" (or a non-dict body) is NO_MATCH — the
        service answered "no such metabolite". Data is VOTED.
        """
        if self._past_batch_deadline():
            # Past the shared batch deadline: mark UNAVAILABLE with NO network call.
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        try:
            data = self._do_refmet_request(metabolite_name)
        except CircuitBreakerError:
            logging.debug(f"RefMet API outage, skipping '{metabolite_name}' (circuit open)")
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch RefMet data for '{metabolite_name}': {e}")
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        if data is None:
            return RefMetResult(AVAILABILITY_NO_MATCH, source=REFMET_SOURCE_LIVE)
        return RefMetResult(AVAILABILITY_VOTED, data, source=REFMET_SOURCE_LIVE)

    @circuit(failure_threshold=3, recovery_timeout=300)
    def _do_refmet_request(self, metabolite_name: str) -> dict[str, Any] | None:
        """
        Make the actual HTTP request. Protected by circuit breaker - trips on repeated failures,
        meaning it will skip requests until a cooldown period is over (recovery_timeout).

        The retry lives HERE, INSIDE the circuit-decorated call, on purpose. circuitbreaker counts
        one raise per decorated call, so an internal retry that ultimately fails counts ONCE and one
        that succeeds resets the breaker; a retry placed ABOVE the decorator would instead trip the
        breaker faster — the reverse of the intent. At most ``MAX_RETRIES`` extra attempts, each
        after a short bounded backoff (skipped after the final attempt).
        """
        attempts = self.MAX_RETRIES + 1
        for attempt in range(attempts):
            try:
                return self._request_once(metabolite_name)
            except requests.RequestException:
                # Do not spend another attempt+backoff once the shared batch deadline has passed.
                if attempt + 1 >= attempts or self._past_batch_deadline():
                    raise
                self._sleep(self.RETRY_BACKOFF_S)
        return None  # unreachable: the loop returns or raises

    def _request_once(self, metabolite_name: str) -> dict[str, Any] | None:
        """One HTTP attempt. Returns the data dict, or None for a genuine no-match.

        ``safe=""`` is load-bearing, and this is the highest-consequence of the six call sites that
        needed it. MW's REST is PATH-SEGMENT addressed, and ``quote`` defaults to ``safe="/"``, so a
        slash in the query name splits into an extra segment: the service reads the tail as
        ``output_item`` and returns NO CANDIDATE rather than an error. This annotator's candidates
        enter the vote and the source-weighting, so unlike the ``structure_resolver`` fallback rung
        this silently reached ``chosen_kg_id``. It also corrupted ``independent_of_selection``: with
        no RefMet vote, ``metabolomics-workbench`` is absent from the committed node's sources, so a
        Tier-B-via-MW verdict was reported independent -- true in fact, false in reason, on exactly
        the lipid population L26 exists to stratify.
        """
        url = f"{self.BASE_URL}/{quote(metabolite_name, safe='')}"

        # When a batch deadline is armed, cap this attempt's timeout by the remaining budget so a
        # lookup started just before the deadline cannot run a full timeout past it.
        timeout = self.REQUEST_TIMEOUT_S
        if self._batch_deadline is not None:
            timeout = max(0.1, min(timeout, self._batch_deadline - self._clock()))
        response = self._session.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            if data.get("refmet_id") == "-":
                # /match endpoint returns dict with "-" values when no match found
                return None
            return data

        # A 200 whose body is not a dict (null, an array, a proxy-injected blob) is a DEGRADED
        # response, not RefMet's refmet_id == "-" negative. Surface it as UNAVAILABLE (raise ->
        # classified UNAVAILABLE by _fetch_refmet_data), never a genuine no-match.
        raise requests.RequestException(f"non-dict RefMet response for '{metabolite_name}'")
