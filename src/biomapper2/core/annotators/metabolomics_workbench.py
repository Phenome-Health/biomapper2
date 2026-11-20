"""Metabolomics Workbench RefMet API annotator for metabolite entities."""

import contextvars
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import pandas as pd
import requests
import requests_cache
from circuitbreaker import CircuitBreakerError, circuit

from ... import config
from ...config import CACHE_IGNORED_PARAMETERS, get_cache_dir
from ...utils import AssignedIDsDict, quote_name_path_segment
from ..name_case import canonical_query
from . import refmet_snapshot, refmet_store
from .base import (
    AVAILABILITY_MATCHED_NO_ID,
    AVAILABILITY_NO_MATCH,
    AVAILABILITY_NOT_QUERIED,
    AVAILABILITY_UNAVAILABLE,
    AVAILABILITY_VOTED,
    REFMET_SOURCE_FREEZE_BACKUP,
    REFMET_SOURCE_KEPT_LIVE_MATCHED_NO_ID,
    REFMET_SOURCE_KEPT_LIVE_NO_MATCH,
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
    VOTED result; NO_MATCH, MATCHED_NO_ID and UNAVAILABLE carry ``None`` so an UNAVAILABLE outcome
    can never be mistaken for (or folded into) an empty vote.

    ``source`` is one of the ``REFMET_SOURCE_*`` strings — WHICH source served the row (the pinned
    local freeze, a freeze miss, the live endpoint, or an unreachable service) — on a parallel axis
    to ``status``. ``version`` is the freeze version when a snapshot served (or was consulted for)
    the row, else None.
    """

    status: str
    data: dict[str, Any] | None = None
    source: str = REFMET_SOURCE_NOT_QUERIED
    version: str | None = None


@dataclass(frozen=True)
class _MatchedNoId:
    """What one request returns when /match matched a name and no id could be had for the match.

    A per-name answer, RETURNED from the circuit-decorated call (never raised), so the breaker counts
    it as a success: RefMet answered. ``refmet_name`` is the matched name, kept for the log line.
    """

    refmet_name: str


_BATCH_DEADLINES: contextvars.ContextVar[dict[int, float] | None] = contextvars.ContextVar(
    "_MW_BATCH_DEADLINES", default=None
)

# Names skipped with no request because the breaker was open, per armed batch and keyed by instance
# id like the deadline. The list is created when a batch is armed and only ever MUTATED afterwards,
# so a nested call (or a worker thread running on a copy of the context) adds to the same list the
# outermost armer reads when it disarms.
_BATCH_BREAKER_SKIPS: contextvars.ContextVar[dict[int, list[str]] | None] = contextvars.ContextVar(
    "_MW_BATCH_BREAKER_SKIPS", default=None
)

# How many skipped names the per-batch INFO line lists before it stops.
_BREAKER_SKIP_NAMES_LOGGED = 5


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
    # /match stopped returning refmet_id (observed 2026-11-10; it still returns refmet_name). The
    # /name endpoint keyed by that refmet_name still returns the id, so it backfills the id.
    NAME_URL = "https://www.metabolomicsworkbench.org/rest/refmet/name"

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

    # Context-local deadline keyed by instance id so concurrent API worker requests never share,
    # inherit, or disarm another request's RefMet batch deadline, while an instance built via
    # __new__ (some tests bypass __init__) still reads a safe unarmed None default.
    @property
    def _batch_deadline(self) -> float | None:
        deadlines = _BATCH_DEADLINES.get()
        return deadlines.get(id(self)) if deadlines is not None else None

    @_batch_deadline.setter
    def _batch_deadline(self, value: float | None) -> None:
        current = _BATCH_DEADLINES.get() or {}
        updated = dict(current)
        if value is None:
            updated.pop(id(self), None)
        else:
            updated[id(self)] = value
        _BATCH_DEADLINES.set(updated or None)

    # Default-OFF. When a snapshot is present, a freeze MISS resolves deterministically to NO_MATCH
    # (source=not_in_snapshot) with NO network call. Only when this is True does a miss fall through
    # to the live /match + breaker path — deliberately opt-in so the breaker stays out of the
    # default resolution path.
    LIVE_API_FALLBACK: bool = False

    # RefMet freeze mode (D5). ``None`` (the default when constructed directly) preserves the
    # historical snapshot-first-if-present dispatch. The engine resolves ``get_refmet_freeze_mode()``
    # and passes an explicit ``off`` | ``frozen`` | ``live_backup`` string, which selects the code path
    # in ``_fetch_refmet_data`` (``off``->live only; ``frozen``->immutable freeze; ``live_backup``->
    # live-first + write-through backup store).
    freeze_mode: str | None = None

    def __init__(
        self,
        *,
        request_timeout_s: float | None = None,
        max_retries: int | None = None,
        retry_backoff_s: float | None = None,
        batch_deadline_s: float | None = None,
        live_api_fallback: bool | None = None,
        freeze_mode: str | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._session = requests_cache.CachedSession(
            get_cache_dir() / "metabolomics_workbench_http",
            expire_after=timedelta(days=7),
            ignored_parameters=CACHE_IGNORED_PARAMETERS,
            # allowable_codes stays default (200 only): timeouts/5xx are NOT cached, so a transient
            # UNAVAILABLE is never frozen into the on-disk cache. Only a genuine no-match (a 200 with
            # refmet_id == "-") is cached.
        )
        # Injectable so a test can override without a real sleep or wall clock; default to real ones.
        # The batch deadline and skip list are keyed by id(self), and CPython reuses the id of a
        # collected object. A new instance must not inherit a deadline an earlier one left armed in
        # this context (it would read every name as past the deadline: UNAVAILABLE, no request).
        self.disarm_batch_deadline()
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
        if freeze_mode is not None:
            self.freeze_mode = freeze_mode
        # Mode live_backup seeds the store from the pinned freeze on first start (D8): a no-op when the
        # store already has rows or no freeze is configured, so it is safe to run at every construction.
        if self.freeze_mode == "live_backup":
            if refmet_store.is_configured():
                snapshot_path = config.get_refmet_snapshot_path()
                if snapshot_path is not None:
                    # Best-effort like every other store call: a store or migration error here must
                    # not fail mapper start-up for this worker.
                    try:
                        refmet_store.seed_from_tsv(snapshot_path)
                    except Exception as e:  # noqa: BLE001 — backup store is best-effort
                        logging.warning("RefMet backup store seed failed (ignored): %s", e)
            else:
                # Selected live_backup but no REFMET_STORE_PATH: the store no-ops, so there is NO outage
                # fallback and behavior collapses to live-only. Surface it loudly rather than silently.
                logging.warning(
                    "REFMET_FREEZE_MODE=live_backup but REFMET_STORE_PATH is not set: the backup store "
                    "is inactive (no outage fallback, no write-back) — behaving as live-only. Set "
                    "REFMET_STORE_PATH to enable the write-through backup."
                )

    def arm_batch_deadline(self) -> bool:
        """Start the shared per-batch wall-clock bound; return True iff THIS call set it.
        Only the FIRST (outermost) arm sets the clock, so an API /batch loop that maps one entity at
        a time bounds the WHOLE loop. A nested caller (each entity re-enters ``_fetch_all``) gets
        False and must NOT disarm, or it would wipe the outer deadline. Pair with a guarded
        ``disarm_batch_deadline`` in a finally.
        """
        if self._batch_deadline is None:
            self._batch_deadline = self._clock() + self.BATCH_DEADLINE_S
            _BATCH_BREAKER_SKIPS.set({**(_BATCH_BREAKER_SKIPS.get() or {}), id(self): []})
            return True
        return False

    def disarm_batch_deadline(self) -> None:
        """Clear the shared per-batch deadline so the next batch (or a single lookup) is unbounded.

        Also the end of the batch for logging: names skipped because the breaker was open are reported
        here in ONE INFO line with their count, so collateral from an open breaker is visible in
        production logs without one line per name.
        """
        self._batch_deadline = None
        skips_by_instance = dict(_BATCH_BREAKER_SKIPS.get() or {})
        skipped = skips_by_instance.pop(id(self), None)
        _BATCH_BREAKER_SKIPS.set(skips_by_instance or None)
        if skipped:
            shown = ", ".join(repr(name) for name in skipped[:_BREAKER_SKIP_NAMES_LOGGED])
            more = len(skipped) - _BREAKER_SKIP_NAMES_LOGGED
            logging.info(
                "RefMet circuit open: skipped %d name(s) in this batch with no request (%s%s)",
                len(skipped),
                shown,
                f", and {more} more" if more > 0 else "",
            )

    def _note_breaker_skip(self, metabolite_name: str) -> None:
        """Record one name skipped because the breaker is open (see ``disarm_batch_deadline``)."""
        skipped = (_BATCH_BREAKER_SKIPS.get() or {}).get(id(self))
        if skipped is None:
            # No armed batch (a single lookup): there is no batch end to report at, so say it now.
            logging.info("RefMet circuit open: skipped %r with no request", metabolite_name)
            return
        skipped.append(metabolite_name)
        logging.debug("RefMet circuit open: skipped %r with no request", metabolite_name)

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

        Emits the vote only. VOTED yields ``{slug: {refmet_id: {...}}}``; NO_MATCH, MATCHED_NO_ID and
        UNAVAILABLE all yield ``{slug: {}}`` (an empty vote). The distinction between them lives in
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
        """RefMet availability for one row: VOTED / NO_MATCH / MATCHED_NO_ID / UNAVAILABLE (or NOT_QUERIED).

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
        # One memo per batch, keyed by the exact query string: case variants of one name share the
        # canonical-case query, so it is issued once and every such spelling gets the same answer.
        memo: dict[str, RefMetResult] = {}
        armed_here = self.arm_batch_deadline()
        try:
            for name in names:
                # _fetch_refmet_data marks names past the armed deadline UNAVAILABLE with no network.
                cache[name] = self._fetch_refmet_data(name, memo=memo)
        finally:
            # Only the outermost armer disarms; a route-armed batch deadline survives each entity's
            # nested _fetch_all so the whole /batch loop stays bounded (not re-armed per row).
            if armed_here:
                self.disarm_batch_deadline()
        return cache

    def _fetch_refmet_data(self, metabolite_name: str, memo: dict[str, RefMetResult] | None = None) -> RefMetResult:
        """Resolve one name to a RefMetResult, case-robustly.

        RefMet ``/match`` reads some notations case-sensitively: ``acetylcarnitine (C2)`` matches and
        ``acetylcarnitine (c2)`` does not, and ``...-gpc (p-16:0/18:1)*`` matches a different, less
        specific record than ``...-GPC (P-16:0/18:1)*``. When the name carries a recognized notation token
        in non-canonical case, the spelling from :func:`~biomapper2.core.name_case.canonical_query` is
        queried FIRST; the name as given is the fallback, used only when the canonical query is a
        NO_MATCH (never on UNAVAILABLE, which is an outage, not an answer, and never on MATCHED_NO_ID,
        where the canonical spelling did match). Any other name is queried exactly as given, as before.
        """
        canonical = canonical_query(metabolite_name)
        if canonical is None:
            return self._memo_dispatch(metabolite_name, memo)
        result = self._memo_dispatch(canonical, memo)
        if result.status != AVAILABILITY_NO_MATCH:
            return result
        fallback = self._memo_dispatch(metabolite_name, memo)
        # The canonical miss stands only if the raw spelling is also a definitive miss: a raw-spelling
        # outage stays UNAVAILABLE (the raw spelling might have matched), never a confirmed NO_MATCH.
        return fallback if fallback.status != AVAILABILITY_NO_MATCH else result

    def _memo_dispatch(self, query: str, memo: dict[str, RefMetResult] | None) -> RefMetResult:
        """Dispatch one exact query, reusing an answer already fetched for it in this batch."""
        if memo is None:
            return self._dispatch_refmet(query)
        if query not in memo:
            memo[query] = self._dispatch_refmet(query)
        return memo[query]

    def _dispatch_refmet(self, metabolite_name: str) -> RefMetResult:
        """Resolve one exact query string, dispatching on the configured freeze mode (D5).

        - ``off``         : live ``/match`` only (breaker+retry+deadline), source=live_api/unavailable.
        - ``frozen``      : the immutable freeze served first (deterministic; breaker out of the path);
                            a miss is not_in_snapshot NO_MATCH unless ``LIVE_API_FALLBACK`` is on.
        - ``live_backup`` : live-first + write-through store served as backup when live is down; a
                            stored vote is kept against a live no-match (source=kept_live_no_match)
                            and against a live match with no id (source=kept_live_matched_no_id).
        - ``None`` (legacy, direct construction): the historical snapshot-first-if-present dispatch,
          preserved byte-for-byte so a caller that only sets ``REFMET_SNAPSHOT_PATH`` is unchanged.
        """
        if self.freeze_mode == "live_backup":
            return self._fetch_live_backup(metabolite_name)
        if self.freeze_mode == "frozen":
            return self._fetch_from_snapshot(metabolite_name)
        if self.freeze_mode == "off":
            return self._fetch_live(metabolite_name)
        # Legacy dispatch (freeze_mode unset): snapshot-first-if-present, exactly as before the modes.
        if refmet_snapshot.is_present():
            return self._fetch_from_snapshot(metabolite_name)
        return self._fetch_live(metabolite_name)

    def _fetch_live_backup(self, metabolite_name: str) -> RefMetResult:
        """Live-first with a write-through store as an outage backup (mode ``live_backup``, D3/D6/D7).

        Try live ``/match`` first. A live answer (VOTED or NO_MATCH) is returned as source=live_api
        and written back to the store when it differs from what is stored (D7 — no write on an
        identical hit, avoiding write amplification), with ONE exception (see refmet_store): a live NO_MATCH
        for a query whose stored row is a vote with an id. RefMet ``/match`` has stopped matching names
        it used to match, so that vote is kept: it is returned as source=kept_live_no_match, the store
        row is not overwritten, and the change in the live answer is recorded once in the store's
        change log. A live NO_MATCH with no stored vote is still a real negative and is written back
        (D6). A live match with a different id still wins.

        A live MATCHED_NO_ID (live matched the name and gave no id for the match) is handled like the
        kept vote, and for the same reason: the name still matches, only the id is missing from the
        answer. A stored vote with an id for this exact query is served as
        source=kept_live_matched_no_id. With no stored vote the live answer stands (MATCHED_NO_ID,
        source=live_api). Either way NOTHING is written: the store's live columns hold only a vote or a
        no-match, and this answer is neither, so the row and the change log are left as they were.

        When live is DOWN (breaker open / timeout / transport → UNAVAILABLE), serve the store's
        last-known-good value as source=freeze_backup (D3: any age); on a store miss, UNAVAILABLE.
        """
        live = self._fetch_live(metabolite_name)
        if live.status == AVAILABILITY_MATCHED_NO_ID:
            try:
                hit = refmet_store.get(metabolite_name)
            except Exception as e:  # noqa: BLE001 — backup store is best-effort; a failure is not fatal
                logging.warning("RefMet backup store read failed for %r (treated as miss): %s", metabolite_name, e)
                return live
            if hit is not None and hit.status == AVAILABILITY_VOTED and hit.refmet_id:
                logging.info(
                    "RefMet live match for %r has no id; served stored vote %s (source=%s)",
                    metabolite_name,
                    hit.refmet_id,
                    REFMET_SOURCE_KEPT_LIVE_MATCHED_NO_ID,
                )
                return RefMetResult(
                    AVAILABILITY_VOTED,
                    {"refmet_id": hit.refmet_id},
                    source=REFMET_SOURCE_KEPT_LIVE_MATCHED_NO_ID,
                    version=hit.snapshot_version,
                )
            return live
        if live.status in (AVAILABILITY_VOTED, AVAILABILITY_NO_MATCH):
            # Every store call here is BEST-EFFORT: an unreadable / unwritable / full / corrupt /
            # locked store must NEVER abort a valid live resolution (or the surrounding mapping).
            refmet_id = live.data.get("refmet_id") if live.data else None
            if live.status == AVAILABILITY_VOTED and not refmet_id:
                # Never overwrite a last-known-good id with a vote that carries none.
                logging.warning("RefMet live vote for %r has no refmet_id; backup store not updated", metabolite_name)
                return live
            try:
                hit = refmet_store.get(metabolite_name)
            except Exception as e:  # noqa: BLE001 — backup store is best-effort; a failure is not fatal
                # Without the row there is no telling whether a vote is stored, so nothing is written:
                # a blind write-back of a no-match could overwrite one.
                logging.warning("RefMet backup store read failed for %r (store not updated): %s", metabolite_name, e)
                return live
            keep_vote = (
                live.status == AVAILABILITY_NO_MATCH
                and hit is not None
                and hit.status == AVAILABILITY_VOTED
                and bool(hit.refmet_id)
            )
            if self._store_write_due(hit, live.status, refmet_id):
                try:
                    kind = refmet_store.apply_live_answer(metabolite_name, live.status, refmet_id=refmet_id)
                except Exception as e:  # noqa: BLE001 — backup store is best-effort; a failure is not fatal
                    logging.warning("RefMet backup store write failed for %r (ignored): %s", metabolite_name, e)
                    kind = None
                if kind == refmet_store.CHANGE_VOTE_KEPT and hit is not None:
                    # Once per query (the store records the change once), not once per request.
                    logging.warning(
                        "RefMet live /match no longer matches %r; kept stored vote %s (source=%s)",
                        metabolite_name,
                        hit.refmet_id,
                        REFMET_SOURCE_KEPT_LIVE_NO_MATCH,
                    )
            if keep_vote and hit is not None:
                return RefMetResult(
                    AVAILABILITY_VOTED,
                    {"refmet_id": hit.refmet_id},
                    source=REFMET_SOURCE_KEPT_LIVE_NO_MATCH,
                    version=hit.snapshot_version,
                )
            return live
        # Live UNAVAILABLE: fall back to the last-known-good store value. A store READ failure likewise
        # degrades to UNAVAILABLE rather than aborting the mapping.
        try:
            hit = refmet_store.get(metabolite_name)
        except Exception as e:  # noqa: BLE001 — backup store is best-effort; a failure is not fatal
            logging.warning("RefMet backup store read failed for %r (treated as miss): %s", metabolite_name, e)
            hit = None
        if hit is not None:
            return self._result_from_store_hit(hit)
        return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)

    @staticmethod
    def _store_write_due(hit: refmet_store.StoreHit | None, live_status: str, live_refmet_id: str | None) -> bool:
        """Whether a live answer needs any store write, given the row read for its query.

        False only when the row already says everything the answer would write: its vote columns need
        no write-back and its last-recorded live answer is the same answer. The store function
        re-checks all of this inside its own transaction; this only spares the write on a repeat.
        """
        if hit is None:
            return True
        has_vote = hit.status == AVAILABILITY_VOTED and bool(hit.refmet_id)
        if not has_vote and hit.status != AVAILABILITY_NO_MATCH:
            return True  # an `unavailable` row or a vote without an id: no last-known-good, overwrite it
        answer = (live_status, live_refmet_id if live_status == AVAILABILITY_VOTED else None)
        stored = (hit.status, hit.refmet_id if has_vote else None)
        # Empty live columns mean "live last said what the row says" (see refmet_store's docstring).
        last_live = stored
        if hit.live_status is not None:
            last_live = (hit.live_status, hit.live_refmet_id if hit.live_status == AVAILABILITY_VOTED else None)
        keeps_vote = live_status == AVAILABILITY_NO_MATCH and has_vote
        return last_live != answer or (not keeps_vote and stored != answer)

    @staticmethod
    def _result_from_store_hit(hit: refmet_store.StoreHit) -> RefMetResult:
        """Translate a backup store row into a RefMetResult (source=freeze_backup).

        A VOTED row replays its refmet_id; a NO_MATCH row replays the negative deterministically. A
        stored UNAVAILABLE row carries no usable last-known-good, so it is surfaced as a plain
        UNAVAILABLE (source=unavailable), never a misleading freeze_backup.
        """
        if hit.status == AVAILABILITY_VOTED and hit.refmet_id:
            return RefMetResult(
                AVAILABILITY_VOTED,
                {"refmet_id": hit.refmet_id},
                source=REFMET_SOURCE_FREEZE_BACKUP,
                version=hit.snapshot_version,
            )
        if hit.status == AVAILABILITY_NO_MATCH:
            return RefMetResult(
                AVAILABILITY_NO_MATCH,
                source=REFMET_SOURCE_FREEZE_BACKUP,
                version=hit.snapshot_version,
            )
        return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)

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
        """Live /match fetch, classifying the outcome as VOTED / NO_MATCH / MATCHED_NO_ID / UNAVAILABLE.

        A CircuitBreakerError (breaker open), a transport error, or a timeout is UNAVAILABLE — the
        service did not answer. A 200 whose refmet_id is "-" (or a non-dict body) is NO_MATCH — the
        service answered "no such metabolite". Data is VOTED.

        MATCHED_NO_ID is the third answer: /match matched the name and the id lookup returned no id
        for the match. It is about the name, not the service, so ``_request_once`` RETURNS it (it is
        never raised) and the breaker does not count it. It used to be raised, which let three such
        names open the breaker and mark every other name in the process UNAVAILABLE.

        A per-name 404 (a name the Workbench does not know) is a NO_MATCH here, not an outage.
        ``_request_once`` classifies it as NO_MATCH rather than raising, so it never counts toward the
        circuit breaker, which is what used to trip on lipid panels and then skip RefMet /match for
        EVERY other name (and every other annotator instance) for the recovery window. Slash-bearing
        names are sent with a literal slash and resolve normally (see ``_request_once``).
        """
        if self._past_batch_deadline():
            # Past the shared batch deadline: mark UNAVAILABLE with NO network call.
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        try:
            data = self._do_refmet_request(metabolite_name)
        except CircuitBreakerError:
            self._note_breaker_skip(metabolite_name)
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch RefMet data for '{metabolite_name}': {e}")
            return RefMetResult(AVAILABILITY_UNAVAILABLE, source=REFMET_SOURCE_UNAVAILABLE)
        if data is None:
            return RefMetResult(AVAILABILITY_NO_MATCH, source=REFMET_SOURCE_LIVE)
        if isinstance(data, _MatchedNoId):
            logging.warning(
                "RefMet matched %r as %r but returned no refmet_id for it (%s; not a service failure)",
                metabolite_name,
                data.refmet_name,
                AVAILABILITY_MATCHED_NO_ID,
            )
            return RefMetResult(AVAILABILITY_MATCHED_NO_ID, source=REFMET_SOURCE_LIVE)
        return RefMetResult(AVAILABILITY_VOTED, data, source=REFMET_SOURCE_LIVE)

    @circuit(failure_threshold=3, recovery_timeout=300)
    def _do_refmet_request(self, metabolite_name: str) -> dict[str, Any] | _MatchedNoId | None:
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

    def _lookup_refmet_id(self, refmet_name: Any) -> str | None:  # noqa: ANN401 — raw JSON value
        """The RefMet id for a ``refmet_name`` /match returned, via ``/refmet/name/{name}/refmet_id``.

        Returns None when /name ANSWERED and has no id for this name: a 200 with no usable
        ``refmet_id`` (live RefMet sends an empty JSON array) or a per-name Bad Request / Not Found.
        That is an answer about the name, so it is returned, never raised, and the breaker does not
        count it; the caller reports MATCHED_NO_ID and casts no vote. Observed live on 2026-11-20 for
        every refmet_name containing an apostrophe (``5'-Methylthioadenosine``): /name returns ``[]``
        whether the apostrophe is sent literally or as ``%27``.

        A 5xx, a timeout or a transport error still raises: the service did not answer, and that is
        what the breaker is for.
        """
        if not isinstance(refmet_name, str) or not refmet_name.strip():
            raise requests.RequestException("RefMet /match matched without a refmet_name or refmet_id")
        # Its own timeout, re-capped by what is left of the batch deadline after /match answered, so
        # the backfill can never run past BATCH_DEADLINE_S.
        timeout = self.REQUEST_TIMEOUT_S
        if self._batch_deadline is not None:
            remaining = self._batch_deadline - self._clock()
            if remaining <= 0:
                raise requests.Timeout(f"batch deadline reached before the RefMet id lookup for {refmet_name!r}")
            timeout = max(0.1, min(timeout, remaining))
        url = f"{self.NAME_URL}/{quote_name_path_segment(refmet_name)}/refmet_id"
        response = self._session.get(url, timeout=timeout)
        if response.status_code in (400, 404):
            return None
        response.raise_for_status()
        body = response.json()
        refmet_id = body.get("refmet_id") if isinstance(body, dict) else None
        if not isinstance(refmet_id, str) or not refmet_id.strip() or refmet_id == "-":
            return None
        return refmet_id

    @staticmethod
    def _is_message_reply(data: dict[str, Any]) -> bool:
        """Whether a /match reply with no id is a message in ``refmet_name``, not a metabolite.

        /match answers some queries with advice instead of a record, for example ``C0`` ->
        ``'C' is ambiguous:Use 'FA'(fatty acids) or 'CAR'(acylcarnitines)*``. Such a reply describes
        no compound: every field except ``refmet_name`` is the "-" placeholder (formula, exact mass
        and the three classes), where a real match fills them in. The test needs at least one such
        field to be present, so a reply that carries a name and nothing else is not called a message.
        """
        others = [value for key, value in data.items() if key not in ("refmet_name", "refmet_id")]
        return bool(others) and all(value is None or str(value).strip() in ("", "-") for value in others)

    def _request_once(self, metabolite_name: str) -> dict[str, Any] | _MatchedNoId | None:
        """One HTTP attempt. Returns the data dict, None for a genuine no-match, or a ``_MatchedNoId``.

        The name goes through ``quote_name_path_segment``: a slash between lipid chain specs is sent
        literally (the Workbench resolves ``PC 16:0/18:1`` that way and 404s on ``%2F``), and any other
        slash is encoded, because sent literally the Workbench answers for the text before the slash
        alone (``mannitol/sorbitol`` -> Mannitol). Verified live 2026-10-21.

        A per-name client error (Bad Request or Not Found) is a definitive "no such entry", not a
        service outage: it is returned as the no-match sentinel (None) rather than raised, so it never
        counts toward the circuit breaker. Only 5xx, timeouts and transport errors should trip it.
        This is what stops an unknown name from tripping the breaker and degrading RefMet for every
        other name in the process.

        Two more answers are about the name and are likewise returned, never raised: a reply whose
        ``refmet_name`` is a message (a no-match, with no id lookup) and a match the id lookup has no
        id for (a ``_MatchedNoId``).
        """
        url = f"{self.BASE_URL}/{quote_name_path_segment(metabolite_name)}"

        # When a batch deadline is armed, cap this attempt's timeout by the remaining budget so a
        # lookup started just before the deadline cannot run a full timeout past it.
        timeout = self.REQUEST_TIMEOUT_S
        if self._batch_deadline is not None:
            timeout = max(0.1, min(timeout, self._batch_deadline - self._clock()))
        response = self._session.get(url, timeout=timeout)
        if response.status_code in (400, 404):
            # A path-addressed Bad-Request / Not-Found means "this name is not a match" for a
            # per-name registry. Classify it as a genuine no-match so it is never raised out of the
            # circuit-decorated call and never counted as a breaker failure.
            return None
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            # /match answers a no-match with "-" values. It used to carry refmet_id == "-"; since
            # 2026-11-10 it omits refmet_id entirely, so refmet_name == "-" marks the negative too.
            if data.get("refmet_id") == "-" or data.get("refmet_name") == "-":
                return None
            if not data.get("refmet_id"):
                if self._is_message_reply(data):
                    # refmet_name is advice ("'C' is ambiguous: ..."), not a metabolite: a no-match.
                    return None
                # A match without its id: backfill it from /name, keyed by the matched refmet_name.
                refmet_name = data.get("refmet_name")
                refmet_id = self._lookup_refmet_id(refmet_name)
                if refmet_id is None:
                    return _MatchedNoId(str(refmet_name))
                return {**data, "refmet_id": refmet_id}
            return data

        # A 200 whose body is not a dict (null, an array, a proxy-injected blob) is a DEGRADED
        # response, not RefMet's refmet_id == "-" negative. Surface it as UNAVAILABLE (raise ->
        # classified UNAVAILABLE by _fetch_refmet_data), never a genuine no-match.
        raise requests.RequestException(f"non-dict RefMet response for '{metabolite_name}'")
