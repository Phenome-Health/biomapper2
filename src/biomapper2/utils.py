"""
Utility functions for biomapper2.

Provides logging setup and mathematical helpers for metric calculations.
"""

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, TypeGuard
from urllib.parse import urlparse

import requests
import requests_cache

from .config import (
    CACHE_DIR,
    CACHE_IGNORED_PARAMETERS,
    KESTREL_BATCHING_ENABLED,
    KESTREL_BISECT_MAX_CONSECUTIVE_FAILURES,
    KESTREL_BISECT_MAX_REQUESTS,
    KESTREL_BISECT_MAX_RETRIES,
    KESTREL_BISECT_MAX_WALL_CLOCK_S,
    KESTREL_BISECT_MIN_INTER_REQUEST_DELAY_S,
    KESTREL_BISECT_ON_5XX_ENABLED,
    KESTREL_REQUEST_TIMEOUT_S,
    LOG_LEVEL,
    PUBLIC_KESTREL_API_URL,
    get_kestrel_api_key,
    get_kestrel_api_url,
)
from .models import AssignedIDsDict as AssignedIDsDict  # Re-export for backward compatibility

# Type hint for annotation mode
AnnotationMode = Literal["all", "missing", "none"]

VALIDATOR_PROP = "validator"
CLEANER_PROP = "cleaner"
ALIASES_PROP = "aliases"


# ------------------------------------------------------------------------------------------------
# One session for the process, built lazily
# ------------------------------------------------------------------------------------------------
# A fresh CachedSession per request means a new adapter, a new connection pool and no keep-alive on
# every call, which is the most plausible mechanical cause of the dropped connections seen on long
# runs. Lazy rather than module-level so the cache directory's import-time creation and any
# monkeypatching still work. Process-global state needs an explicit reset, which lives in
# tests/conftest.py -- without one, a test that primes this leaks into every later test.
_SESSION: "requests_cache.CachedSession | None" = None

# Per-entry expiry is measured from insertion, so a longer-lived session does not extend staleness.
_SESSION_EXPIRE_AFTER = timedelta(hours=1)


def get_session() -> "requests_cache.CachedSession":
    """The process's single cached session, built on first use."""
    global _SESSION
    if _SESSION is None:
        # ``_KestrelCachedSession`` and ``ignored_parameters`` are NOT optional here. Centralizing
        # construction moved the only call site that carried them, so building a plain
        # CachedSession would write the API key into the on-disk cache in cleartext again -- the
        # defect PR #50 fixed. Any change to this factory has to preserve both.
        _SESSION = _KestrelCachedSession(
            CACHE_DIR / "kestrel_http",
            expire_after=_SESSION_EXPIRE_AFTER,
            allowable_methods=["GET", "POST"],
            ignored_parameters=CACHE_IGNORED_PARAMETERS,
        )
    return _SESSION


def reset_session() -> None:
    """Drop the process's session. Called between tests, and available to a long-running driver."""
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:  # noqa: BLE001 - a failure to close must never fail the reset
            logging.debug("closing the cached session raised; dropping it anyway", exc_info=True)
    _SESSION = None


# ------------------------------------------------------------------------------------------------
# Per-endpoint request counters
# ------------------------------------------------------------------------------------------------
# Nothing counted server errors or dropped connections before this. The counts that circulated were
# read off a multi-megabyte log by hand and recount differently depending on what one decides to
# count, so the definitions live here in code and the totals ride in the run manifests.
#
# These are process-global. A suite runs every dataset in ONE process, so a counter without a
# per-dataset reset makes every dataset after the first cumulative and wrong.
_COUNTER_FIELDS = (
    "requests",
    "retries",
    "terminal_5xx",
    "transient_errors",
    "bisect_isolated",
    "from_cache_hits",
    "from_cache_misses",
)

_REQUEST_COUNTERS: dict[str, dict[str, int]] = {}


def _bump(endpoint: str, field_name: str, amount: int = 1) -> None:
    bucket = _REQUEST_COUNTERS.setdefault(endpoint, dict.fromkeys(_COUNTER_FIELDS, 0))
    bucket[field_name] += amount


def reset_request_counters() -> None:
    """Zero every endpoint's counters. Call once per dataset, not once per suite."""
    _REQUEST_COUNTERS.clear()


def request_counter_snapshot() -> dict[str, dict[str, int]]:
    """A deep copy of the counters, safe to embed in a manifest."""
    return {endpoint: dict(counts) for endpoint, counts in _REQUEST_COUNTERS.items()}


# ------------------------------------------------------------------------------------------------
# Bisect-on-5xx budgets
# ------------------------------------------------------------------------------------------------
class BisectBudgetExceeded(RuntimeError):
    """A bisect budget was exhausted. Always raised, never swallowed.

    Bisecting is only a sane response to a payload-determined failure. Under a load or transient
    condition it degenerates into a retry storm against a shared service, and the difference is
    visible precisely as budget exhaustion. Failing loud here is the point.
    """


@dataclass
class BisectBudget:
    """Caps on what one bisect may spend, counted in requests rather than recursion depth.

    Depth is bounded near ten by construction and bounds nothing that matters. Volume is what a
    shared, unrated service notices, and volume is what bisect changes.
    """

    max_requests: int = KESTREL_BISECT_MAX_REQUESTS
    max_wall_clock_s: float = KESTREL_BISECT_MAX_WALL_CLOCK_S
    max_consecutive_failures: int = KESTREL_BISECT_MAX_CONSECUTIVE_FAILURES
    min_inter_request_delay_s: float = KESTREL_BISECT_MIN_INTER_REQUEST_DELAY_S
    # Live spend, reset at the start of each bisect so one dataset cannot inherit another's.
    requests_spent: int = field(default=0, init=False)
    consecutive_failures: int = field(default=0, init=False)
    started_monotonic: float | None = field(default=None, init=False)
    isolated_items: list = field(default_factory=list, init=False)

    def start(self) -> None:
        self.requests_spent = 0
        self.consecutive_failures = 0
        self.started_monotonic = time.monotonic()
        self.isolated_items = []

    def charge(self) -> None:
        if self.started_monotonic is None:
            self.start()
        self.requests_spent += 1
        if self.requests_spent > self.max_requests:
            raise BisectBudgetExceeded(
                f"bisect exceeded its request budget ({self.max_requests}); the failure is not "
                f"behaving like a payload defect, so bisecting is amplifying it rather than "
                f"isolating it. Stopping instead of continuing to load a shared service."
            )
        assert self.started_monotonic is not None
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > self.max_wall_clock_s:
            raise BisectBudgetExceeded(
                f"bisect exceeded its wall-clock budget ({self.max_wall_clock_s}s) after "
                f"{self.requests_spent} request(s)"
            )

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures > self.max_consecutive_failures:
            raise BisectBudgetExceeded(
                f"bisect saw {self.consecutive_failures} consecutive failures, above the cap of "
                f"{self.max_consecutive_failures}; failures unrelated to payload content mean "
                f"bisecting is a retry storm, not a diagnosis"
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0


def chunk_list(items: list, chunk_size: int) -> Iterator[list]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Yields:
        List chunks of at most chunk_size items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def setup_logging():
    """Configure logging based on LOG_LEVEL in config.py."""
    if not logging.getLogger().hasHandlers():  # Skip setup if it's already been done
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level = LOG_LEVEL.upper()

        if level not in valid_levels:
            print(f"Invalid log level '{LOG_LEVEL}', defaulting to INFO")
            level = "INFO"

        logging.basicConfig(
            level=getattr(logging, level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def text_is_not_empty(value: Any) -> TypeGuard[str]:
    """Check if a name/text field value is a valid non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def to_list(item: Any) -> list[Any]:
    if item is None:
        return []
    elif isinstance(item, list):
        return item
    elif isinstance(item, (str, int, float)):
        return [item]
    else:
        return list(item)


def to_set(item: Any) -> set[Any]:
    if item is None:
        return set()
    elif isinstance(item, set):
        return item
    elif isinstance(item, (str, int, float)):
        return {item}
    else:
        return set(item)


def safe_divide(numerator, denominator) -> float | None:
    """
    Divide two numbers, returning None if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value

    Returns:
        Result of division, or None if denominator is zero
    """
    # Cast to float to handle potential numpy types
    numerator = float(numerator)
    denominator = float(denominator)

    if denominator == 0.0:
        # Return None, which will be serialized as 'null' in JSON.
        # This is more accurate for metrics like 'accuracy'
        # where a 0 denominator means 'not applicable'.
        return None

    result = numerator / denominator
    return result


_key_withheld_warned = False

# Credential header name, shared by request construction and the redirect scrubber below.
_KESTREL_KEY_HEADER = "X-API-Key"


class _KestrelCachedSession(requests_cache.CachedSession):
    """Cached session that drops the Kestrel credential on a cross-origin redirect.

    ``requests`` strips only ``Authorization`` when a redirect changes origin (see
    ``SessionRedirectMixin.rebuild_auth``); a custom credential header is replayed verbatim to the
    new host. So a 301 from the internal endpoint to the public one — the obvious way to retire the
    internal host — would hand the internal key to a third party even though
    :func:`kestrel_host_accepts_credentials` cleared the *configured* URL. Scrub it at the transport layer,
    where the actual destination is known.
    """

    def rebuild_auth(self, prepared_request, response):  # type: ignore[no-untyped-def]
        super().rebuild_auth(prepared_request, response)
        if response.request.url and self.should_strip_auth(response.request.url, prepared_request.url):
            prepared_request.headers.pop(_KESTREL_KEY_HEADER, None)


def _normalized_host(url: str) -> str | None:
    """Lowercased hostname with any FQDN trailing dot removed, or None if there is no host.

    ``urlparse`` already lowercases, but ``https://host./api`` and ``https://host/api`` are the same
    destination while comparing unequal as strings, so the trailing dot has to go too.
    """
    host = urlparse(url).hostname
    return host.rstrip(".") if host else None


def kestrel_host_accepts_credentials(url: str) -> bool:
    """False for the public Kestrel, which needs no key and must never receive one.

    Compared by normalized hostname, so none of a trailing slash, a different path, a trailing FQDN
    dot, or an embedded ``user@`` can smuggle the credential to the public host.

    Fails CLOSED: a URL with no parseable host (e.g. a missing scheme) withholds the key rather than
    sending it, since we cannot prove where it would go.
    """
    host = _normalized_host(url)
    if host is None:
        return False
    return host != _normalized_host(PUBLIC_KESTREL_API_URL)


def _warn_key_withheld_once(url: str) -> None:
    """Say once that a configured key is deliberately not being sent to the public endpoint.

    Takes the URL the decision was actually made against. Reading the import-time constant here
    would name a different host than the one the key was withheld from whenever the URL was
    overridden after import.
    """
    global _key_withheld_warned
    if _key_withheld_warned:
        return
    _key_withheld_warned = True
    logging.warning(
        f"KESTREL_API_KEY is set but KESTREL_API_URL points at the public endpoint "
        f"({url}), which needs no key — withholding the credential rather than "
        "sending it to a public host. Unset KESTREL_API_KEY, or set KESTREL_API_URL to the "
        "endpoint the key was issued for."
    )


def bulk_kestrel_request(
    method: str,
    endpoint: str,
    session: requests.Session | None = None,
    auth_required: bool = True,
    *,
    max_retries: int = 3,
    retry_backoff_base: float = 2.0,
    **kwargs,
) -> Any:
    """
    Make a single Kestrel API request with the full payload.

    This is the low-level function that sends one request. For batching support,
    use kestrel_request() instead.

    Transient server-side failures (HTTP 5xx) and connection/timeout errors are
    retried with exponential backoff; client errors (4xx) are raised immediately
    since they will not self-heal. A single Kestrel blip therefore no longer kills
    a long batch run.

    Args:
        method: HTTP method ('GET' or 'POST')
        endpoint: API endpoint path
        session: Optional requests session (defaults to cached session)
        auth_required: Whether to include the API key header. When False, the
            request is sent without authentication (e.g. for GET /categories).
            Default is True to preserve existing behavior.
        max_retries: Number of retries on transient errors (5xx / connection /
            timeout) before giving up. 0 disables retrying.
        retry_backoff_base: Base for exponential backoff; the delay before retry
            ``attempt`` (0-indexed) is ``retry_backoff_base ** attempt`` seconds.
        **kwargs: Additional arguments to pass to requests (json, params, etc.)

    Returns:
        JSON response from API

    Raises:
        requests.exceptions.HTTPError: If API returns error status (after retries
            for 5xx; immediately for 4xx)
        requests.exceptions.RequestException: If request fails after retries
    """
    # Sort search_text in json payload for consistent cache keys (if handling a batch)
    if "json" in kwargs and isinstance(kwargs["json"], dict):
        payload = kwargs["json"]
        if "search_text" in payload and isinstance(payload["search_text"], list):
            payload["search_text"].sort()

    if session is None:
        # The inline construction this replaced carried the cache-redaction arguments. They now
        # live in ``get_session``; see the note there -- dropping them re-opens PR #50's defect.
        session = get_session()

    # A default, not plumbing: the kwarg already forwarded to the transport, but the mapping-path
    # callers supply none, so without this a wedged request has no timeout at all. See
    # ``config.KESTREL_REQUEST_TIMEOUT_S`` for how the value is sized.
    kwargs.setdefault("timeout", KESTREL_REQUEST_TIMEOUT_S)

    # Resolved per call rather than from the import-time KESTREL_API_URL constant: the
    # --kestrel-url pytest option used by kg-regression sets os.environ AFTER this module is
    # imported, so a captured constant would silently ignore the override and the regression job
    # would report on whichever backend happened to be the default.
    #
    # The credential decision below reads this SAME resolved value. Deciding against the import-time
    # constant while sending to the resolved URL is how the key reaches a host it was withheld from:
    # override to the public endpoint after import and the constant still names the internal host,
    # so the guard says "accepts credentials" about a request going somewhere else entirely.
    base_url = get_kestrel_api_url()

    headers: dict[str, str] = {}
    api_key = get_kestrel_api_key() if auth_required else None
    if api_key:
        if kestrel_host_accepts_credentials(base_url):
            headers[_KESTREL_KEY_HEADER] = api_key
        else:
            _warn_key_withheld_once(base_url)

    url = f"{base_url}/{endpoint}"
    transient = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
    for attempt in range(max_retries + 1):
        try:
            _bump(endpoint, "requests")
            response = session.request(method, url, headers=headers, **kwargs)
            # Cache hit-vs-miss belongs in the manifest: a repeat run that replays a large cache
            # reports a flip rate near zero and would publish "the backend is perfectly stable"
            # from an instrument that cannot observe its own failure mode.
            _bump(endpoint, "from_cache_hits" if getattr(response, "from_cache", False) else "from_cache_misses")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Retry only transient server errors (5xx); 4xx (auth, bad payload) won't self-heal.
            if status is not None and 500 <= status < 600 and attempt < max_retries:
                delay = retry_backoff_base**attempt
                _bump(endpoint, "retries")
                logging.warning(
                    f"Kestrel API {status} on {endpoint} "
                    f"(attempt {attempt + 1}/{max_retries + 1}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            if status is not None and 500 <= status < 600:
                _bump(endpoint, "terminal_5xx")
            # Remediation hint for the keyless-default misconfiguration, appended rather than
            # branched so the auth path keeps the exception text and traceback.
            hint = ""
            if status in (401, 403) and not api_key:
                hint = (
                    f" No KESTREL_API_KEY is set and {base_url} requires authentication; "
                    f"set it, or point KESTREL_API_URL at the public endpoint "
                    f"({PUBLIC_KESTREL_API_URL}), which needs no key."
                )
            logging.error(f"Kestrel API HTTP error ({endpoint}): {e}{hint}", exc_info=True)
            raise
        except transient as e:
            if attempt < max_retries:
                delay = retry_backoff_base**attempt
                _bump(endpoint, "retries")
                logging.warning(
                    f"Kestrel API transient error on {endpoint} ({type(e).__name__}) "
                    f"(attempt {attempt + 1}/{max_retries + 1}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            _bump(endpoint, "transient_errors")
            logging.error(f"Kestrel API request failed ({endpoint}): {e}", exc_info=True)
            raise
        except requests.exceptions.RequestException as e:
            _bump(endpoint, "transient_errors")
            logging.error(f"Kestrel API request failed ({endpoint}): {e}", exc_info=True)
            raise


def _record_poison(path: "Path | str | None", endpoint: str, batch_field: str, item: Any) -> None:
    """Append one isolated payload to a run-local file.

    This is the deliverable that turns a failed run into a bug report the upstream team can act on:
    the exact item, not merely the fact that something in a chunk of a thousand was rejected.
    """
    _bump(endpoint, "bisect_isolated")
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"endpoint": endpoint, "batch_field": batch_field, "item": item}) + "\n")


def _bisect_chunk(
    method: str,
    endpoint: str,
    batch_field: str,
    chunk: list,
    json_payload: dict,
    session: requests.Session | None,
    budget: "BisectBudget",
    poison_log_path: "Path | str | None",
    **kwargs,
) -> dict:
    """Halve a failing chunk until the rejected items are isolated, or a budget stops the attempt.

    Sub-chunks are contiguous slices of the already-sorted chunk, so a rerun produces the same
    sub-chunks and therefore the same cache keys. Server errors are never cached (the cache accepts
    only successful responses), so bisect and the cache do not fight.

    A single-item chunk that still fails is the isolated payload: it is recorded and dropped, and
    the surrounding rows are returned. Under a load or transient condition no single item is at
    fault, splitting eventually succeeds everywhere or the budgets fire -- which is exactly the
    signal that distinguishes the hypotheses.
    """
    # The caller's retry ladder is deliberately discarded here: bisect composes with it
    # multiplicatively, so one rejected item becomes dozens of nodes times four attempts each,
    # plus minutes of backoff sleep, against a service that is already returning errors.
    kwargs.pop("max_retries", None)
    kwargs.pop("retry_backoff_base", None)
    if budget.min_inter_request_delay_s > 0:
        time.sleep(budget.min_inter_request_delay_s)
    budget.charge()
    payload = {**json_payload, batch_field: chunk}
    try:
        result = bulk_kestrel_request(
            method,
            endpoint,
            session=session,
            json=payload,
            max_retries=KESTREL_BISECT_MAX_RETRIES,
            **kwargs,
        )
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status is None or not (500 <= status < 600):
            raise  # a client error will not self-heal and is not what bisect is for
        return _bisect_failed_chunk(
            method, endpoint, batch_field, chunk, json_payload, session, budget, poison_log_path, **kwargs
        )
    budget.record_success()
    return result if isinstance(result, dict) else {}


def _bisect_failed_chunk(
    method: str,
    endpoint: str,
    batch_field: str,
    chunk: list,
    json_payload: dict,
    session: requests.Session | None,
    budget: "BisectBudget",
    poison_log_path: "Path | str | None",
    **kwargs,
) -> dict:
    """Handle a chunk already KNOWN to have failed: isolate it, or split and descend.

    Separated from :func:`_bisect_chunk` so a chunk whose failure has already been observed is
    never resubmitted verbatim. Resubmitting it wastes a request against a service that has just
    returned an error, and on a flaky backend it can succeed by luck, which would end the bisect
    with the wrong conclusion.
    """
    budget.record_failure()
    if len(chunk) <= 1:
        item = chunk[0] if chunk else None
        budget.isolated_items.append(item)
        logging.warning(f"bisect isolated a rejected payload on {endpoint}: {item!r}")
        _record_poison(poison_log_path, endpoint, batch_field, item)
        return {}
    mid = len(chunk) // 2
    merged: dict = {}
    for half in (chunk[:mid], chunk[mid:]):
        merged.update(
            _bisect_chunk(
                method,
                endpoint,
                batch_field,
                half,
                json_payload,
                session,
                budget,
                poison_log_path,
                **kwargs,
            )
        )
    return merged


def kestrel_request(
    method: str,
    endpoint: str,
    batch_field: str,
    batch_items: list,
    batch_size: int,
    session: requests.Session | None = None,
    *,
    bisect_on_5xx: bool | None = None,
    budget: "BisectBudget | None" = None,
    poison_log_path: "Path | str | None" = None,
    **kwargs,
) -> dict:
    """
    Make Kestrel API requests with automatic batching for large payloads.

    Splits batch_items into chunks, makes separate API calls for each chunk,
    and merges the results. Assumes API returns dict keyed by input items.

    When KESTREL_BATCHING_ENABLED is False, sends all items in a single request
    (useful for performance testing).

    Args:
        method: HTTP method ('GET' or 'POST')
        endpoint: API endpoint path
        batch_field: JSON field name for batch items (e.g., 'search_text', 'curies')
        batch_items: List of items to batch
        batch_size: Maximum items per request (ignored if batching disabled)
        session: Optional requests session
        **kwargs: Additional arguments (json, params, etc.)

    Returns:
        Merged dict of results from all batches
    """
    if not batch_items:
        return {}

    json_payload = kwargs.pop("json", {})

    # If batching is disabled, send all items in a single request
    if not KESTREL_BATCHING_ENABLED:
        full_payload = {**json_payload, batch_field: batch_items}
        result = bulk_kestrel_request(method, endpoint, session=session, json=full_payload, **kwargs)
        return result if isinstance(result, dict) else {}

    # Batch the request
    chunks = list(chunk_list(batch_items, batch_size))
    num_chunks = len(chunks)

    if num_chunks > 1:
        logging.info(f"Batching {len(batch_items)} items into {num_chunks} chunks of {batch_size} for {endpoint}")

    # This is the only seam where bisecting is possible: the lower function receives an opaque
    # payload and does not know which key holds the batch, and its return is not necessarily a
    # dict. Here we have the batch field, the chunk list and the merge.
    if bisect_on_5xx is None:
        bisect_on_5xx = KESTREL_BISECT_ON_5XX_ENABLED

    merged_results: dict = {}
    chunk_budget: BisectBudget | None = None  # started at most once, on the first failing chunk
    for chunk in chunks:
        chunk_payload = {**json_payload, batch_field: chunk}
        try:
            chunk_results = bulk_kestrel_request(method, endpoint, session=session, json=chunk_payload, **kwargs)
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if not bisect_on_5xx or status is None or not (500 <= status < 600):
                raise
            # Start the budget ONCE per call, not once per failing chunk. ``start()`` zeroes
            # requests_spent, consecutive_failures, started_monotonic AND isolated_items, so
            # restarting it here made the documented per-dataset cap a PER-CHUNK cap -- a dataset
            # with twelve failing chunks could spend twelve times the request and wall-clock
            # budget against a shared public service -- and left ``isolated_items`` holding only
            # the last failing chunk's isolates. Dormant while the flag is off; this is a load
            # promise made to somebody else's service, so it is fixed before it can be enabled.
            if chunk_budget is None:
                chunk_budget = budget if budget is not None else BisectBudget()
                chunk_budget.start()
            chunk_results = _bisect_failed_chunk(
                method,
                endpoint,
                batch_field,
                list(chunk),
                json_payload,
                session,
                chunk_budget,
                poison_log_path,
                **kwargs,
            )

        if isinstance(chunk_results, dict):
            merged_results.update(chunk_results)

    return merged_results
