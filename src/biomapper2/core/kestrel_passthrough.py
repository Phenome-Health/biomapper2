"""Raw Kestrel search passthrough collector.

An opt-in side channel (``MappingOptions.kestrel_top_n``): return the top-N RAW rows Kestrel returned
for each search endpoint the pipeline ACTUALLY used, untouched, alongside the normal mapping result.

Two invariants define this module:

- **Selection is never touched.** The collector issues a DEDICATED ``limit=N`` call per endpoint
  (``fetch_strategy="separate_call"``); it never reads or mutates the selection/annotation/resolution
  path. Rows are returned exactly as Kestrel gave them — BEFORE the hybrid ``>=0.5`` filter,
  ``stable_result_order``, category guards, or re-ranking (R3).
- **It reads a RECORDED plan, it does not re-derive.** ``endpoints`` is the list of Kestrel endpoints
  the pipeline recorded as actually invoked for this entity (empty when it skipped). There is no
  ``_select_annotators`` re-derivation here, so an entity that made no Kestrel call yields ``[]`` (R6)
  and passthrough can never fire a call the baseline skipped.

A passthrough call failure is isolated: it is classified into an enumerated ``error`` code, the full
exception is logged server-side (never surfaced, so internal Kestrel wiring is not leaked), ``rows`` is
empty, and nothing propagates — a failed passthrough never turns a successful mapping into an error (R7).
"""

from __future__ import annotations

import logging

import requests

from ..api.models.responses import KestrelRequestParams, KestrelSearchResult
from ..config import KESTREL_BATCH_SIZE_SEARCH
from ..utils import BisectBudgetExceeded, kestrel_request

logger = logging.getLogger(__name__)

# The enumerated error classes surfaced on KestrelSearchResult.error.
_ErrorCode = str


def _classify_error(exc: Exception) -> _ErrorCode:
    """Map a raised exception to the enumerated ``error`` class (never str(exc), R7).

    ``kestrel_request`` RAISES on failure: ``raise_for_status`` re-raises 5xx/4xx as ``HTTPError``,
    connection/timeout as ``RequestException`` subclasses, ``response.json()`` raises ``ValueError``
    on a malformed body, and ``BisectBudgetExceeded`` fires when a 5xx storm exhausts the bisect budget.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.HTTPError):
        return "upstream_error"
    # requests' JSONDecodeError subclasses ValueError; a plain malformed body raises ValueError too.
    if isinstance(exc, ValueError):
        return "malformed_response"
    if isinstance(exc, BisectBudgetExceeded):
        # A budget exhausted by a 5xx storm is an upstream condition, not a client/parse error.
        return "upstream_error"
    if isinstance(exc, requests.exceptions.RequestException):
        return "upstream_error"
    return "other"


def collect(
    search_text: str,
    category: str,
    prefixes: list[str] | None,
    endpoints: list[str],
    n: int,
) -> list[KestrelSearchResult]:
    """Fetch the top-N raw rows for each RECORDED Kestrel endpoint (``separate_call``).

    Args:
        search_text: The term the pipeline searched (identical to the selection call's search text).
        category: Biolink category the selection call sent (row + cache-key parity).
        prefixes: Allowed CURIE prefixes the selection call sent (or None/empty for none).
        endpoints: Kestrel endpoints the pipeline RECORDED as used (e.g. ``["hybrid-search"]``);
            empty → ``[]`` (R6).
        n: Number of raw rows to return per endpoint (the request's ``kestrel_top_n``).

    Returns:
        One ``KestrelSearchResult`` per endpoint, in the recorded order. Never raises (R7).
    """
    results: list[KestrelSearchResult] = []
    for endpoint in endpoints:
        request_params = KestrelRequestParams(
            search_text=search_text,
            limit=n,
            category=category,
            prefix=list(prefixes) if prefixes else None,
        )
        # Byte-for-byte the selection payload shape (see kestrel_{text,vector,hybrid}._kestrel_*_search)
        # with limit=n, so the row set and the requests_cache key line up with what selection saw.
        payload: dict = {"limit": n, "category": category}
        if prefixes:
            payload["prefix"] = list(prefixes)
        try:
            raw = kestrel_request(
                method="POST",
                endpoint=endpoint,
                batch_field="search_text",
                batch_items=[search_text],
                batch_size=KESTREL_BATCH_SIZE_SEARCH,
                json=payload,
                # Provenance isolation: a response-only option must not move the shared request/cache/
                # retry/failure counters that benchmark manifests persist (request_counter_snapshot),
                # or enabling kestrel_top_n would make passthrough traffic indistinguishable from
                # mapping traffic. count_requests=False keeps selection's counters byte-identical.
                count_requests=False,
            )
            # Verbatim passthrough: the raw dicts Kestrel returned, untouched (no model round-trip →
            # no coercion, no null-fill, no dropped/added field). Truncated to N, Kestrel's order (R3).
            rows = list(raw.get(search_text) or [])[:n]
            results.append(
                KestrelSearchResult(
                    endpoint=endpoint,  # type: ignore[arg-type]
                    request=request_params,
                    rows=rows,
                    fetch_strategy="separate_call",
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 — passthrough failure must never escape (R7)
            error = _classify_error(exc)
            logger.warning(
                "kestrel passthrough failed on %s for %r (classified=%s)",
                endpoint,
                search_text,
                error,
                exc_info=True,
            )
            results.append(
                KestrelSearchResult(
                    endpoint=endpoint,  # type: ignore[arg-type]
                    request=request_params,
                    rows=[],
                    fetch_strategy="separate_call",
                    error=error,  # type: ignore[arg-type]
                )
            )
    return results
