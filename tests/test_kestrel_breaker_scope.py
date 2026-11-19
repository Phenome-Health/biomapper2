"""The shared Kestrel breaker must not be openable by one caller.

The breaker is process-wide. These tests pin that a single API request adds at most one failure,
that a plain internal-server-error reply (Kestrel rejecting one input) never counts, and that a
failed half-open probe is always settled back to OPEN even when its request was already counted.
"""

import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from biomapper2.api.routes.mapping import _run_with_api_context
from biomapper2.utils import (
    _CIRCUIT_BREAKER,
    BreakerScope,
    get_current_api_context,
    kestrel_request,
    reset_api_context,
    reset_circuit_breaker,
    set_api_context,
)


@pytest.fixture(autouse=True)
def _small_breaker(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_COOLDOWN_S", "0.05")
    reset_circuit_breaker()
    yield
    reset_circuit_breaker()


def _failing_session(status: int) -> MagicMock:
    def respond(*_args: Any, **_kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        return resp

    session = MagicMock()
    session.request.side_effect = respond
    return session


def _call_kestrel(session: MagicMock, scope: BreakerScope | None) -> None:
    token = set_api_context(timeout_s=1.0, max_retries=0, breaker_scope=scope)
    try:
        with patch("biomapper2.utils.get_session", return_value=session):
            with pytest.raises(requests.RequestException):
                kestrel_request("POST", "hybrid-search", "search_text", ["x"], 1, json={})
    finally:
        reset_api_context(token)


@pytest.mark.unit
def test_one_request_adds_at_most_one_failure() -> None:
    session = _failing_session(503)
    scope = BreakerScope()
    for _ in range(5):
        _call_kestrel(session, scope)
    assert _CIRCUIT_BREAKER.state == "closed"

    _call_kestrel(session, BreakerScope())
    assert _CIRCUIT_BREAKER.state == "open"


@pytest.mark.unit
def test_plain_500_never_counts_but_unavailable_statuses_do() -> None:
    for _ in range(5):
        _call_kestrel(_failing_session(500), BreakerScope())
    assert _CIRCUIT_BREAKER.state == "closed"

    for status in (503, 504):
        _call_kestrel(_failing_session(status), BreakerScope())
    assert _CIRCUIT_BREAKER.state == "open"


@pytest.mark.unit
def test_counted_request_still_settles_a_failed_half_open_probe() -> None:
    session = _failing_session(503)
    scope = BreakerScope()
    _call_kestrel(session, scope)
    _call_kestrel(session, BreakerScope())
    assert _CIRCUIT_BREAKER.state == "open"

    time.sleep(0.06)
    _call_kestrel(session, scope)  # already counted; this is the half-open probe
    assert _CIRCUIT_BREAKER.state == "open"
    assert _CIRCUIT_BREAKER.can_attempt() is False


@pytest.mark.unit
def test_plain_500_on_half_open_probe_closes_the_breaker() -> None:
    for _ in range(2):
        _call_kestrel(_failing_session(503), BreakerScope())
    assert _CIRCUIT_BREAKER.state == "open"

    time.sleep(0.06)
    _call_kestrel(_failing_session(500), BreakerScope())
    assert _CIRCUIT_BREAKER.state == "closed"


@pytest.mark.unit
def test_route_wrapper_activates_the_request_scope() -> None:
    scope = BreakerScope()
    ctx = _run_with_api_context(get_current_api_context, _breaker_scope=scope)
    assert ctx is not None
    assert ctx.breaker_scope is scope
    assert get_current_api_context() is None
