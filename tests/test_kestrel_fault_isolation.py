"""Unit tests for Kestrel fault isolation and transport resilience mechanisms."""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from biomapper2.provenance import fetch_kg_build_info
from biomapper2.utils import (
    _CIRCUIT_BREAKER,
    CircuitBreaker,
    get_current_api_context,
    get_session,
    kestrel_request,
    reset_api_context,
    reset_circuit_breaker,
    reset_session,
    set_api_context,
)


@pytest.mark.unit
def test_circuit_breaker_trips_and_fast_fails() -> None:
    """Circuit breaker opens after reaching failure threshold and rejects further attempts."""
    threshold = 3
    cooldown = 10.0
    breaker = CircuitBreaker(failure_threshold=threshold, cooldown_s=cooldown)

    assert breaker.can_attempt() is True
    for _ in range(threshold - 1):
        breaker.record_failure()
        assert breaker.can_attempt() is True

    breaker.record_failure()
    assert breaker.can_attempt() is False


@pytest.mark.unit
def test_circuit_breaker_half_open_probe_and_recovery() -> None:
    """Circuit breaker allows single probe after cooldown and recovers on success."""
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.05)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.can_attempt() is False

    time.sleep(0.08)
    assert breaker.can_attempt() is True
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.can_attempt() is True


@pytest.mark.unit
def test_thread_local_session_isolation() -> None:
    """Requests sessions are distinct per thread and cleared completely by reset_session."""
    session_ids: set[int] = set()
    lock = threading.Lock()

    def worker() -> None:
        sess = get_session()
        with lock:
            session_ids.add(id(sess))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(session_ids) == 2
    reset_session()


@pytest.mark.unit
def test_api_request_context_propagation_and_timeout_override() -> None:
    """ApiRequestContext configures timeout settings and resets cleanly via token."""
    assert get_current_api_context() is None

    token = set_api_context(timeout_s=1.5, max_retries=0)
    try:
        ctx = get_current_api_context()
        assert ctx is not None
        assert ctx.timeout_s == 1.5
        assert ctx.max_retries == 0
    finally:
        reset_api_context(token)

    assert get_current_api_context() is None


@pytest.mark.unit
def test_kestrel_request_fast_retry_on_5xx() -> None:
    """kestrel_request performs fast retry on server error when API context is active."""
    token = set_api_context(timeout_s=1.0, max_retries=1, retry_backoff_base=0.01)
    call_count = 0

    def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        return resp

    reset_circuit_breaker()
    try:
        with patch("biomapper2.utils.get_session") as mock_get_sess:
            mock_sess = MagicMock()
            mock_sess.request.side_effect = mock_post
            mock_get_sess.return_value = mock_sess

            with pytest.raises(requests.HTTPError):
                kestrel_request(
                    method="POST",
                    endpoint="test-endpoint",
                    batch_field="items",
                    batch_items=["item1"],
                    batch_size=1,
                    json={},
                )

        assert call_count == 2
    finally:
        reset_api_context(token)
        reset_circuit_breaker()


@pytest.mark.unit
def test_provenance_fetch_kg_build_info_degrades_gracefully_under_fault() -> None:
    """fetch_kg_build_info falls back to unknown build info without raising exceptions."""
    token = set_api_context(timeout_s=0.5, max_retries=0)
    try:
        with patch("biomapper2.provenance.requests.get", side_effect=requests.Timeout("Mock hang")):
            raw_info, info = fetch_kg_build_info("http://127.0.0.1:9999")
            assert raw_info == "unknown"
            assert info.kg_version == "unknown"
            assert info.biolink_version == "unknown"
    finally:
        reset_api_context(token)


@pytest.mark.unit
def test_circuit_breaker_half_open_settles_every_exit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``HALF_OPEN`` exit path settles the breaker to ``closed`` or ``open``, never ``half_open``."""
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_COOLDOWN_S", "0.02")
    token = set_api_context(timeout_s=0.5, max_retries=0, retry_backoff_base=0.01, use_circuit_breaker=True)

    def _enter_ready_for_half_open() -> None:
        reset_circuit_breaker()
        _CIRCUIT_BREAKER.record_failure()
        _CIRCUIT_BREAKER.record_failure()
        assert _CIRCUIT_BREAKER.state == "open"
        time.sleep(0.03)

    def _http_resp(status_code: int, payload: Any = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.from_cache = False
        if status_code >= 400:
            resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}", response=resp)
        else:
            resp.raise_for_status.return_value = None
            resp.json.return_value = payload if payload is not None else {"items": {"a": []}}
        return resp

    def _call_once(side_effect: Any) -> None:
        mock_sess = MagicMock()
        mock_sess.request.side_effect = side_effect
        with patch("biomapper2.utils.get_session", return_value=mock_sess):
            kestrel_request(
                method="POST",
                endpoint="hybrid-search",
                batch_field="search_texts",
                batch_items=["a"],
                batch_size=1,
                json={},
            )

    try:
        # 1. Success (``200`` valid JSON) -> ``closed``
        _enter_ready_for_half_open()
        _call_once([_http_resp(200, {"a": []})])
        assert _CIRCUIT_BREAKER.state == "closed"

        # 2. Timeout -> ``open``
        _enter_ready_for_half_open()
        with pytest.raises(requests.Timeout):
            _call_once(requests.Timeout("read timed out"))
        assert _CIRCUIT_BREAKER.state == "open"

        # 3a. Non-``429`` ``4xx`` (``404``) -> ``closed``; repeated ``4xx`` in ``closed`` never trips ``open``
        _enter_ready_for_half_open()
        with pytest.raises(requests.HTTPError):
            _call_once([_http_resp(404)])
        assert _CIRCUIT_BREAKER.state == "closed"
        for _ in range(5):
            with pytest.raises(requests.HTTPError):
                _call_once([_http_resp(400)])
        assert _CIRCUIT_BREAKER.state == "closed"

        # 3b. ``429`` rate-limit response -> ``open``
        _enter_ready_for_half_open()
        with pytest.raises(requests.HTTPError):
            _call_once([_http_resp(429)])
        assert _CIRCUIT_BREAKER.state == "open"

        # 4. ``5xx`` server error -> ``open``
        _enter_ready_for_half_open()
        with pytest.raises(requests.HTTPError):
            _call_once([_http_resp(503)])
        assert _CIRCUIT_BREAKER.state == "open"

        # 5. Non-requests exception (e.g., cache sqlite error or bad JSON ValueError) -> ``open``
        _enter_ready_for_half_open()
        with pytest.raises(RuntimeError, match="sqlite cache failure"):
            _call_once(RuntimeError("sqlite cache failure"))
        assert _CIRCUIT_BREAKER.state == "open"

        # 6. Cancellation (``BaseException``) on both ``kestrel_request`` and ``fetch_kg_build_info`` -> ``open``
        _enter_ready_for_half_open()
        with pytest.raises(asyncio.CancelledError):
            _call_once(asyncio.CancelledError())
        assert _CIRCUIT_BREAKER.state == "open"

        _enter_ready_for_half_open()
        with (
            patch("biomapper2.provenance.requests.get", side_effect=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            fetch_kg_build_info("http://127.0.0.1:9999")
        assert _CIRCUIT_BREAKER.state == "open"

        # 7. Unsettled exits (cancellation / ``BaseException``) in ``closed`` do not increment failure count
        reset_circuit_breaker()
        assert _CIRCUIT_BREAKER.state == "closed"
        for _ in range(5):
            with pytest.raises(asyncio.CancelledError):
                _call_once(asyncio.CancelledError())
            with (
                patch("biomapper2.provenance.requests.get", side_effect=asyncio.CancelledError()),
                pytest.raises(asyncio.CancelledError),
            ):
                fetch_kg_build_info("http://127.0.0.1:9999")
        assert _CIRCUIT_BREAKER.state == "closed"
        assert _CIRCUIT_BREAKER._consecutive_failures == 0
        # One real failure afterward still leaves threshold=2 breaker ``closed`` (proving count stayed zero)
        with pytest.raises(requests.Timeout):
            _call_once(requests.Timeout("single real timeout"))
        assert _CIRCUIT_BREAKER.state == "closed"
    finally:
        reset_api_context(token)
        reset_circuit_breaker()
