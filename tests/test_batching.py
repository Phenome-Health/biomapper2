"""Tests for Kestrel API batching functionality."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from biomapper2.config import KESTREL_BATCH_SIZE_CANONICALIZE, KESTREL_BATCH_SIZE_SEARCH, KESTREL_BATCHING_ENABLED
from biomapper2.utils import bulk_kestrel_request, chunk_list, kestrel_request


def _fake_response(status: int, json_body: dict | None = None) -> MagicMock:
    """A mock requests.Response whose raise_for_status mirrors real 4xx/5xx behavior."""
    resp = MagicMock()
    resp.status_code = status
    if status >= 400:
        err = requests.exceptions.HTTPError(f"{status} Error")
        err.response = resp
        resp.raise_for_status.side_effect = err
    else:
        resp.raise_for_status.return_value = None
        resp.json.return_value = json_body if json_body is not None else {"ok": True}
    return resp


def test_bulk_kestrel_request_retries_5xx_then_succeeds():
    """A transient 5xx is retried with backoff and the eventual 200 is returned."""
    session = MagicMock()
    session.request.side_effect = [_fake_response(500), _fake_response(503), _fake_response(200, {"ok": 1})]
    with patch("biomapper2.utils.time.sleep") as sleep_mock:
        out = bulk_kestrel_request(
            "POST", "text-search", session=session, auth_required=False, max_retries=3, json={"search_text": ["x"]}
        )
    assert out == {"ok": 1}
    assert session.request.call_count == 3  # 2 failures + 1 success
    assert sleep_mock.call_count == 2  # backed off before each retry


def test_bulk_kestrel_request_retries_connection_error():
    """A transient connection error is retried, not raised on the first blip."""
    session = MagicMock()
    session.request.side_effect = [requests.exceptions.ConnectionError("boom"), _fake_response(200, {"ok": 2})]
    with patch("biomapper2.utils.time.sleep"):
        out = bulk_kestrel_request("POST", "text-search", session=session, auth_required=False, json={})
    assert out == {"ok": 2}
    assert session.request.call_count == 2


def test_bulk_kestrel_request_does_not_retry_4xx():
    """Client errors (4xx) will not self-heal, so they raise immediately without retry."""
    session = MagicMock()
    session.request.side_effect = [_fake_response(422)]
    with patch("biomapper2.utils.time.sleep"):
        with pytest.raises(requests.exceptions.HTTPError):
            bulk_kestrel_request("POST", "text-search", session=session, auth_required=False, max_retries=3, json={})
    assert session.request.call_count == 1  # no retry on a 4xx


def test_bulk_kestrel_request_raises_after_exhausting_retries():
    """A persistent 5xx raises after 1 + max_retries attempts (the overnight-outage case)."""
    session = MagicMock()
    session.request.side_effect = [_fake_response(500) for _ in range(4)]
    with patch("biomapper2.utils.time.sleep"):
        with pytest.raises(requests.exceptions.HTTPError):
            bulk_kestrel_request("POST", "text-search", session=session, auth_required=False, max_retries=3, json={})
    assert session.request.call_count == 4  # 1 initial + 3 retries


pytestmark = pytest.mark.unit


def test_batch_size_constants_exist():
    """Verify batch size constants are defined with reasonable defaults."""
    assert isinstance(KESTREL_BATCH_SIZE_SEARCH, int)
    assert isinstance(KESTREL_BATCH_SIZE_CANONICALIZE, int)
    assert KESTREL_BATCH_SIZE_SEARCH > 0
    assert KESTREL_BATCH_SIZE_CANONICALIZE > 0


def test_batch_size_defaults():
    """Verify default batch sizes match expected values."""
    assert KESTREL_BATCH_SIZE_SEARCH == 1000
    assert KESTREL_BATCH_SIZE_CANONICALIZE == 2000


def test_batching_enabled_by_default():
    """Verify batching is enabled by default."""
    assert KESTREL_BATCHING_ENABLED is True


def test_chunk_list_empty():
    """Empty list returns empty list of chunks."""
    assert list(chunk_list([], 10)) == []


def test_chunk_list_smaller_than_chunk_size():
    """List smaller than chunk size returns single chunk."""
    items = [1, 2, 3]
    chunks = list(chunk_list(items, 10))
    assert chunks == [[1, 2, 3]]


def test_chunk_list_exact_multiple():
    """List that's exact multiple of chunk size."""
    items = [1, 2, 3, 4, 5, 6]
    chunks = list(chunk_list(items, 3))
    assert chunks == [[1, 2, 3], [4, 5, 6]]


def test_chunk_list_with_remainder():
    """List with remainder after chunking."""
    items = [1, 2, 3, 4, 5, 6, 7]
    chunks = list(chunk_list(items, 3))
    assert chunks == [[1, 2, 3], [4, 5, 6], [7]]


def test_chunk_list_chunk_size_one():
    """Chunk size of 1 returns each item as separate chunk."""
    items = ["a", "b", "c"]
    chunks = list(chunk_list(items, 1))
    assert chunks == [["a"], ["b"], ["c"]]


def test_kestrel_request_small_payload_no_chunking():
    """Small payloads should make single request without chunking."""
    with patch("biomapper2.utils.bulk_kestrel_request") as mock_request:
        mock_request.return_value = {"term1": [{"id": "A"}], "term2": [{"id": "B"}]}

        result = kestrel_request(
            method="POST",
            endpoint="text-search",
            batch_field="search_text",
            batch_items=["term1", "term2"],
            batch_size=1000,
            json={"limit": 10},
        )

        # Single call with all items
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["json"]["search_text"] == ["term1", "term2"]
        assert result == {"term1": [{"id": "A"}], "term2": [{"id": "B"}]}


def test_kestrel_request_large_payload_chunks():
    """Large payloads should be chunked into multiple requests."""
    with patch("biomapper2.utils.bulk_kestrel_request") as mock_request:
        # Simulate responses for each chunk
        mock_request.side_effect = [
            {"term1": [{"id": "A"}], "term2": [{"id": "B"}]},
            {"term3": [{"id": "C"}]},
        ]

        result = kestrel_request(
            method="POST",
            endpoint="text-search",
            batch_field="search_text",
            batch_items=["term1", "term2", "term3"],
            batch_size=2,  # Force chunking
            json={"limit": 10},
        )

        # Two calls due to chunking
        assert mock_request.call_count == 2
        # Results merged
        assert result == {"term1": [{"id": "A"}], "term2": [{"id": "B"}], "term3": [{"id": "C"}]}


def test_kestrel_request_empty_items():
    """Empty batch items should return empty dict without API call."""
    with patch("biomapper2.utils.bulk_kestrel_request") as mock_request:
        result = kestrel_request(
            method="POST",
            endpoint="text-search",
            batch_field="search_text",
            batch_items=[],
            batch_size=1000,
            json={"limit": 10},
        )

        mock_request.assert_not_called()
        assert result == {}


def test_kestrel_request_disabled_sends_single_request():
    """When batching disabled, large payloads should make single request."""
    with patch("biomapper2.utils.KESTREL_BATCHING_ENABLED", False):
        with patch("biomapper2.utils.bulk_kestrel_request") as mock_request:
            mock_request.return_value = {
                "term1": [{"id": "A"}],
                "term2": [{"id": "B"}],
                "term3": [{"id": "C"}],
            }

            result = kestrel_request(
                method="POST",
                endpoint="text-search",
                batch_field="search_text",
                batch_items=["term1", "term2", "term3"],
                batch_size=2,  # Would normally force chunking
                json={"limit": 10},
            )

            # Single call with all items (batching disabled)
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["json"]["search_text"] == ["term1", "term2", "term3"]
            assert result == {"term1": [{"id": "A"}], "term2": [{"id": "B"}], "term3": [{"id": "C"}]}


def test_linker_get_kg_ids_uses_kestrel_request():
    """Linker.get_kg_ids should use kestrel_request."""
    from biomapper2.core.linker import Linker

    with patch("biomapper2.core.linker.kestrel_request") as mock_kestrel:
        mock_kestrel.return_value = {"CHEBI:123": "n001", "PUBCHEM.COMPOUND:456": "n002"}

        result = Linker.get_kg_ids(["CHEBI:123", "PUBCHEM.COMPOUND:456"])

        mock_kestrel.assert_called_once()
        call_args = mock_kestrel.call_args
        assert call_args[1]["endpoint"] == "canonicalize"
        assert call_args[1]["batch_field"] == "curies"
        assert set(call_args[1]["batch_items"]) == {"CHEBI:123", "PUBCHEM.COMPOUND:456"}
        assert result == {"CHEBI:123": "n001", "PUBCHEM.COMPOUND:456": "n002"}


def test_kestrel_text_annotator_uses_kestrel_request():
    """KestrelTextSearchAnnotator should use kestrel_request."""
    from biomapper2.core.annotators.kestrel_text import KestrelTextSearchAnnotator

    with patch("biomapper2.core.annotators.kestrel_text.kestrel_request") as mock_kestrel:
        mock_kestrel.return_value = {"glucose": [{"id": "CHEBI:123", "score": 0.9}]}

        KestrelTextSearchAnnotator._kestrel_text_search(
            search_text=["glucose"],
            category="biolink:SmallMolecule",
            prefixes=None,
            limit=1,
        )

        mock_kestrel.assert_called_once()
        call_args = mock_kestrel.call_args
        assert call_args[1]["endpoint"] == "text-search"
        assert call_args[1]["batch_field"] == "search_text"


def test_kestrel_vector_annotator_uses_kestrel_request():
    """KestrelVectorSearchAnnotator should use kestrel_request."""
    from biomapper2.core.annotators.kestrel_vector import KestrelVectorSearchAnnotator

    with patch("biomapper2.core.annotators.kestrel_vector.kestrel_request") as mock_kestrel:
        mock_kestrel.return_value = {"glucose": [{"id": "CHEBI:123", "score": 0.8}]}

        KestrelVectorSearchAnnotator._kestrel_vector_search(
            search_text=["glucose"],
            category="biolink:SmallMolecule",
            prefixes=None,
            limit=1,
        )

        mock_kestrel.assert_called_once()
        call_args = mock_kestrel.call_args
        assert call_args[1]["endpoint"] == "vector-search"
        assert call_args[1]["batch_field"] == "search_text"


def test_kestrel_hybrid_annotator_uses_kestrel_request():
    """KestrelHybridSearchAnnotator should use kestrel_request."""
    from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator

    with patch("biomapper2.core.annotators.kestrel_hybrid.kestrel_request") as mock_kestrel:
        mock_kestrel.return_value = {"glucose": [{"id": "CHEBI:123", "score": 2.5}]}

        KestrelHybridSearchAnnotator._kestrel_hybrid_search(
            search_text=["glucose"],
            category="biolink:SmallMolecule",
            prefixes=None,
            limit=1,
        )

        mock_kestrel.assert_called_once()
        call_args = mock_kestrel.call_args
        assert call_args[1]["endpoint"] == "hybrid-search"
        assert call_args[1]["batch_field"] == "search_text"


def test_kestrel_request_logs_chunking(caplog):
    """kestrel_request should log when chunking occurs."""
    with patch("biomapper2.utils.bulk_kestrel_request") as mock_request:
        mock_request.side_effect = [
            {"term1": [], "term2": []},
            {"term3": []},
        ]

        with caplog.at_level(logging.INFO):
            kestrel_request(
                method="POST",
                endpoint="text-search",
                batch_field="search_text",
                batch_items=["term1", "term2", "term3"],
                batch_size=2,
                json={},
            )

        assert "Batching 3 items into 2 chunks" in caplog.text
