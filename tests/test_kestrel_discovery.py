"""Unit tests for kestrel_discovery module (mocked Kestrel)."""

from pathlib import Path
from unittest.mock import patch

import requests

from biomapper2.api.kestrel_discovery import (
    STATIC_FALLBACK,
    derive_all_presets,
    derive_presets_with_fallback,
    load_from_disk,
    save_to_disk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_CATEGORIES = ["biolink:SmallMolecule", "biolink:Protein", "biolink:Gene"]

# 20 nodes total: CHEBI in 16 (80%), HMDB in 14 (70%), RARE in 1 (5% -- at threshold, NOT >5%).
MOCK_NODES_SMALL_MOL: list[dict] = (
    [{"id": f"CHEBI:{i}", "score": 0.9, "prefixes": ["CHEBI", "HMDB"]} for i in range(12)]
    + [{"id": f"CHEBI:{i}", "score": 0.8, "prefixes": ["CHEBI"]} for i in range(12, 16)]
    + [{"id": f"HMDB:{i}", "score": 0.7, "prefixes": ["HMDB"]} for i in range(100, 102)]
    + [{"id": "COMMON:1", "score": 0.6, "prefixes": ["COMMON"]}]
    + [{"id": "RARE:1", "score": 0.5, "prefixes": ["RARE"]}]
)

MOCK_NODES_PROTEIN: list[dict] = [{"id": f"PR:{i}", "score": 0.9, "prefixes": ["PR", "UniProtKB"]} for i in range(5)]

ALIASES_FIXTURE: dict[str, str] = {
    "metabolite": "biolink:SmallMolecule",
    "protein": "biolink:Protein",
}


def _mock_bulk_request(method: str, endpoint: str, auth_required: bool = True, **kwargs) -> list | dict:
    """Side-effect for mocking bulk_kestrel_request."""
    if endpoint == "categories":
        return MOCK_CATEGORIES

    if endpoint == "text-search":
        payload = kwargs.get("json", {})
        cat = payload.get("category_filter", "")
        terms = payload.get("search_text", [])
        if cat == "biolink:SmallMolecule":
            return {term: MOCK_NODES_SMALL_MOL for term in terms}
        if cat == "biolink:Protein":
            return {term: MOCK_NODES_PROTEIN for term in terms}
        return {term: [] for term in terms}

    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKestrelDiscovery:
    """Tests for kestrel_discovery module."""

    @patch("biomapper2.api.kestrel_discovery.bulk_kestrel_request", side_effect=_mock_bulk_request)
    def test_derive_all_presets_happy_path(self, mock_req):
        """derive_all_presets returns correct mapping with frequency ranking."""
        presets, completed = derive_all_presets(ALIASES_FIXTURE)

        # SmallMolecule should have CHEBI first (highest freq), then HMDB
        assert "biolink:SmallMolecule" in presets
        sm_prefixes = presets["biolink:SmallMolecule"]
        assert sm_prefixes[0] == "CHEBI"
        assert "HMDB" in sm_prefixes

        # Protein should have PR and UniProtKB
        assert "biolink:Protein" in presets
        assert "PR" in presets["biolink:Protein"]

        # Non-aliased category (Gene) should have empty presets
        assert presets.get("biolink:Gene") == []

    @patch("biomapper2.api.kestrel_discovery.bulk_kestrel_request", side_effect=_mock_bulk_request)
    def test_prefixes_ranked_by_frequency(self, mock_req):
        """Prefixes are returned in descending frequency order."""
        presets, _ = derive_all_presets(ALIASES_FIXTURE)
        sm_prefixes = presets["biolink:SmallMolecule"]
        assert sm_prefixes.index("CHEBI") < sm_prefixes.index("RARE")

    @patch("biomapper2.api.kestrel_discovery.bulk_kestrel_request", return_value=[])
    def test_empty_categories_list(self, mock_req):
        """Empty categories list returns empty presets dict."""
        presets, _ = derive_all_presets(ALIASES_FIXTURE)
        assert presets == {}

    @patch("biomapper2.api.kestrel_discovery.bulk_kestrel_request")
    def test_text_search_returns_zero_results(self, mock_req):
        """Category with zero text-search results gets empty prefix list."""

        def side_effect(method, endpoint, auth_required=True, **kwargs):
            if endpoint == "categories":
                return ["biolink:SmallMolecule"]
            return {"term": []}  # empty results

        mock_req.side_effect = side_effect
        presets, _ = derive_all_presets({"metabolite": "biolink:SmallMolecule"})
        assert presets["biolink:SmallMolecule"] == []

    @patch(
        "biomapper2.api.kestrel_discovery.bulk_kestrel_request",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    )
    def test_kestrel_unreachable_falls_back_to_disk(self, mock_req, tmp_path: Path):
        """When Kestrel is unreachable, falls back to disk cache."""
        # Prepare a valid disk cache
        cache_path = tmp_path / "presets.json"
        cached_presets = {"biolink:SmallMolecule": ["CHEBI", "HMDB"]}
        save_to_disk(cached_presets, cache_path)

        with patch("biomapper2.api.kestrel_discovery.PRESET_CACHE_PATH", cache_path):
            result = derive_presets_with_fallback(ALIASES_FIXTURE)

        assert result == cached_presets

    @patch("biomapper2.api.kestrel_discovery.bulk_kestrel_request")
    def test_unauthorized_text_search_returns_categories_with_empty_presets(self, mock_req):
        """GET /categories succeeds but POST /text-search is rejected for want of a valid key.

        Previously this simulated ``ValueError("KESTREL_API_KEY environment variable is not set")``,
        but an unset key no longer raises anywhere — the test asserted tolerance of a failure that
        can no longer occur. The real modern failure at this layer is a 401 from the endpoint.
        """

        def side_effect(method, endpoint, auth_required=True, **kwargs):
            if endpoint == "categories":
                return ["biolink:SmallMolecule"]
            response = requests.Response()
            response.status_code = 401
            raise requests.exceptions.HTTPError("401 Client Error: Unauthorized", response=response)

        mock_req.side_effect = side_effect
        presets, _ = derive_all_presets({"metabolite": "biolink:SmallMolecule"})
        assert "biolink:SmallMolecule" in presets
        assert presets["biolink:SmallMolecule"] == []

    def test_save_and_load_round_trip(self, tmp_path: Path):
        """Presets saved to disk can be loaded back with correct data."""
        cache_path = tmp_path / "presets.json"
        original = {"biolink:SmallMolecule": ["CHEBI", "HMDB"], "biolink:Protein": ["PR"]}

        save_to_disk(original, cache_path)
        loaded = load_from_disk(cache_path)

        assert loaded == original

    def test_corrupted_file_falls_through_to_static(self, tmp_path: Path):
        """Corrupted cache file triggers static fallback."""
        cache_path = tmp_path / "presets.json"
        cache_path.write_text("{corrupted json!!")

        with (
            patch(
                "biomapper2.api.kestrel_discovery.bulk_kestrel_request",
                side_effect=requests.exceptions.ConnectionError("down"),
            ),
            patch("biomapper2.api.kestrel_discovery.PRESET_CACHE_PATH", cache_path),
        ):
            result = derive_presets_with_fallback(ALIASES_FIXTURE)

        assert result == STATIC_FALLBACK
