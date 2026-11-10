"""Unit tests for MetabolomicsWorkbenchAnnotator.

Test suite covering:
- Basic annotator structure (1 test)
- Core annotation behavior with /match endpoint (1 test)
- Edge cases: no match, missing input (2 tests)
- Bulk operations (1 test)
- Engine integration (1 test)
- Real API calls: exact + fuzzy match (1 integration test)
- Full Mapper pipeline (1 end-to-end test)
"""

import pandas as pd
import pytest

from biomapper2.core.annotators.base import BaseAnnotator
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _RoutedSession:
    """Stands in for the annotator's cached session (it never uses ``requests.get``).

    Routes by URL prefix: ``/refmet/match/...`` and ``/refmet/name/...``. Records every URL.
    """

    def __init__(self, match_body, name_body=None):
        self.match_body = match_body
        self.name_body = name_body
        self.urls: list[str] = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if url.startswith(MetabolomicsWorkbenchAnnotator.NAME_URL):
            return _Resp(self.name_body)
        return _Resp(self.match_body)


def _offline(annotator, match_body, name_body=None):
    annotator._session = _RoutedSession(match_body, name_body)  # type: ignore[assignment]
    return annotator._session


class TestMetabolomicsWorkbenchAnnotator:
    """Consolidated test suite for MetabolomicsWorkbenchAnnotator."""

    # =========================================================================
    # Test 1: Basic annotator structure
    # =========================================================================
    @pytest.mark.unit
    def test_annotator_basics(self):
        """Test annotator has correct slug and inherits from BaseAnnotator."""
        annotator = MetabolomicsWorkbenchAnnotator()
        assert annotator.slug == "metabolomics-workbench"
        assert isinstance(annotator, BaseAnnotator)

    # =========================================================================
    # Test 2: Core annotation behavior
    # =========================================================================
    @pytest.mark.unit
    def test_get_annotations_returns_refmet_id(self):
        """Test that annotations return refmet_id (KRAKEN has all equivalencies)."""
        annotator = MetabolomicsWorkbenchAnnotator()
        # The pre-2026-11-10 /match shape, which still carried refmet_id: no /name call is needed.
        session = _offline(
            annotator,
            {
                "refmet_name": "Carnitine",
                "formula": "C7H15NO3",
                "exactmass": "161.1052",
                "super_class": "Fatty Acyls",
                "main_class": "Fatty acyl carnitines",
                "sub_class": "Acyl carnitines",
                "refmet_id": "RM0008606",
            },
        )
        entity = {"name": "Carnitine"}

        # Act
        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")
        assert not any(u.startswith(MetabolomicsWorkbenchAnnotator.NAME_URL) for u in session.urls)

        # Assert - only refmet_id is extracted (KRAKEN has all equivalencies)
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]
        # Other fields NOT extracted since we only need refmet_id
        assert "pubchem_cid" not in annotations

    # =========================================================================
    # Test 3: Edge case - not found
    # =========================================================================
    @pytest.mark.unit
    @pytest.mark.parametrize("with_id_field", [True, False], ids=["legacy-dash-id", "no-id-field"])
    def test_no_match_response(self, with_id_field):
        """A /match no-match (dash values) is an empty vote, with or without the refmet_id field."""
        body = {
            "refmet_name": "-",
            "formula": "-",
            "exactmass": "-",
            "super_class": "-",
            "main_class": "-",
            "sub_class": "-",
        }
        if with_id_field:
            body["refmet_id"] = "-"
        annotator = MetabolomicsWorkbenchAnnotator()
        session = _offline(annotator, body)
        entity = {"name": "NonexistentMetabolite"}

        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")

        assert result == {"metabolomics-workbench": {}}
        assert not any(u.startswith(MetabolomicsWorkbenchAnnotator.NAME_URL) for u in session.urls)

    # =========================================================================
    # Test 4: Edge case - missing input
    # =========================================================================
    @pytest.mark.unit
    def test_missing_name_field(self):
        """Test that entity without name field returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"other_field": "value"}

        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")

        assert result == {}

    # =========================================================================
    # Test 5: Bulk operations
    # =========================================================================
    @pytest.mark.unit
    def test_bulk_returns_series(self):
        """Test that get_annotations_bulk returns Series with matching index."""
        annotator = MetabolomicsWorkbenchAnnotator()
        _offline(annotator, {"refmet_name": "Carnitine", "refmet_id": "RM0008606"})
        entities = pd.DataFrame(
            {"name": ["Carnitine", "Glucose", "Alanine"]},
            index=[10, 20, 30],
        )

        result = annotator.get_annotations_bulk(entities, name_field="name", category="biolink:SmallMolecule")

        assert isinstance(result, pd.Series)
        assert list(result.index) == [10, 20, 30]
        assert len(result) == 3

    # =========================================================================
    # Test 6: Engine integration
    # =========================================================================
    @pytest.mark.unit
    def test_engine_selects_annotator_for_metabolite(self):
        """Test that AnnotationEngine selects MetabolomicsWorkbenchAnnotator for metabolite."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()
        annotator_slugs = engine._select_annotators("biolink:SmallMolecule")

        assert MetabolomicsWorkbenchAnnotator.slug in annotator_slugs

    # =========================================================================
    # Test 7: Real API call (integration)
    # =========================================================================
    @pytest.mark.integration
    @pytest.mark.third_party
    def test_real_api_match_endpoint(self):
        """Test real API call: exact match and fuzzy match."""
        annotator = MetabolomicsWorkbenchAnnotator()

        # Test 1: Exact match - "Carnitine"
        result = annotator.get_annotations({"name": "Carnitine"}, name_field="name", category="biolink:SmallMolecule")
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]

        # Test 2: Fuzzy match - "cholate" → "Cholic acid" (RM0135798)
        result = annotator.get_annotations({"name": "cholate"}, name_field="name", category="biolink:SmallMolecule")
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0135798" in annotations["refmet_id"]

    # =========================================================================
    # Test 8: Full Mapper pipeline (end-to-end)
    # =========================================================================
    @pytest.mark.integration
    @pytest.mark.third_party
    @pytest.mark.requires_api
    def test_mapper_end_to_end(self):
        """Test full pipeline using Mapper.map_entity_to_kg() with MW annotator."""
        from biomapper2.mapper import Mapper

        mapper = Mapper()
        # Use pd.Series to get a pd.Series back (dict input returns dict)
        entity = pd.Series({"name": "Carnitine"})

        # Run full pipeline: annotation -> normalization -> linking -> resolution
        result = mapper.map_entity_to_kg(
            item=entity,
            name_field="name",
            provided_id_fields=[],
            entity_type="metabolite",
            annotation_mode="all",
        )

        # Verify result is a Series (since input was Series)
        assert isinstance(result, pd.Series)

        # Verify assigned_ids contain MW annotations
        assert "assigned_ids" in result.index
        assigned_ids = result["assigned_ids"]
        assert "metabolomics-workbench" in assigned_ids

        # Verify MW annotations have refmet_id (only field extracted now)
        mw_annotations = assigned_ids["metabolomics-workbench"]
        assert "refmet_id" in mw_annotations
        assert "RM0008606" in mw_annotations["refmet_id"]


def test_a_slash_in_the_query_name_is_sent_literally():
    """The Workbench resolves a literal slash as part of the name and 404s on an encoded one.

    Live (2026-10-21): ``.../match/PC%2016%3A0/18%3A1`` returns RM0015001 (``PC 16:0/18:1``), while
    ``.../match/PC%2016%3A0%2F18%3A1`` is rejected with a 404 before any lookup. Encoding the slash
    (``safe=""``, engine a4aeef7) silently dropped the RefMet answer for every slash-bearing name.

    This is the consequential site: the annotator is called with the QUERY name and its candidates
    enter the vote and the source-weighting, so a lost answer reaches ``chosen_kg_id``. It also
    corrupts ``independent_of_selection``: with no RefMet vote, ``metabolomics-workbench`` is absent
    from the committed node's sources.
    """
    from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

    calls = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"refmet_id": "RM0015001", "name": "PC 16:0/18:1"}

    class _Session:
        def get(self, url, timeout=None):
            calls.append(url)
            return _Resp()

    annotator = MetabolomicsWorkbenchAnnotator.__new__(MetabolomicsWorkbenchAnnotator)
    annotator._session = _Session()  # type: ignore[assignment]  # captures the outbound URL only
    annotator._do_refmet_request("PC 16:0/18:1")

    url = calls[0]
    base = MetabolomicsWorkbenchAnnotator.BASE_URL
    assert "%2F" not in url.upper(), f"the slash was percent-encoded, which the Workbench 404s: {url}"
    assert url == f"{base}/PC%2016%3A0/18%3A1", url
