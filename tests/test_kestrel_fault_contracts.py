"""Unit tests for Kestrel fault isolation contracts (Rules 3.1, 3.2, 3.3)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from biomapper2.core.fault_tracking import (
    KestrelFaultError,
    record_kestrel_fault,
)
from biomapper2.mapper import Mapper


@pytest.mark.unit
def test_fault_contract_entity_unresolved_with_kestrel_fault_raises() -> None:
    """Unresolved entity with Kestrel fault raises KestrelFaultError per Rule 3.2."""
    mapper = Mapper.__new__(Mapper)
    mapper.biolink_client = MagicMock()
    mapper.biolink_client.standardize_entity_type.return_value = "biolink:Disease"
    mapper.normalizer = MagicMock()
    mapper.normalizer.get_standard_prefix.return_value = []
    mapper.normalizer.normalize.return_value = pd.Series({"curies": [], "curies_provided": [], "curies_assigned": {}})
    mapper.linker = MagicMock()
    mapper.linker.link.return_value = pd.Series({"kg_ids": {}, "kg_ids_provided": {}, "kg_ids_assigned": {}})
    mapper.resolver = MagicMock()
    mapper.resolver.resolve.return_value = pd.Series({"chosen_kg_id": None, "chosen_kg_id_review": None})

    def fake_annotate(*args: Any, **kwargs: Any) -> pd.Series:
        record_kestrel_fault(None, "kestrel-hybrid-search: transport timeout")
        return pd.Series({"annotator_availability": {}, "annotator_source": {}, "kestrel_endpoints_used": []})

    mapper.annotation_engine = MagicMock()
    mapper.annotation_engine.annotate.side_effect = fake_annotate
    mapper._certify_and_reresolve = MagicMock()
    cert_mock = MagicMock()
    cert_mock.to_api_dict.return_value = None
    cert_mock.tier_b_snapshot_version = None
    mapper._certify_and_reresolve.return_value = (cert_mock, None, {}, None)

    with patch("biomapper2.mapper.derive_chosen_kg_id_review", return_value=None):
        with pytest.raises(KestrelFaultError) as exc_info:
            mapper.map_entity_to_kg(
                item={"name": "rare disease"},
                name_field="name",
                provided_id_fields=[],
                entity_type="disease",
            )

    assert "transport timeout" in str(exc_info.value)


@pytest.mark.unit
def test_fault_contract_entity_resolved_with_kestrel_fault_marks_degraded() -> None:
    """Resolved entity with Kestrel fault is tagged with kestrel_degraded True per Rule 3.3."""
    mapper = Mapper.__new__(Mapper)
    mapper.biolink_client = MagicMock()
    mapper.biolink_client.standardize_entity_type.return_value = "biolink:SmallMolecule"
    mapper.normalizer = MagicMock()
    mapper.normalizer.get_standard_prefix.return_value = ["CHEBI"]
    curies_series = pd.Series({"curies": ["CHEBI:1"], "curies_provided": ["CHEBI:1"], "curies_assigned": {}})
    mapper.normalizer.normalize.return_value = curies_series
    mapper.linker = MagicMock()
    kg_ids_series = pd.Series(
        {"kg_ids": {"CHEBI:1": ["CHEBI:1"]}, "kg_ids_provided": {"CHEBI:1": ["CHEBI:1"]}, "kg_ids_assigned": {}}
    )
    mapper.linker.link.return_value = kg_ids_series
    mapper.resolver = MagicMock()
    mapper.resolver.resolve.return_value = pd.Series({"chosen_kg_id": "CHEBI:1", "chosen_kg_id_review": None})
    mapper.linker.get_equivalent_ids_checked.return_value = ({}, True)

    def fake_annotate(*args: Any, **kwargs: Any) -> pd.Series:
        record_kestrel_fault(None, "kestrel-hybrid-search: HTTP 500 error")
        return pd.Series({"annotator_availability": {}, "annotator_source": {}, "kestrel_endpoints_used": []})

    mapper.annotation_engine = MagicMock()
    mapper.annotation_engine.annotate.side_effect = fake_annotate
    mapper._certify_and_reresolve = MagicMock()
    cert_mock = MagicMock()
    cert_mock.to_api_dict.return_value = {}
    cert_mock.tier_b_snapshot_version = None
    mapper._certify_and_reresolve.return_value = (cert_mock, "CHEBI:1", {}, None)

    with patch("biomapper2.mapper.derive_chosen_kg_id_review", return_value=None):
        res = mapper.map_entity_to_kg(
            item={"name": "Cholic acid", "chebi": "1"},
            name_field="name",
            provided_id_fields=["chebi"],
            entity_type="metabolite",
        )

    assert res.get("chosen_kg_id") == "CHEBI:1"
    assert res.get("kestrel_degraded") is True


@pytest.mark.unit
def test_fault_contract_entity_healthy_omits_degraded() -> None:
    """Healthy entity mapping does not tag kestrel_degraded."""
    mapper = Mapper.__new__(Mapper)
    mapper.biolink_client = MagicMock()
    mapper.biolink_client.standardize_entity_type.return_value = "biolink:SmallMolecule"
    mapper.normalizer = MagicMock()
    mapper.normalizer.get_standard_prefix.return_value = ["CHEBI"]
    curies_series = pd.Series({"curies": ["CHEBI:1"], "curies_provided": ["CHEBI:1"], "curies_assigned": {}})
    mapper.normalizer.normalize.return_value = curies_series
    mapper.linker = MagicMock()
    kg_ids_series = pd.Series(
        {"kg_ids": {"CHEBI:1": ["CHEBI:1"]}, "kg_ids_provided": {"CHEBI:1": ["CHEBI:1"]}, "kg_ids_assigned": {}}
    )
    mapper.linker.link.return_value = kg_ids_series
    mapper.resolver = MagicMock()
    mapper.resolver.resolve.return_value = pd.Series({"chosen_kg_id": "CHEBI:1", "chosen_kg_id_review": None})
    mapper.linker.get_equivalent_ids_checked.return_value = ({}, True)

    mapper.annotation_engine = MagicMock()
    mapper.annotation_engine.annotate.return_value = pd.Series(
        {"annotator_availability": {}, "annotator_source": {}, "kestrel_endpoints_used": []}
    )
    mapper._certify_and_reresolve = MagicMock()
    cert_mock = MagicMock()
    cert_mock.to_api_dict.return_value = {}
    cert_mock.tier_b_snapshot_version = None
    mapper._certify_and_reresolve.return_value = (cert_mock, "CHEBI:1", {}, None)

    with patch("biomapper2.mapper.derive_chosen_kg_id_review", return_value=None):
        res = mapper.map_entity_to_kg(
            item={"name": "glucose", "chebi": "1"},
            name_field="name",
            provided_id_fields=["chebi"],
            entity_type="metabolite",
        )

    assert res.get("chosen_kg_id") == "CHEBI:1"
    assert res.get("kestrel_degraded") is None


@pytest.mark.unit
def test_fault_contract_gh1_human_ortholog_discard_on_get_nodes_fault() -> None:
    """GH1 curated fallback discards search ortholog and raises when get-nodes fails."""
    from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator

    annotator = KestrelHybridSearchAnnotator.__new__(KestrelHybridSearchAnnotator)
    annotator._resolver = MagicMock()
    annotator._resolver.resolve.return_value = None
    annotator._resolver.last_call_faulted.return_value = True

    search_mock = {"GH1": [{"id": "NCBIGene:9999", "score": 0.9, "categories": ["biolink:Gene"]}]}
    with patch.object(annotator, "_kestrel_hybrid_search", return_value=search_mock):
        with patch.object(annotator, "_select_result", return_value=(None, False)):
            result = annotator.get_annotations(
                entity={"name": "GH1"},
                name_field="name",
                category="biolink:Gene",
                prefer_human=True,
            )

    annot_dict = result.get(annotator.slug, {})
    assert annot_dict == {}
