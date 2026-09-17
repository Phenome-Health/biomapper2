"""Unit 5: the additive ``lipid_resolution`` object and its ``lipid_``-prefixed flat columns.

No live services. The goslin metadata and the committed node's matched level are injected as
fixtures shaped like the real pipeline output (``assigned_ids`` for the goslin votes,
``kg_ids_assigned`` for the linked nodes joined on the raw id, ``chosen_kg_id`` for the committed
node). Every field is derived from that metadata, so the object is null off the lipid path and
exposes the SKOS mapping relation between the committed node and the effective query level.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from biomapper2.api.models.responses import LipidResolution
from biomapper2.api.routes.mapping import extract_mapping_result
from biomapper2.core.certificate import issue
from biomapper2.core.resolver import lipid_flat_columns
from biomapper2.mapper import Mapper

# A committed molecular-species node and the raw goslin id it reconciles to. The raw id carries no
# colon (the form goslin stamps), while the linked node is a curie; the join must bridge the two.
_NODE = "RM:0015001"
_RAW_ID = "RM0015001"
_EFFECTIVE = "molecular_species"

_BASE_META: dict[str, Any] = {
    "goslin_dialect": "GOSLIN",
    "goslin_formula": "C42H82NO8P",
    "goslin_mass": 759.578,
    "query_lipid_level_asserted": "sn_position",
    "query_lipid_level_effective": _EFFECTIVE,
}


def _lipid_item(matched_level: str) -> dict[str, Any]:
    """A mapped item shaped like the pipeline output for a lipid committed to ``_NODE``."""
    meta = dict(_BASE_META)
    meta["matched_level"] = matched_level
    return {
        "name": "PC 16:0/18:1",
        "chosen_kg_id": _NODE,
        "chosen_kg_id_review": None,
        "assigned_ids": {"goslin-lipid": {"REFMET": {_RAW_ID: meta}}},
        "kg_ids_assigned": {"goslin-lipid": {_NODE: [_NODE]}},
    }


def test_lipid_resolution_model_serializes_to_json() -> None:
    model = LipidResolution(
        query_lipid_level_asserted="sn_position",
        query_lipid_level_effective=_EFFECTIVE,
        matched_lipid_level=_EFFECTIVE,
        mapping_relation="exact",
        mapping_predicate="skos:exactMatch",
        query_transformed="slash_to_underscore",
    )
    payload = json.loads(model.model_dump_json())
    assert payload["mapping_predicate"] == "skos:exactMatch"
    assert payload["ambiguous"] is False


def test_non_lipid_row_has_null_lipid_resolution() -> None:
    mapped = {"name": "glucose", "chosen_kg_id": "CHEBI:4167", "assigned_ids": {}, "kg_ids_assigned": {}}
    result = extract_mapping_result(mapped, "glucose")
    assert result.lipid_resolution is None


def test_exact_match_lipid_row_exposes_every_field() -> None:
    # The committed node matched at the effective level, so the relation is an exact SKOS match and,
    # because the asserted level was finer than the effective one, the query was slash-downgraded.
    result = extract_mapping_result(_lipid_item(_EFFECTIVE), "PC 16:0/18:1")
    lr = result.lipid_resolution
    assert lr is not None
    assert lr.matched_lipid_level == _EFFECTIVE
    assert lr.mapping_relation == "exact"
    assert lr.mapping_predicate == "skos:exactMatch"
    assert lr.query_transformed == "slash_to_underscore"
    assert lr.query_lipid_level_asserted == "sn_position"
    assert lr.goslin_dialect == "GOSLIN"
    assert lr.goslin_formula == "C42H82NO8P"


def test_broad_match_lipid_row_sets_broad_relation_and_predicate() -> None:
    # The committed node matched only at species, broader than the effective molecular-species query:
    # the generalized case the flat lipid hint marks, now carried as a broad SKOS relation.
    result = extract_mapping_result(_lipid_item("species"), "PC 16:0/18:1")
    lr = result.lipid_resolution
    assert lr is not None
    assert lr.matched_lipid_level == "species"
    assert lr.mapping_relation == "broad"
    assert lr.mapping_predicate == "skos:broadMatch"


def test_certificate_provenance_mirrors_relation_and_ambiguous() -> None:
    # Decision 5: a certificate read alone must show the lipid relation and ambiguity. The mapper
    # merges them onto provenance without replacing the default cache/Tier B keys.
    mapper = Mapper.__new__(Mapper)
    mapper.resolver = MagicMock()
    mapper.resolver.is_small_molecule.return_value = False
    mapper.tier_b = None
    certificate = mapper._issue_certificate(
        query_name="PC 16:0/18:1",
        category="metabolite",
        chosen_kg_id=_NODE,
        kg_equivalent_ids={},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        kg_ids_assigned={"goslin-lipid": {_NODE: [_NODE]}},
        lipid_mapping_relation="broad",
        lipid_ambiguous=False,
    )
    api = certificate.to_api_dict()
    assert api["provenance"]["mapping_relation"] == "broad"
    assert api["provenance"]["ambiguous"] is False
    # The default cache/Tier B provenance survives the merge rather than being replaced.
    assert "tier_b_enabled" in api["provenance"]
    assert certificate.to_flat_columns()["certificate_provenance_mapping_relation"] == "broad"


def test_issue_extra_provenance_leaves_non_lipid_certificate_untouched() -> None:
    # Without extra provenance the default keys are the only ones present, so a non-lipid certificate
    # gains no lipid fields.
    certificate = issue(
        chosen_kg_id="CHEBI:4167",
        is_small_molecule=True,
        kg_equivalent_ids={"INCHIKEY": ["WQZGKKKJIJFFOK-GASJEMHNSA-N"]},
        equivalent_ids_lookup_ok=True,
    )
    assert "mapping_relation" not in certificate.provenance


def test_dataset_lipid_flat_columns_emitted_for_lipid_and_null_for_non_lipid() -> None:
    lipid = {
        "matched_lipid_level": _EFFECTIVE,
        "mapping_relation": "exact",
        "mapping_predicate": "skos:exactMatch",
    }
    lipid_columns = lipid_flat_columns(lipid)
    assert lipid_columns["lipid_mapping_relation"] == "exact"
    assert lipid_columns["lipid_matched_lipid_level"] == _EFFECTIVE
    # A non-lipid row still gets the full column set, every value null, so the schema stays stable.
    null_columns = lipid_flat_columns(None)
    assert set(null_columns) == set(lipid_columns)
    assert all(value is None for value in null_columns.values())
