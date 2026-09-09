"""The certificate's ``refmet_availability`` field and its two surfaces (U3), plus the API (U4).

A runtime availability input like ``equivalent_ids_lookup_ok``: recorded on the certificate and
mirrored per-row on the API result, but it does NOT drive the state machine. The existing state-table
invariants must be unaffected.
"""

from __future__ import annotations

import json

import pytest

from biomapper2.api.models.responses import EntityMappingResult, ResolutionCertificateModel
from biomapper2.api.routes.mapping import extract_mapping_result
from biomapper2.core.certificate import CertificateState, issue

NODE = "CHEBI:15365"
WITH_KEY = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]}
ALL_VALUES = ["voted", "no_match", "unavailable", "not_queried"]


def _issue(**overrides):
    kwargs = dict(
        chosen_kg_id=NODE,
        is_small_molecule=True,
        kg_equivalent_ids=WITH_KEY,
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
    )
    kwargs.update(overrides)
    return issue(**kwargs)  # pyright: ignore[reportArgumentType]


def test_field_defaults_to_not_queried():
    assert _issue().refmet_availability == "not_queried"


@pytest.mark.parametrize("value", ALL_VALUES)
def test_field_is_carried_for_all_four_values(value: str):
    assert _issue(refmet_availability=value).refmet_availability == value


def test_availability_does_not_change_the_state_machine():
    """The state is a function of structure evidence, not of whether RefMet answered."""
    base = _issue().state
    assert all(_issue(refmet_availability=v).state is base for v in ALL_VALUES)
    assert base is CertificateState.UNCORROBORATED


@pytest.mark.parametrize("value", ALL_VALUES)
def test_both_surfaces_carry_the_field(value: str):
    cert = _issue(refmet_availability=value)
    api = cert.to_api_dict()
    assert api["refmet_availability"] == value
    assert json.loads(json.dumps(api)) == api  # still plain JSON
    flat = cert.to_flat_columns()
    assert flat["certificate_refmet_availability"] == value
    assert flat["certificate_refmet_availability"] is None or isinstance(flat["certificate_refmet_availability"], str)


def test_response_model_defaults_and_accepts_the_field():
    assert ResolutionCertificateModel.model_fields["refmet_availability"].default == "not_queried"
    model = ResolutionCertificateModel(
        state="uncorroborated",
        structure_status="structure_present",
        comparison_rule="r",
        equivalent_ids_lookup_ok=True,
        refmet_availability="unavailable",
    )
    assert model.refmet_availability == "unavailable"


def test_entity_result_mirrors_availability_and_defaults_present():
    """Per-row mirror (R2): always present, never None; error rows still construct."""
    assert EntityMappingResult(name="glucose").refmet_availability == "not_queried"
    assert EntityMappingResult(name="glucose", error="boom").refmet_availability == "not_queried"
    assert EntityMappingResult(name="glucose", refmet_availability="voted").refmet_availability == "voted"


def test_extract_mapping_result_passes_availability_through():
    result = extract_mapping_result(
        {"name": "retinol", "chosen_kg_id": NODE, "refmet_availability": "unavailable"}, "retinol"
    )
    assert result.refmet_availability == "unavailable"
    # A mapped item missing the field (older caller / error row) falls back to not_queried.
    assert extract_mapping_result({"name": "x", "chosen_kg_id": None}, "x").refmet_availability == "not_queried"
