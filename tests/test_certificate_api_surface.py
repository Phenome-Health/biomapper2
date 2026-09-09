"""Serialization at both API call sites, because both break by default.

Pydantic rejects a raw dataclass at ``EntityMappingResult``, and the streaming endpoint builds a
plain, unvalidated dict whose ``json.dumps`` sits OUTSIDE the surrounding try/except -- a dataclass
there raises ``TypeError`` mid-stream, after a 200 has already been sent to the client. Neither is a
hypothetical: they are the default behaviour of the code as it stands.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from biomapper2.api.models.responses import EntityMappingResult, ResolutionCertificateModel
from biomapper2.api.routes.mapping import extract_mapping_result
from biomapper2.core.certificate import issue

NODE = "CHEBI:15365"
WITH_KEY = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]}


def _certificate(**overrides) -> dict[str, Any]:
    kwargs = dict(
        chosen_kg_id=NODE,
        is_small_molecule=True,
        kg_equivalent_ids=WITH_KEY,
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
    )
    kwargs.update(overrides)
    return issue(**kwargs).to_api_dict()  # pyright: ignore[reportArgumentType]


def test_response_model_accepts_the_emitted_dict() -> None:
    result = EntityMappingResult(
        name="glucose", resolution_certificate=_certificate()  # pyright: ignore[reportArgumentType]
    )
    assert result.resolution_certificate is not None
    assert result.resolution_certificate.state == "uncorroborated"


def test_response_model_round_trips_to_json() -> None:
    result = EntityMappingResult(
        name="glucose", resolution_certificate=_certificate()  # pyright: ignore[reportArgumentType]
    )
    payload = json.loads(result.model_dump_json())
    assert payload["resolution_certificate"]["structure_status"] == "structure_present"
    assert payload["resolution_certificate"]["node_inchikey_blocks"] == ["BSYNRYMUTXBXSQ"]


def test_certificate_is_optional_so_error_rows_still_construct() -> None:
    assert EntityMappingResult(name="glucose", error="boom").resolution_certificate is None


@pytest.mark.parametrize("as_series", [False, True])
def test_extract_mapping_result_carries_the_certificate_through(as_series: bool) -> None:
    mapped: Any = {
        "name": "glucose",
        "chosen_kg_id": NODE,
        "chosen_kg_id_review": None,
        "resolution_certificate": _certificate(),
    }
    if as_series:
        mapped = pd.Series(mapped)
    result = extract_mapping_result(mapped, "glucose")
    assert result.resolution_certificate is not None
    assert result.resolution_certificate.comparison_rule.startswith("inchikey_first_block_set_intersection")


def test_extract_mapping_result_survives_a_row_with_no_certificate() -> None:
    """Older callers and error rows must not raise a server error on a missing field."""
    result = extract_mapping_result({"name": "glucose", "chosen_kg_id": None}, "glucose")
    assert result.resolution_certificate is None


def test_streaming_payload_is_json_dumpable() -> None:
    """The NDJSON generator dumps outside its try/except; a non-plain type there is a mid-stream
    failure the client sees as a truncated 200."""
    row = {
        "row_index": 0,
        "name": "glucose",
        "chosen_kg_id": NODE,
        "resolution_certificate": _certificate(),
    }
    assert json.loads(json.dumps(row))["resolution_certificate"]["state"] == "uncorroborated"


def test_deprecated_legacy_field_still_documented_and_populated() -> None:
    """``chosen_kg_id_review`` keeps its field with a deprecation note naming the replacement."""
    description = EntityMappingResult.model_fields["chosen_kg_id_review"].description or ""
    assert "resolution_certificate" in description
    result = extract_mapping_result(
        {
            "name": "glucose",
            "chosen_kg_id": NODE,
            "chosen_kg_id_review": "divergent_refmet",
            "resolution_certificate": _certificate(selection_conflict="divergent_refmet"),
        },
        "glucose",
    )
    assert result.chosen_kg_id_review == "divergent_refmet"


def test_contradicted_is_documented_as_a_human_prompt_not_a_verdict() -> None:
    """PubChem name lookup returned an acyl-shifted variant for a real query name, so a
    ``contradicted`` row means "a human should look", never "the resolver is wrong". The wording
    belongs in the field description, where an API consumer actually reads it."""
    description = ResolutionCertificateModel.model_fields["state"].description or ""
    assert "human" in description.lower()


def test_column_scope_is_documented_on_the_api_field() -> None:
    """``chosen_kg_id_provided`` and ``chosen_kg_id_assigned`` never receive a category and get no
    certificate, so the scope has to be stated where it can be read."""
    description = EntityMappingResult.model_fields["resolution_certificate"].description or ""
    assert "chosen_kg_id" in description
