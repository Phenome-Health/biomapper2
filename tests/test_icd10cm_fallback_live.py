"""Unit 3, live: the success-criteria ICD-10-CM codes against public Kestrel match the gate study's routes.

The study (studies/analysis/icd10cm_twin_control.py) recorded each code's route on the pinned KG build
into tests/fixtures/icd10cm_twin_regression.json. A drift here means the KG moved, not necessarily a
code bug: rerun the study before changing expectations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomapper2 import config
from biomapper2.mapper import Mapper

pytestmark = [pytest.mark.integration, pytest.mark.requires_api]

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "icd10cm_twin_regression.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("code", sorted(FIXTURE["success_criteria_routes"]))
def test_success_criteria_code_matches_study_route(shared_mapper: Mapper, code: str) -> None:
    expected = FIXTURE["success_criteria_routes"][code]
    result = shared_mapper.map_entity_to_kg(
        {"name": code, "icd10cm": code},
        name_field="name",
        provided_id_fields=["icd10cm"],
        entity_type="disease",
        annotation_mode="none",
    )
    assert isinstance(result, dict)
    if expected["route"] == "twin":
        assert result["chosen_kg_id"] == expected["twin_node"]
        assert result["icd10cm_route"] == "twin"
        assert result["mapping_relation"] in {config.ICD10CM_TWIN_MAPPING_RELATION, "broad"}
        assert result["kg_ids_provided"] == {expected["twin_node"]: [f"ICD10CM:{code}"]}
        assert result["kg_build"]
    else:
        # Twin miss: unresolved until the Phase 1b walk-up lands.
        assert result["chosen_kg_id"] is None
        assert result["icd10cm_route"] is None
