"""Units 3 and 5, live: the success-criteria ICD-10-CM codes against public Kestrel match their routes.

The study (studies/analysis/icd10cm_twin_control.py) recorded each code's route on the pinned KG build
into tests/fixtures/icd10cm_twin_regression.json. A drift here means the KG moved, not necessarily a
code bug: rerun the study before changing expectations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomapper2 import config
from biomapper2.core import icd10cm_hierarchy
from biomapper2.mapper import Mapper

pytestmark = [pytest.mark.integration, pytest.mark.requires_api]

# Phase 1b success criteria: twin misses that the walk-up resolves, broad, via the named ancestor.
WALKUP_EXPECTED = {"E11.65": ("MONDO:0005148", "walkup:E11"), "F32.9": ("MONDO:0002009", "walkup:F32")}

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
        assert result["icd10cm_snapshot_version"] is None
    elif config.ICD10CM_WALKUP_ENABLED:
        node, route = WALKUP_EXPECTED[code]
        assert result["chosen_kg_id"] == node
        assert result["icd10cm_route"] == route
        assert result["mapping_relation"] == "broad"
        assert result["mapping_predicate"] == "skos:broadMatch"
        assert result["kg_ids_provided"] == {node: [f"ICD10CM:{code}"]}
        assert result["icd10cm_snapshot_version"] == icd10cm_hierarchy.version()
        assert result["kg_build"]
    else:
        assert result["chosen_kg_id"] is None
        assert result["icd10cm_route"] is None
