"""primary_curie is derived from the committed node, and the response says which source produced each CURIE.

Offline only. The regression half replays recorded prod v1.2.1 rows (tests/fixtures/primary_curie_v121_rows.json,
built by tests/fixtures/build_primary_curie_fixture.py) through the new rule.
"""

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.main import app
from biomapper2.api.routes.mapping import extract_mapping_result, identity_fields
from biomapper2.core.primary_curie import chosen_node_identifiers, select_primary_curie

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "primary_curie_v121_rows.json"

# Annotator slugs are illustrative: the recorded client rows do not say which Kestrel search
# annotator returned the losing CHEBI hit (that is the provenance gap this change closes).
KRAKEN_SEARCH = "kestrel-hybrid-search"
REFMET = "metabolomics-workbench"


def _divergent_refmet_item(losing_curie: str, refmet_node: str, losing_node: str) -> dict[str, Any]:
    """A mapped item shaped like the prod rows: KRAKEN search's CURIE sorts first, RefMet's node wins."""
    return {
        "name": "x",
        "curies": sorted([losing_curie, refmet_node]),
        "curies_provided": [],
        "curies_assigned": {KRAKEN_SEARCH: [losing_curie], REFMET: [refmet_node]},
        "kg_ids": {losing_node: [losing_curie], refmet_node: [refmet_node]},
        "chosen_kg_id": refmet_node,
        "chosen_kg_id_review": "divergent_refmet",
        "kg_equivalent_ids": {"RM": [refmet_node.split(":", 1)[1]]},
        "assigned_ids": {
            KRAKEN_SEARCH: {"CHEBI": {losing_curie.split(":", 1)[1]: {"score": 2.4}}},
            REFMET: {"refmet_id": {"RM" + refmet_node.split(":", 1)[1]: {}}},
        },
    }


# --------------------------------------------------------------------------- the rule


class TestSelectPrimaryCurie:
    def test_unresolved_keeps_legacy_first_curie(self):
        assert select_primary_curie(["CHEBI:1", "RM:2"], None) == "CHEBI:1"
        assert select_primary_curie([], None) is None
        assert select_primary_curie(None, None) is None

    def test_first_curie_already_on_chosen_node_is_unchanged(self):
        # curies[0] links to the chosen node -> exactly curies[0], even though a later CURIE also does.
        kg_ids = {"RM:9": ["CHEBI:1", "RM:9"]}
        assert select_primary_curie(["CHEBI:1", "RM:9"], "RM:9", kg_ids, {}) == "CHEBI:1"

    def test_first_curie_on_another_node_is_skipped_via_kg_ids(self):
        kg_ids = {"CHEBI:1": ["CHEBI:1"], "RM:9": ["RM:9"]}
        assert select_primary_curie(["CHEBI:1", "RM:9"], "RM:9", kg_ids, {}) == "RM:9"

    def test_membership_via_equivalent_ids(self):
        # HMDB:HMDB2 did not show up in kg_ids but the chosen node lists it as an equivalent.
        equiv = {"HMDB": ["HMDB2"], "RM": ["9"]}
        assert select_primary_curie(["CHEBI:1", "HMDB:HMDB2"], "RM:9", {}, equiv) == "HMDB:HMDB2"

    def test_no_curie_on_chosen_node_falls_back_to_chosen_kg_id(self):
        assert select_primary_curie(["CHEBI:1"], "RM:9", {"CHEBI:1": ["CHEBI:1"]}, {}) == "RM:9"
        assert select_primary_curie([], "RM:9") == "RM:9"

    def test_equivalent_ids_rejoin_on_first_colon(self):
        # The linker splits "PREFIX:local" on the FIRST colon, so a local id containing ':' round-trips.
        ids = chosen_node_identifiers("X:1", {}, {"HGVS": ["NC_000021.9:g.1A>G"]})
        assert "HGVS:NC_000021.9:g.1A>G" in ids


# --------------------------------------------------------------------------- the two reported rows


@pytest.mark.parametrize(
    "losing_curie,refmet_node,losing_node",
    [
        # dihomo-linolenate (20:3n3 or n6): chosen DGLA, curies[0] was dihomo-linoleate 20:2.
        ("CHEBI:88670", "RM:0153916", "CHEBI:88670"),
        # dihomolinolenate (20:3n3 or 3n6): chosen RM:0152308, curies[0] was dihomolinoleate.
        ("CHEBI:83066", "RM:0152308", "CHEBI:83066"),
    ],
)
def test_reported_examples_report_the_chosen_node(losing_curie, refmet_node, losing_node):
    result = extract_mapping_result(_divergent_refmet_item(losing_curie, refmet_node, losing_node), "x")
    assert result.curies[0] == losing_curie  # the field the client used to read; order unchanged
    assert result.primary_curie == refmet_node == result.chosen_kg_id
    assert result.curies_assigned == {KRAKEN_SEARCH: [losing_curie], REFMET: [refmet_node]}


def _rows() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["rows"]


@pytest.mark.parametrize(
    "name,chosen,expected",
    [
        ("dihomo-linolenate (20:3n3 or n6)", "RM:0153916", "RM:0153916"),
        ("dihomolinolenate (20:3n3 or 3n6)", "RM:0152308", "RM:0152308"),
    ],
)
def test_reported_examples_from_recorded_prod_rows(name, chosen, expected):
    rows = [r for r in _rows() if r["name"] == name]
    assert rows, name
    for r in rows:
        assert r["chosen_kg_id"] == chosen
        assert r["recorded_primary_curie"].startswith("CHEBI:")  # the bug as recorded
        assert select_primary_curie(r["curies"], r["chosen_kg_id"], {}, r["kg_equivalent_ids"]) == expected


# --------------------------------------------------------------------------- recorded-fixture regression


def test_recorded_rows_unchanged_when_already_consistent_and_fixed_otherwise():
    """Over every recorded v1.2.1 row: consistent rows keep their value byte-for-byte; the rest now
    name the chosen node. Also pins the impact count reported in the PR."""
    recorded = unchanged = changed = 0
    changed_by_review: dict[str, int] = {}
    for r in _rows():
        weight = len(r["panels"])
        recorded += weight
        new = select_primary_curie(r["curies"], r["chosen_kg_id"], {}, r["kg_equivalent_ids"])
        if not r["chosen_kg_id"]:
            assert new == r["recorded_primary_curie"]
            unchanged += weight
            continue
        node = chosen_node_identifiers(r["chosen_kg_id"], {}, r["kg_equivalent_ids"])
        assert new in node
        if r["recorded_primary_curie"] in node:
            assert new == r["recorded_primary_curie"], r["name"]
            unchanged += weight
        else:
            assert new != r["recorded_primary_curie"]
            changed += weight
            key = r["chosen_kg_id_review"] or "none"
            changed_by_review[key] = changed_by_review.get(key, 0) + weight
    assert recorded == 3973
    assert changed == 1364
    assert unchanged == recorded - changed
    assert changed_by_review == {"divergent_refmet": 248, "conflict_no_structure": 4, "none": 1112}


# --------------------------------------------------------------------------- provenance + surfaces


def test_identity_fields_provenance_is_per_source_and_sorted():
    fields = identity_fields(
        {
            "curies": ["CHEBI:1", "HMDB:HMDB2", "RM:9"],
            "curies_provided": ["HMDB:HMDB2"],
            "curies_assigned": {REFMET: ["RM:9"], KRAKEN_SEARCH: ["RM:9", "CHEBI:1"]},
            "kg_ids": {"RM:9": ["RM:9", "HMDB:HMDB2"], "CHEBI:1": ["CHEBI:1"]},
            "chosen_kg_id": "RM:9",
        }
    )
    assert fields == {
        "curies": ["CHEBI:1", "HMDB:HMDB2", "RM:9"],
        "primary_curie": "HMDB:HMDB2",
        "curies_provided": ["HMDB:HMDB2"],
        "curies_assigned": {REFMET: ["RM:9"], KRAKEN_SEARCH: ["CHEBI:1", "RM:9"]},
    }


def test_error_result_has_empty_identity_fields():
    from biomapper2.api.models import EntityMappingResult

    r = EntityMappingResult(name="x", error="boom")
    assert r.primary_curie is None and r.curies_provided == [] and r.curies_assigned == {}


@pytest.fixture
def client_with_mapper():
    with TestClient(app) as c:
        fa = c.app
        assert isinstance(fa, FastAPI)
        original = fa.state.mapper
        mapper = MagicMock()
        mapper.map_entity_to_kg.return_value = _divergent_refmet_item("CHEBI:88670", "RM:0153916", "CHEBI:88670")
        fa.state.mapper = mapper
        try:
            yield c
        finally:
            fa.state.mapper = original


def test_entity_and_batch_routes_emit_primary_curie_and_provenance(client_with_mapper: TestClient):
    single = client_with_mapper.post("/api/v1/map/entity", json={"name": "x", "entity_type": "metabolite"})
    assert single.status_code == 200
    batch = client_with_mapper.post(
        "/api/v1/map/batch", json={"entities": [{"name": "x", "entity_type": "metabolite"}]}
    )
    assert batch.status_code == 200
    for result in (single.json()["result"], batch.json()["results"][0]):
        assert result["primary_curie"] == "RM:0153916"
        assert result["curies"] == ["CHEBI:88670", "RM:0153916"]  # existing field: same shape, same order
        assert result["curies_provided"] == []
        assert result["curies_assigned"] == {KRAKEN_SEARCH: ["CHEBI:88670"], REFMET: ["RM:0153916"]}
        assert set(result["assigned_ids"]) == {KRAKEN_SEARCH, REFMET}  # raw per-annotator ids unchanged


def test_dataset_stream_emits_primary_curie_and_provenance(client_with_mapper: TestClient):
    resp = client_with_mapper.post(
        "/api/v1/map/dataset/stream?entity_type=metabolite&name_column=name&provided_id_columns=kegg",
        files={"file": ("in.csv", io.BytesIO(b"name,kegg\nx,\n"), "text/csv")},
    )
    assert resp.status_code == 200
    line = json.loads(resp.text.splitlines()[0])
    assert line["primary_curie"] == "RM:0153916"
    assert line["curies"] == ["CHEBI:88670", "RM:0153916"]
    assert line["curies_assigned"] == {KRAKEN_SEARCH: ["CHEBI:88670"], REFMET: ["RM:0153916"]}
