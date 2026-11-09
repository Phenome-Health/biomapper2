"""Unit 2: the entity-general ``mapping_relation`` / ``mapping_predicate`` / ``icd10cm_route`` / ``kg_build``.

One broad-vs-exact vocabulary across entity types. The fields are ADDITIVE and null by default: every
pre-existing key, column and value is unchanged, and a non-ICD, non-lipid certificate is byte-identical.
No live services.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.main import app
from biomapper2.api.routes.mapping import extract_mapping_result
from biomapper2.core.resolver import MAPPING_FIELDS, mapping_fields, mapping_predicate_for
from biomapper2.mapper import Mapper

pytestmark = pytest.mark.unit

_NEW_KEYS = ("mapping_relation", "mapping_predicate", "icd10cm_route", "kg_build")


def test_mapping_fields_tuple_is_the_contract() -> None:
    assert MAPPING_FIELDS == _NEW_KEYS


def test_predicates_reuse_the_shared_skos_vocabulary() -> None:
    assert mapping_predicate_for("exact") == "skos:exactMatch"
    assert mapping_predicate_for("broad") == "skos:broadMatch"
    assert mapping_predicate_for("narrow") == "skos:narrowMatch"
    assert mapping_predicate_for("unknown") is None
    assert mapping_predicate_for(None) is None


def test_lipid_row_lifts_relation_to_the_top_level() -> None:
    lipid = {"mapping_relation": "broad", "mapping_predicate": "skos:broadMatch", "ambiguous": False}
    fields = mapping_fields(lipid_resolution=lipid, icd10cm=None)
    assert fields == {
        "mapping_relation": "broad",
        "mapping_predicate": "skos:broadMatch",
        "icd10cm_route": None,
        "kg_build": None,
    }
    assert lipid == {"mapping_relation": "broad", "mapping_predicate": "skos:broadMatch", "ambiguous": False}


def test_icd10cm_fallback_row_carries_route_and_build() -> None:
    fields = mapping_fields(
        lipid_resolution=None, icd10cm={"mapping_relation": "broad", "icd10cm_route": "twin", "kg_build": "2.3.0@abc"}
    )
    assert fields == {
        "mapping_relation": "broad",
        "mapping_predicate": "skos:broadMatch",
        "icd10cm_route": "twin",
        "kg_build": "2.3.0@abc",
    }


def test_plain_row_has_every_new_field_null() -> None:
    assert mapping_fields(lipid_resolution=None, icd10cm=None) == dict.fromkeys(_NEW_KEYS)


# ------------------------------------------------------------------------------------------------
# API result model
# ------------------------------------------------------------------------------------------------
_PLAIN_ITEM: dict[str, Any] = {
    "name": "glucose",
    "curies": ["KEGG.COMPOUND:C00031"],
    "curies_provided": ["KEGG.COMPOUND:C00031"],
    "curies_assigned": {},
    "kg_ids": {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]},
    "kg_ids_assigned": {},
    "chosen_kg_id": "CHEBI:17234",
    "assigned_ids": {},
}


def test_plain_metabolite_api_json_has_null_new_keys_and_unchanged_old_ones() -> None:
    payload = json.loads(extract_mapping_result(dict(_PLAIN_ITEM), "glucose").model_dump_json())
    for key in _NEW_KEYS:
        assert key in payload and payload[key] is None
    assert payload["chosen_kg_id"] == "CHEBI:17234"
    assert payload["lipid_resolution"] is None
    assert payload["kg_ids"] == {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]}


def test_api_result_carries_the_mapper_row_fields() -> None:
    item = dict(_PLAIN_ITEM)
    item.update(
        {"mapping_relation": "broad", "mapping_predicate": "skos:broadMatch", "icd10cm_route": "twin", "kg_build": "b"}
    )
    result = extract_mapping_result(item, "glucose")
    assert (result.mapping_relation, result.mapping_predicate, result.icd10cm_route, result.kg_build) == (
        "broad",
        "skos:broadMatch",
        "twin",
        "b",
    )


# ------------------------------------------------------------------------------------------------
# Certificate mirror
# ------------------------------------------------------------------------------------------------
def _bare_mapper() -> Mapper:
    mapper = Mapper.__new__(Mapper)
    mapper.resolver = MagicMock()
    mapper.resolver.is_small_molecule.return_value = False
    mapper.tier_b = None
    return mapper


def _cert(mapper: Mapper, **extra: Any):
    return mapper._issue_certificate(
        query_name="x",
        category="biolink:Disease",
        chosen_kg_id="MONDO:0010526",
        kg_equivalent_ids={},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        kg_ids_assigned={},
        **extra,
    )


def test_plain_certificate_is_byte_identical_when_no_mapping_provenance() -> None:
    mapper = _bare_mapper()
    assert _cert(mapper).to_api_dict() == _cert(mapper, mapping_provenance=None).to_api_dict()


def test_icd_certificate_mirrors_relation_route_and_build_but_not_ambiguous() -> None:
    cert = _cert(
        _bare_mapper(),
        mapping_provenance={"mapping_relation": "broad", "icd10cm_route": "twin", "kg_build": "2.3.0@abc"},
    ).to_api_dict()
    prov = cert["provenance"]
    assert prov["mapping_relation"] == "broad"
    assert prov["icd10cm_route"] == "twin"
    assert prov["kg_build"] == "2.3.0@abc"
    assert "ambiguous" not in prov


# ------------------------------------------------------------------------------------------------
# Dataset TSV + NDJSON stream
# ------------------------------------------------------------------------------------------------
def test_dataset_tsv_gains_empty_new_columns_on_a_non_icd_row(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", lambda **kw: {c: c for c in kw["batch_items"]})
    from biomapper2.provenance import KgBuildInfo

    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", lambda url: ("unknown", KgBuildInfo()))
    df = pd.DataFrame({"name": ["Fabry disease"], "mondo": ["MONDO:0010526"]})
    out_path, _ = Mapper().map_dataset_to_kg(
        df.copy(),
        entity_type="disease",
        name_column="name",
        provided_id_columns=["mondo"],
        output_dir=tmp_path,
        annotation_mode="none",
    )
    out = pd.read_csv(out_path, sep="\t")
    cols = list(out.columns)
    assert cols[-len(_NEW_KEYS) :] == list(_NEW_KEYS)
    for key in _NEW_KEYS:
        assert out[key].isna().all()
    assert out.loc[0, "chosen_kg_id"] == "MONDO:0010526"


def test_dataset_preserves_input_columns_that_share_an_output_name(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", lambda **kw: {c: c for c in kw["batch_items"]})
    from biomapper2.provenance import KgBuildInfo

    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", lambda url: ("unknown", KgBuildInfo()))
    df = pd.DataFrame(
        {"name": ["Fabry disease"], "mondo": ["MONDO:0010526"], "kg_build": ["mine"], "mapping_relation": ["x"]}
    )
    out_path, _ = Mapper().map_dataset_to_kg(
        df.copy(),
        entity_type="disease",
        name_column="name",
        provided_id_columns=["mondo"],
        output_dir=tmp_path,
        annotation_mode="none",
    )
    out = pd.read_csv(out_path, sep="\t")
    assert out.loc[0, "kg_build_input"] == "mine"
    assert out.loc[0, "mapping_relation_input"] == "x"
    assert out["kg_build"].isna().all()
    assert out["mapping_relation"].isna().all()


def test_preserve_input_columns_refuses_to_clobber_an_existing_input_copy() -> None:
    from biomapper2.mapper import _preserve_input_columns

    df = pd.DataFrame({"kg_build": ["a"], "kg_build_input": ["b"]})
    with pytest.raises(ValueError, match="kg_build_input"):
        _preserve_input_columns(df, ("kg_build",))


@pytest.fixture
def stream_client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        fa = c.app
        assert isinstance(fa, FastAPI)
        original = fa.state.mapper
        mapper = MagicMock()
        item = dict(_PLAIN_ITEM)
        item.update({"mapping_relation": "broad", "mapping_predicate": "skos:broadMatch", "icd10cm_route": "twin"})
        mapper.map_entity_to_kg.return_value = item
        fa.state.mapper = mapper
        try:
            yield c
        finally:
            fa.state.mapper = original


def test_ndjson_stream_emits_the_new_keys(stream_client: TestClient) -> None:
    resp = stream_client.post(
        "/api/v1/map/dataset/stream?entity_type=metabolite&name_column=name&provided_id_columns=kegg",
        files={"file": ("in.csv", io.BytesIO(b"name,kegg\nx,\n"), "text/csv")},
    )
    assert resp.status_code == 200
    line = json.loads(resp.text.splitlines()[0])
    assert "error" not in line
    assert line["mapping_relation"] == "broad"
    assert line["mapping_predicate"] == "skos:broadMatch"
    assert line["icd10cm_route"] == "twin"
    assert line["kg_build"] is None
