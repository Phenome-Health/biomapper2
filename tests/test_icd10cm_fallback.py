"""Unit 3: the ICD-10-CM same-code twin retry in the Linker.

Since v1.3.0 a provided ``ICD10CM:<code>`` keeps its prefix, and KRAKEN has no ``ICD10CM`` nodes, so the
code comes back unresolved. The fallback asks for ``ICD10:<code>`` IN THE SAME batched /canonicalize
call and, per row, writes a twin hit back under the ORIGINAL ``ICD10CM:`` key, so voting and
``kg_ids_provided`` need no change. Tiers: direct > twin. A fallback lookup that is not a definite null
is an error, never a silent unresolved (R7a).

Kestrel is mocked from recorded live answers (``tests/fixtures/icd10cm_twin_regression.json``, written
by studies/analysis/icd10cm_twin_control.py); live checks are in test_icd10cm_fallback_live.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from biomapper2 import config
from biomapper2.config import KESTREL_BATCH_SIZE_CANONICALIZE
from biomapper2.core import icd10cm_fallback
from biomapper2.core.linker import Linker
from biomapper2.mapper import Mapper
from biomapper2.provenance import KgBuildInfo

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((REPO_ROOT / "tests" / "fixtures" / "icd10cm_twin_regression.json").read_text())
ARTIFACT_PATH = REPO_ROOT / "studies" / "analysis" / "results" / "icd10cm_twin_control.json"

FABRY = "MONDO:0010526"
HYPERTENSION = "MONDO:0005044"
BUILD = KgBuildInfo(kg_version="2.3.0", git_commit="3dd08a5b7b17e320ceb0c0091e09b781d4d90242")


class _KestrelDouble:
    """A /canonicalize double that answers like live Kestrel: explicit null for a miss."""

    def __init__(self, answers: dict[str, str | None]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, **kw: Any) -> dict[str, Any]:
        items = list(kw["batch_items"])
        self.calls.append(items)
        if kw.get("endpoint") == "canonicalize":
            return {c: self.answers.get(c) for c in items}
        return {}  # /get-nodes enrichment: nothing


def _kestrel_from(answers: dict[str, str | None]) -> _KestrelDouble:
    return _KestrelDouble(answers)


def _live_like_answers() -> dict[str, str | None]:
    answers: dict[str, str | None] = {}
    for code, rec in FIXTURE["codes"].items():
        answers[f"ICD10:{code}"] = rec["kg_id"]
        answers[f"ICD10CM:{code}"] = None
    for code in ("E11.65", "F32.9"):
        answers[f"ICD10:{code}"] = None
        answers[f"ICD10CM:{code}"] = None
    answers[FABRY] = FABRY
    answers["MONDO:0000001"] = "MONDO:0000001"
    return answers


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch):
    icd10cm_fallback.reset()
    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", lambda url: ("0.5.0", BUILD))
    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", True)
    yield
    icd10cm_fallback.reset()


@pytest.fixture
def kestrel(monkeypatch: pytest.MonkeyPatch):
    fake = _kestrel_from(_live_like_answers())
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)
    return fake


@pytest.fixture(scope="module")
def mapper() -> Mapper:
    return Mapper()


def _map(mapper: Mapper, entity_type: str = "disease", **ids: str) -> dict[str, Any]:
    item = {"name": "x", **ids}
    result = mapper.map_entity_to_kg(
        item, name_field="name", provided_id_fields=list(ids), entity_type=entity_type, annotation_mode="none"
    )
    assert isinstance(result, dict)
    return result


# ------------------------------------------------------------------------------------------------
# Regression fixture (written first; red on the pre-fallback code)
# ------------------------------------------------------------------------------------------------
def test_every_regression_code_links_to_its_recorded_node(kestrel) -> None:
    codes = sorted(FIXTURE["codes"])
    df = pd.DataFrame(
        {
            "curies": [[f"ICD10CM:{c}"] for c in codes],
            "curies_provided": [[f"ICD10CM:{c}"] for c in codes],
            "curies_assigned": [{} for _ in codes],
        }
    )
    linked = Linker().link(df)
    assert isinstance(linked, pd.DataFrame)
    for code, kg_ids_provided in zip(codes, linked["kg_ids_provided"], strict=True):
        assert kg_ids_provided == {FIXTURE["codes"][code]["kg_id"]: [f"ICD10CM:{code}"]}, code


def test_success_criteria_codes_match_the_study_route_labels(mapper: Mapper, kestrel) -> None:
    for code, rec in FIXTURE["success_criteria_routes"].items():
        result = _map(mapper, icd10cm=code)
        if rec["route"] == "twin":
            assert result["chosen_kg_id"] == rec["twin_node"], code
            assert result["icd10cm_route"] == "twin", code
        else:
            assert result["chosen_kg_id"] is None, code
            assert result["icd10cm_route"] is None, code


# ------------------------------------------------------------------------------------------------
# Happy paths
# ------------------------------------------------------------------------------------------------
def test_fabry_resolves_through_the_twin_keyed_by_the_original_curie(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E75.21")
    assert result["chosen_kg_id"] == FABRY
    assert result["kg_ids_provided"] == {FABRY: ["ICD10CM:E75.21"]}
    assert result["kg_ids"] == {FABRY: ["ICD10CM:E75.21"]}
    assert result["mapping_relation"] == config.ICD10CM_TWIN_MAPPING_RELATION
    assert result["icd10cm_route"] == "twin"
    assert result["kg_build"] == "2.3.0@3dd08a5b7b17e320ceb0c0091e09b781d4d90242"
    # The internal side-map never leaks into the mapped entity.
    assert icd10cm_fallback.SIDE_MAP_FIELD not in result


def test_hypertension_resolves_via_twin(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="I10")
    assert result["chosen_kg_id"] == HYPERTENSION
    assert result["icd10cm_route"] == "twin"


def test_fallback_row_certificate_mirrors_relation_route_and_build(mapper: Mapper, kestrel) -> None:
    prov = _map(mapper, icd10cm="E75.21")["resolution_certificate"]["provenance"]
    assert prov["mapping_relation"] == config.ICD10CM_TWIN_MAPPING_RELATION
    assert prov["icd10cm_route"] == "twin"
    assert prov["kg_build"].startswith("2.3.0@")


# ------------------------------------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------------------------------------
def test_direct_hit_on_another_curie_wins_and_fallback_is_ignored(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E75.21", mondo="MONDO:0000001")
    assert result["kg_ids"] == {"MONDO:0000001": ["MONDO:0000001"]}
    assert result["chosen_kg_id"] == "MONDO:0000001"
    assert result["mapping_relation"] is None
    assert result["icd10cm_route"] is None
    assert result["kg_build"] is None


def test_fallback_applies_whatever_the_entity_type(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, entity_type="biolink:NamedThing", icd10cm="E75.21")
    assert result["chosen_kg_id"] == FABRY
    assert result["icd10cm_route"] == "twin"


def test_twin_miss_is_unresolved_with_no_route(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E11.65")
    assert result["chosen_kg_id"] is None
    assert result["mapping_relation"] is None
    assert result["icd10cm_route"] is None
    # Both forms were asked in the same call, so the null is a definite answer.
    assert any("ICD10:E11.65" in call and "ICD10CM:E11.65" in call for call in kestrel.calls)


def test_direct_icd10cm_hit_from_a_future_kraken_is_used_directly(mapper: Mapper, monkeypatch) -> None:
    answers = _live_like_answers()
    answers["ICD10CM:E75.21"] = "MONDO:9999999"
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", _kestrel_from(answers))
    result = _map(mapper, icd10cm="E75.21")
    assert result["chosen_kg_id"] == "MONDO:9999999"
    assert result["mapping_relation"] is None
    assert result["icd10cm_route"] is None


def test_denylisted_code_is_emitted_broad(mapper: Mapper, kestrel, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ICD10CM_TWIN_MAPPING_RELATION", "exact")
    monkeypatch.setattr(icd10cm_fallback, "twin_denylist", lambda: frozenset({"E75.21"}))
    assert _map(mapper, icd10cm="E75.21")["mapping_relation"] == "broad"
    assert _map(mapper, icd10cm="I10")["mapping_relation"] == "exact"


def test_shared_cache_rows_are_order_independent(kestrel) -> None:
    rows = [
        {"curies": ["ICD10CM:E75.21", "MONDO:0000001"], "curies_provided": ["ICD10CM:E75.21", "MONDO:0000001"]},
        {"curies": ["ICD10CM:E75.21"], "curies_provided": ["ICD10CM:E75.21"]},
    ]
    outputs = []
    for order in (rows, list(reversed(rows))):
        df = pd.DataFrame([{**r, "curies_assigned": {}} for r in order])
        linked = Linker().link(df)
        assert isinstance(linked, pd.DataFrame)
        outputs.append({tuple(r["curies"]): lk for r, lk in zip(order, linked["kg_ids"], strict=True)})
    assert outputs[0] == outputs[1]
    by_curies = outputs[0]
    assert by_curies[("ICD10CM:E75.21", "MONDO:0000001")] == {"MONDO:0000001": ["MONDO:0000001"]}
    assert by_curies[("ICD10CM:E75.21",)] == {FABRY: ["ICD10CM:E75.21"]}


# ------------------------------------------------------------------------------------------------
# Error paths
# ------------------------------------------------------------------------------------------------
def _http_error(status: int) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"{status}", response=response)


def test_kestrel_failure_raises_on_the_entity_path(mapper: Mapper, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: Any):
        raise _http_error(503)

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", boom)
    with pytest.raises(requests.exceptions.HTTPError):
        _map(mapper, icd10cm="E75.21")


def test_kestrel_failure_fails_the_dataset_run(mapper: Mapper, monkeypatch, tmp_path) -> None:
    def boom(**_: Any):
        raise _http_error(503)

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", boom)
    with pytest.raises(requests.exceptions.HTTPError):
        mapper.map_dataset_to_kg(
            pd.DataFrame({"name": ["x"], "icd10cm": ["E75.21"]}),
            entity_type="disease",
            name_column="name",
            provided_id_columns=["icd10cm"],
            output_dir=tmp_path,
            annotation_mode="none",
        )


def test_bisect_isolated_twin_raises_instead_of_reading_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    import biomapper2.utils as utils

    def bulk(method, endpoint, session=None, **kw):
        curies = kw["json"]["curies"]
        if "ICD10:E75.21" in curies:
            raise _http_error(500)
        return {c: None for c in curies}

    monkeypatch.setattr(utils, "bulk_kestrel_request", bulk)
    monkeypatch.setattr(utils, "KESTREL_BISECT_ON_5XX_ENABLED", True)
    monkeypatch.setattr(utils.time, "sleep", lambda s: None)
    entity = {"curies": ["ICD10CM:E75.21"], "curies_provided": ["ICD10CM:E75.21"], "curies_assigned": {}}
    with pytest.raises(icd10cm_fallback.Icd10cmFallbackLookupError):
        Linker().link(entity)


# ------------------------------------------------------------------------------------------------
# Switch
# ------------------------------------------------------------------------------------------------
def test_switch_off_restores_pre_fallback_behavior(mapper: Mapper, kestrel, monkeypatch) -> None:
    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", False)
    result = _map(mapper, icd10cm="E75.21")
    assert result["chosen_kg_id"] is None
    assert result["kg_ids"] == {}
    assert result["mapping_relation"] is None
    assert all("ICD10:E75.21" not in call for call in kestrel.calls)
    entity = {"curies": ["ICD10CM:E75.21"], "curies_provided": ["ICD10CM:E75.21"], "curies_assigned": {}}
    linked = Linker().link(entity)
    assert isinstance(linked, pd.Series)
    assert icd10cm_fallback.SIDE_MAP_FIELD not in linked.index


# ------------------------------------------------------------------------------------------------
# Call count: candidates ride the existing batch
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_codes", [3, 1500])
def test_call_count_is_one_batched_canonicalize_per_chunk(monkeypatch: pytest.MonkeyPatch, n_codes: int) -> None:
    import biomapper2.utils as utils

    calls: list[int] = []

    def bulk(method, endpoint, session=None, **kw):
        curies = kw["json"]["curies"]
        calls.append(len(curies))
        return {c: None for c in curies}

    monkeypatch.setattr(utils, "bulk_kestrel_request", bulk)
    codes = [f"Z{i // 100:02d}.{i % 100:02d}" for i in range(n_codes)]
    direct = [f"ICD10CM:{c}" for c in codes]
    df = pd.DataFrame(
        {
            "curies": [[d] for d in direct],
            "curies_provided": [[d] for d in direct],
            "curies_assigned": [{} for _ in direct],
        }
    )
    Linker().link(df)
    expected = math.ceil((len(direct) * 2) / KESTREL_BATCH_SIZE_CANONICALIZE)
    assert len(calls) == expected
    if n_codes == 1500:
        assert expected == 2  # crosses the chunk boundary


def test_non_icd_rows_make_the_same_calls_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _kestrel_from({"MONDO:0000001": "MONDO:0000001"})
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)
    entity = {"curies": ["MONDO:0000001"], "curies_provided": ["MONDO:0000001"], "curies_assigned": {}}
    linked = Linker().link(entity)
    assert fake.calls == [["MONDO:0000001"]]
    assert isinstance(linked, pd.Series)
    assert linked[icd10cm_fallback.SIDE_MAP_FIELD] == {}


# ------------------------------------------------------------------------------------------------
# The shipped relation is bound to the committed gate artifact
# ------------------------------------------------------------------------------------------------
def test_twin_relation_constant_matches_the_gate_artifact() -> None:
    from studies.analysis import icd10cm_twin_control as study

    artifact = json.loads(ARTIFACT_PATH.read_text())
    gate = artifact["gate"]
    assert gate["threshold"] == study.GATE_THRESHOLD
    assert gate["min_n"] == study.GATE_MIN_N
    # Until Trent records a review of the gate, the only acceptable shipped value is broad.
    expected = study.twin_relation_for(gate, reviewed=bool(gate.get("reviewed")))
    assert config.ICD10CM_TWIN_MAPPING_RELATION == expected
    if not gate.get("reviewed"):
        assert config.ICD10CM_TWIN_MAPPING_RELATION == "broad"


def test_gate_kg_build_constant_matches_the_artifact() -> None:
    kg = json.loads(ARTIFACT_PATH.read_text())["kg_build"]
    assert config.ICD10CM_TWIN_GATE_KG_BUILD == f"{kg['kg_version']}@{kg['git_commit']}"


def test_packaged_denylist_matches_the_artifact() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    assert icd10cm_fallback.twin_denylist() == frozenset(artifact["denylist"]["codes"])


def test_kg_build_mismatch_with_the_gate_build_warns_once(monkeypatch, caplog) -> None:
    other = KgBuildInfo(kg_version="2.4.0", git_commit="deadbeef")
    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", lambda url: ("0.5.0", other))
    with caplog.at_level("WARNING"):
        assert icd10cm_fallback.kg_build_stamp() == "2.4.0@deadbeef"
        icd10cm_fallback.kg_build_stamp()
    assert sum("ICD10CM twin gate" in r.getMessage() for r in caplog.records) == 1


def test_failed_kg_build_lookup_is_retried_not_cached_forever(monkeypatch) -> None:
    calls = {"n": 0}
    answers = [KgBuildInfo(), BUILD]

    def fetch(url):
        calls["n"] += 1
        return "0.5.0", answers[min(calls["n"] - 1, 1)]

    clock = {"t": 1000.0}
    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", fetch)
    monkeypatch.setattr(icd10cm_fallback.time, "monotonic", lambda: clock["t"])
    assert icd10cm_fallback.kg_build_stamp() == "unknown@unknown"
    assert icd10cm_fallback.kg_build_stamp() == "unknown@unknown"  # within the retry window: no refetch
    assert calls["n"] == 1
    clock["t"] += icd10cm_fallback._KG_BUILD_RETRY_SECONDS
    assert icd10cm_fallback.kg_build_stamp() == f"{BUILD.kg_version}@{BUILD.git_commit}"
    assert icd10cm_fallback.kg_build_stamp() == f"{BUILD.kg_version}@{BUILD.git_commit}"
    assert calls["n"] == 2


# ------------------------------------------------------------------------------------------------
# Row fields follow the COMMITTED node (re-resolution swaps recompute them)
# ------------------------------------------------------------------------------------------------
def test_row_fields_describe_only_a_node_backed_by_fallback_curies() -> None:
    from biomapper2.mapper import _icd10cm_fields_for

    side_map = {
        "ICD10CM:A00": {"route": "twin", "mapping_relation": "exact", "kg_id": "MONDO:1"},
        "ICD10CM:A01": {"route": "twin", "mapping_relation": "broad", "kg_id": "MONDO:1"},
    }
    kg_ids = {"MONDO:1": ["ICD10CM:A00", "ICD10CM:A01"], "CHEBI:2": ["CHEBI:2"]}
    fields = _icd10cm_fields_for("MONDO:1", kg_ids, side_map)
    assert fields is not None
    assert fields["mapping_relation"] == "broad"  # the weakest supporting relation
    assert fields["icd10cm_route"] == "twin"
    # A swap to a node no fallback curie backs carries no fallback fields.
    assert _icd10cm_fields_for("CHEBI:2", kg_ids, side_map) is None
    assert _icd10cm_fields_for(None, kg_ids, side_map) is None
