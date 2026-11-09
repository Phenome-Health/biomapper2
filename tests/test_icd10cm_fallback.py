"""Units 3 and 5: the ICD-10-CM same-code twin retry and the parent walk-up in the Linker.

Since v1.3.0 a provided ``ICD10CM:<code>`` keeps its prefix, and KRAKEN has no ``ICD10CM`` nodes, so the
code comes back unresolved. The fallback asks for ``ICD10:<code>`` IN THE SAME batched /canonicalize
call and, per row, writes a twin hit back under the ORIGINAL ``ICD10CM:`` key, so voting and
``kg_ids_provided`` need no change. When the twin also misses, the walk-up climbs to the most specific
existing ancestor in the pinned CDC snapshot (asked in the same batch, as ``ICD10CM:`` then ``ICD10:``)
and emits it ``broad``. Tiers: direct > twin > walk-up. A fallback lookup that is not a definite null is
an error, never a silent unresolved (R7a).

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
from biomapper2.core import icd10cm_fallback, icd10cm_hierarchy
from biomapper2.core.linker import Linker
from biomapper2.mapper import Mapper
from biomapper2.provenance import KgBuildInfo

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((REPO_ROOT / "tests" / "fixtures" / "icd10cm_twin_regression.json").read_text())
ARTIFACT_PATH = REPO_ROOT / "studies" / "analysis" / "results" / "icd10cm_twin_control.json"

FABRY = "MONDO:0010526"
HYPERTENSION = "MONDO:0005044"
T2DM = "MONDO:0005148"
DEPRESSION = "MONDO:0002009"
# Phase 1b success criteria: the two codes the twin misses, and the ancestor the walk-up reaches.
WALKUP_EXPECTED = {"E11.65": (T2DM, "walkup:E11"), "F32.9": (DEPRESSION, "walkup:F32")}
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
    icd10cm_hierarchy.reset()
    monkeypatch.setattr("biomapper2.provenance.fetch_kg_build_info", lambda url: ("0.5.0", BUILD))
    monkeypatch.setattr(config, "ICD10CM_FALLBACK_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", True)
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", None)
    yield
    icd10cm_fallback.reset()
    icd10cm_hierarchy.reset()


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


@pytest.mark.parametrize("walkup", [True, False])
def test_success_criteria_codes_match_the_study_route_labels(mapper: Mapper, kestrel, monkeypatch, walkup) -> None:
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", walkup)
    for code, rec in FIXTURE["success_criteria_routes"].items():
        result = _map(mapper, icd10cm=code)
        if rec["route"] == "twin":
            assert result["chosen_kg_id"] == rec["twin_node"], code
            assert result["icd10cm_route"] == "twin", code
        elif walkup:
            # Twin miss -> Phase 1b walk-up to the most specific existing ancestor, emitted broad.
            node, route = WALKUP_EXPECTED[code]
            assert result["chosen_kg_id"] == node, code
            assert result["icd10cm_route"] == route, code
            assert result["mapping_relation"] == "broad", code
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


def test_twin_miss_is_unresolved_with_no_route_when_walkup_is_off(mapper: Mapper, kestrel, monkeypatch) -> None:
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", False)
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
    # Real CDC codes so the walk-up has existing ancestors to add.
    codes = [c for c in sorted(icd10cm_hierarchy.load_or_raise().codes) if "." in c][:n_codes]
    assert len(codes) == n_codes
    direct = [f"ICD10CM:{c}" for c in codes]
    df = pd.DataFrame(
        {
            "curies": [[d] for d in direct],
            "curies_provided": [[d] for d in direct],
            "curies_assigned": [{} for _ in direct],
        }
    )
    Linker().link(df)
    # Twins plus every existing-ancestor form (ICD10CM: and ICD10:) ride the same batch.
    candidates = icd10cm_fallback.expand_candidates(direct)
    assert any(c.startswith("ICD10CM:") for c in candidates)  # the walk-up added ancestor candidates
    expected = math.ceil((len(direct) + len(candidates)) / KESTREL_BATCH_SIZE_CANONICALIZE)
    assert sum(calls) == len(direct) + len(candidates)
    assert len(calls) == expected
    if n_codes == 1500:
        assert expected >= 2  # crosses the chunk boundary


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


# ================================================================================================
# Unit 5: the parent walk-up (Phase 1b)
# ================================================================================================
def _answers_with(monkeypatch: pytest.MonkeyPatch, extra: dict[str, str | None]) -> _KestrelDouble:
    answers = _live_like_answers()
    answers.update(extra)
    fake = _kestrel_from(answers)
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)
    return fake


def test_e11_65_walks_up_to_t2dm_broad(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E11.65")
    assert result["chosen_kg_id"] == T2DM
    assert result["mapping_relation"] == "broad"
    assert result["mapping_predicate"] == "skos:broadMatch"
    assert result["icd10cm_route"] == "walkup:E11"
    assert result["icd10cm_snapshot_version"] == icd10cm_hierarchy.version()
    assert "FY2027" in result["icd10cm_snapshot_version"]
    assert result["kg_build"] == "2.3.0@3dd08a5b7b17e320ceb0c0091e09b781d4d90242"
    # Keyed by the ORIGINAL curie: the user never supplied ICD10:E11.
    assert result["kg_ids_provided"] == {T2DM: ["ICD10CM:E11.65"]}
    # Every more specific candidate was asked in the same single call (so each null is definite).
    asked = set().union(*map(set, kestrel.calls))
    assert {"ICD10CM:E11.65", "ICD10:E11.65", "ICD10CM:E11.6", "ICD10:E11.6", "ICD10CM:E11", "ICD10:E11"} <= asked


def test_f32_9_walks_up_to_depression_broad(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="F32.9")
    assert result["chosen_kg_id"] == DEPRESSION
    assert result["mapping_relation"] == "broad"
    assert result["icd10cm_route"] == "walkup:F32"


def test_walkup_certificate_mirrors_route_and_snapshot_version(mapper: Mapper, kestrel) -> None:
    prov = _map(mapper, icd10cm="E11.65")["resolution_certificate"]["provenance"]
    assert prov["mapping_relation"] == "broad"
    assert prov["icd10cm_route"] == "walkup:E11"
    assert prov["icd10cm_snapshot_version"] == icd10cm_hierarchy.version()


def test_most_specific_ancestor_wins(mapper: Mapper, monkeypatch) -> None:
    _answers_with(monkeypatch, {"ICD10:E11.6": "MONDO:0000116"})
    result = _map(mapper, icd10cm="E11.65")
    assert result["chosen_kg_id"] == "MONDO:0000116"
    assert result["icd10cm_route"] == "walkup:E11.6"


def test_icd10cm_form_preferred_over_icd10_form_within_a_level(mapper: Mapper, monkeypatch) -> None:
    _answers_with(monkeypatch, {"ICD10CM:E11.6": "MONDO:0000106", "ICD10:E11.6": "MONDO:0000116"})
    result = _map(mapper, icd10cm="E11.65")
    assert result["chosen_kg_id"] == "MONDO:0000106"
    assert result["icd10cm_route"] == "walkup:E11.6"


def test_twin_hit_means_the_walkup_is_never_used(mapper: Mapper, kestrel) -> None:
    # ICD10:E75.2 (an ancestor of E75.21) also links in the fixture; the twin tier still wins.
    result = _map(mapper, icd10cm="E75.21")
    assert result["chosen_kg_id"] == FABRY
    assert result["icd10cm_route"] == "twin"
    assert result["mapping_relation"] == config.ICD10CM_TWIN_MAPPING_RELATION
    assert result["icd10cm_snapshot_version"] is None


def test_direct_hit_on_another_curie_ignores_the_walkup(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E11.65", mondo="MONDO:0000001")
    assert result["chosen_kg_id"] == "MONDO:0000001"
    assert result["kg_ids"] == {"MONDO:0000001": ["MONDO:0000001"]}
    assert result["icd10cm_route"] is None
    assert result["icd10cm_snapshot_version"] is None


def test_code_not_in_snapshot_gets_the_twin_only(mapper: Mapper, kestrel, monkeypatch) -> None:
    assert not icd10cm_hierarchy.contains("E11.99")
    result = _map(mapper, icd10cm="E11.99")
    assert result["chosen_kg_id"] is None
    assert result["icd10cm_route"] == "twin_only:not_in_snapshot"
    assert result["mapping_relation"] is None
    assert result["icd10cm_snapshot_version"] is None
    asked = set().union(*map(set, kestrel.calls))
    assert "ICD10:E11.99" in asked and "ICD10:E11" not in asked
    # ...and a twin hit for such a code is still a plain twin hit.
    _answers_with(monkeypatch, {"ICD10:E11.99": "MONDO:0000199"})
    hit = _map(mapper, icd10cm="E11.99")
    assert hit["chosen_kg_id"] == "MONDO:0000199"
    assert hit["icd10cm_route"] == "twin"


def test_multi_code_row_twin_tier_beats_walkup(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E75.21", icd10cm_2="E11.65")
    assert result["chosen_kg_id"] == FABRY
    assert result["kg_ids"] == {FABRY: ["ICD10CM:E75.21"]}
    assert result["icd10cm_route"] == "twin"
    assert result["icd10cm_snapshot_version"] is None


def test_multi_code_walkup_only_row_votes_normally(mapper: Mapper, kestrel) -> None:
    result = _map(mapper, icd10cm="E11.65", icd10cm_2="E11.69", icd10cm_3="F32.9")
    assert result["kg_ids"] == {T2DM: ["ICD10CM:E11.65", "ICD10CM:E11.69"], DEPRESSION: ["ICD10CM:F32.9"]}
    assert result["chosen_kg_id"] == T2DM
    assert result["mapping_relation"] == "broad"
    assert result["icd10cm_route"] == "walkup:E11"


def test_walkup_switch_off_gives_twin_only_behavior(mapper: Mapper, kestrel, monkeypatch) -> None:
    monkeypatch.setattr(config, "ICD10CM_WALKUP_ENABLED", False)
    result = _map(mapper, icd10cm="E11.65")
    assert result["chosen_kg_id"] is None
    assert result["icd10cm_route"] is None
    assert result["icd10cm_snapshot_version"] is None
    asked = set().union(*map(set, kestrel.calls))
    assert "ICD10:E11.65" in asked
    assert not any(c in asked for c in ("ICD10CM:E11.6", "ICD10:E11.6", "ICD10CM:E11", "ICD10:E11"))
    # A twin hit is untouched by the walk-up switch.
    assert _map(mapper, icd10cm="E75.21")["icd10cm_route"] == "twin"
    # And a not-in-snapshot code records no walk-up route either.
    assert _map(mapper, icd10cm="E11.99")["icd10cm_route"] is None


def test_walkup_dataset_path_stamps_snapshot_version_only_on_walkup_rows(mapper: Mapper, kestrel, tmp_path) -> None:
    df = pd.DataFrame({"name": ["a", "b", "c"], "icd10cm": ["E11.65", "E75.21", None]})
    out = mapper.map_dataset_to_kg(
        df,
        entity_type="disease",
        name_column="name",
        provided_id_columns=["icd10cm"],
        output_dir=tmp_path,
        annotation_mode="none",
    )
    mapped = out[0] if isinstance(out, tuple) else out
    if not isinstance(mapped, pd.DataFrame):
        mapped = pd.read_csv(mapped, sep="\t")
    rows = mapped.set_index("name")
    assert rows.loc["a", "chosen_kg_id"] == T2DM
    assert rows.loc["a", "icd10cm_route"] == "walkup:E11"
    assert rows.loc["a", "icd10cm_snapshot_version"] == icd10cm_hierarchy.version()
    assert rows.loc["b", "icd10cm_route"] == "twin"
    assert pd.isna(rows.loc["b", "icd10cm_snapshot_version"])
    assert pd.isna(rows.loc["c", "icd10cm_snapshot_version"])


# ------------------------------------------------------------------------------------------------
# R7a with the walk-up: a broad candidate commits only when every more specific one is a definite null
# ------------------------------------------------------------------------------------------------
def _bisect_isolating(monkeypatch: pytest.MonkeyPatch, poison: str, hits: dict[str, str]) -> None:
    import biomapper2.utils as utils

    def bulk(method, endpoint, session=None, **kw):
        curies = kw["json"]["curies"]
        if poison in curies:
            raise _http_error(500)
        return {c: hits.get(c) for c in curies}

    monkeypatch.setattr(utils, "bulk_kestrel_request", bulk)
    monkeypatch.setattr(utils, "KESTREL_BISECT_ON_5XX_ENABLED", True)
    monkeypatch.setattr(utils.time, "sleep", lambda s: None)


def _entity(*codes: str) -> dict[str, Any]:
    curies = [f"ICD10CM:{c}" for c in codes]
    return {"curies": curies, "curies_provided": curies, "curies_assigned": {}}


def test_isolated_more_specific_ancestor_raises_instead_of_falling_through(monkeypatch) -> None:
    _bisect_isolating(monkeypatch, "ICD10:E11.6", {"ICD10:E11": T2DM})
    with pytest.raises(icd10cm_fallback.Icd10cmFallbackLookupError, match="ICD10:E11.6"):
        Linker().link(_entity("E11.65"))


def test_isolated_ancestor_below_the_committed_hit_does_not_raise(monkeypatch) -> None:
    # ICD10CM:E11.6 linked, so the isolated, LESS specific ICD10:E11 is never consulted.
    _bisect_isolating(monkeypatch, "ICD10:E11", {"ICD10CM:E11.6": "MONDO:0000106"})
    linked = Linker().link(_entity("E11.65"))
    assert isinstance(linked, pd.Series)
    assert linked["kg_ids"] == {"MONDO:0000106": ["ICD10CM:E11.65"]}


def test_isolated_ancestor_is_irrelevant_when_the_twin_hit(monkeypatch) -> None:
    _bisect_isolating(monkeypatch, "ICD10:E75.2", {"ICD10:E75.21": FABRY})
    linked = Linker().link(_entity("E75.21"))
    assert isinstance(linked, pd.Series)
    assert linked["kg_ids"] == {FABRY: ["ICD10CM:E75.21"]}


def test_omitted_more_specific_ancestor_raises_instead_of_falling_through(monkeypatch) -> None:
    # A successful /canonicalize that omits ICD10CM:E11.6 (no key, not an explicit null) must not let
    # the walk-up commit the broader ICD10:E11 hit.
    omitted = "ICD10CM:E11.6"

    def partial(**kw: Any) -> dict[str, Any]:
        if kw.get("endpoint") != "canonicalize":
            return {}
        return {c: ({"ICD10:E11": T2DM}.get(c)) for c in kw["batch_items"] if c != omitted}

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", partial)
    with pytest.raises(icd10cm_fallback.Icd10cmFallbackLookupError, match="ICD10CM:E11.6"):
        Linker().link(_entity("E11.65"))


def test_omitted_twin_raises_instead_of_falling_through_to_walkup(monkeypatch) -> None:
    # A successful /canonicalize that omits the twin ICD10:E11.65 must not let the walk-up commit ICD10:E11.
    omitted = "ICD10:E11.65"

    def partial(**kw: Any) -> dict[str, Any]:
        if kw.get("endpoint") != "canonicalize":
            return {}
        return {c: ({"ICD10:E11": T2DM}.get(c)) for c in kw["batch_items"] if c != omitted}

    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", partial)
    with pytest.raises(icd10cm_fallback.Icd10cmFallbackLookupError, match="ICD10:E11.65"):
        Linker().link(_entity("E11.65"))


def test_walkup_enabled_with_a_missing_snapshot_raises_at_link_time(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "ICD10CM_HIERARCHY_PATH", str(tmp_path / "gone.tsv"))
    fake = _kestrel_from(_live_like_answers())
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)
    with pytest.raises(icd10cm_hierarchy.Icd10cmHierarchyError):
        Linker().link(_entity("E11.65"))
    assert fake.calls == []  # failed before any lookup, never degraded to "no parent"


def test_non_icd_rows_are_untouched_by_the_walkup(monkeypatch) -> None:
    fake = _kestrel_from({"MONDO:0000001": "MONDO:0000001"})
    monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)
    entity = {"curies": ["MONDO:0000001"], "curies_provided": ["MONDO:0000001"], "curies_assigned": {}}
    Linker().link(entity)
    assert fake.calls == [["MONDO:0000001"]]
