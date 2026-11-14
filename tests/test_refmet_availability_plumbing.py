"""Availability plumbing: engine accumulation, mapper forwarding, and the resolver split guard.

Mocked only — no live APIs. Two properties:
- U2: the engine emits a TOTAL per-row availability map (every registered annotator, every path
  including skips), surfaced single-path via the Series / dataset-path via a column; the mapper reads
  RefMet's status and forwards it into ``issue()``.
- U6: the resolver SELECTION code is unchanged. A genuine no-match (empty RefMet vote) resolves as
  today (R6a); a row rescued UNAVAILABLE->VOTED shifts ``chosen_kg_id`` toward the RefMet node (R6b).
  The direction is asserted; invariance is not claimed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from biomapper2.core.annotation_engine import AnnotationEngine
from biomapper2.core.annotators.base import (
    AVAILABILITY_NOT_QUERIED,
    AVAILABILITY_UNAVAILABLE,
    BaseAnnotator,
)
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from biomapper2.core.certificate import REFMET_ANNOTATOR
from biomapper2.core.resolver import Resolver

CATEGORY = "biolink:SmallMolecule"


class _StatusAnnotator(BaseAnnotator):
    """A no-vote annotator that reports a fixed availability status, to exercise accumulation."""

    slug = "fake"

    def __init__(self, status: str) -> None:
        self._status = status

    def get_annotations(
        self,
        entity,
        name_field,
        category,
        prefixes=None,
        prefer_human=True,
        preferred_prefixes=None,
        accepted_categories=None,
        candidate_limit=None,
        cache=None,
    ):
        return {self.slug: {}}

    def get_annotations_bulk(
        self,
        entities,
        name_field,
        category,
        prefixes=None,
        prefer_human=True,
        preferred_prefixes=None,
        accepted_categories=None,
        candidate_limit=None,
        cache=None,
    ):
        return pd.Series([{self.slug: {}} for _ in range(len(entities))], index=entities.index)

    def get_availability(self, entity, name_field, cache=None):
        return {self.slug: self._status}


def _engine() -> AnnotationEngine:
    bc = MagicMock()
    bc.get_descendants.side_effect = lambda c: {c}
    return AnnotationEngine(biolink_client=bc)


def test_engine_emits_total_not_queried_map_when_mode_is_none():
    engine = _engine()
    result = engine.annotate(
        item={"name": "x"}, name_field="name", provided_id_fields=[], category=CATEGORY, prefixes=[], mode="none"
    )
    assert isinstance(result, pd.Series)
    availability = result["annotator_availability"]
    assert set(availability) == set(engine.annotator_registry)
    assert all(v == AVAILABILITY_NOT_QUERIED for v in availability.values())


def test_engine_accumulates_a_real_status_and_leaves_the_rest_not_queried():
    engine = _engine()
    engine.annotator_registry["fake"] = _StatusAnnotator(AVAILABILITY_UNAVAILABLE)
    result = engine.annotate(
        item={"name": "x"},
        name_field="name",
        provided_id_fields=[],
        category=CATEGORY,
        prefixes=[],
        mode="all",
        annotators=["fake"],
    )
    assert isinstance(result, pd.Series)
    availability = result["annotator_availability"]
    assert availability["fake"] == AVAILABILITY_UNAVAILABLE
    # Total: RefMet is still present (not_queried — it was not selected for this row).
    assert availability[REFMET_ANNOTATOR] == AVAILABILITY_NOT_QUERIED


def test_engine_emits_total_map_for_a_provided_id_only_row():
    engine = _engine()
    result = engine.annotate(
        item={"name": "x", "chebi": "CHEBI:1"},
        name_field="name",
        provided_id_fields=["chebi"],
        category=CATEGORY,
        prefixes=[],
        mode="missing",
    )
    assert isinstance(result, pd.Series)
    availability = result["annotator_availability"]
    assert set(availability) == set(engine.annotator_registry)
    assert all(v == AVAILABILITY_NOT_QUERIED for v in availability.values())


def test_engine_dataset_path_adds_a_total_availability_column():
    engine = _engine()
    df = pd.DataFrame({"name": ["a", "b"]})
    result = engine.annotate(
        item=df, name_field="name", provided_id_fields=[], category=CATEGORY, prefixes=[], mode="none"
    )
    assert isinstance(result, pd.DataFrame)
    assert "annotator_availability" in result.columns
    for cell in result["annotator_availability"]:
        assert set(cell) == set(engine.annotator_registry)
        assert all(v == AVAILABILITY_NOT_QUERIED for v in cell.values())


class _Mapper:
    """The narrowest object exposing ``_issue_certificate``, bound to a stub resolver, Tier B off."""

    def __init__(self):
        from biomapper2.mapper import Mapper

        self.tier_b = None
        self.resolver = SimpleNamespace(is_small_molecule=lambda category: True)
        self._issue_certificate = Mapper._issue_certificate.__get__(self, _Mapper)


def test_mapper_forwards_refmet_status_into_issue():
    """The status the engine computed reaches the certificate via ``issue()``'s kwarg."""
    mapper = _Mapper()
    certificate = mapper._issue_certificate(
        query_name="retinol",
        category=CATEGORY,
        chosen_kg_id="CHEBI:1",
        kg_equivalent_ids={"INCHIKEY": ["AAAAAAAAAAAAAA-UHFFFAOYSA-N"]},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        kg_ids_assigned={},
        refmet_availability=AVAILABILITY_UNAVAILABLE,
    )
    assert certificate.refmet_availability == AVAILABILITY_UNAVAILABLE


# --------------------------------------------------------------------------------------------
# U6 — resolver split guard. Harness mirrors tests/test_resolver_source_weighting.py.
# --------------------------------------------------------------------------------------------

# BioMapper's node wins the naive vote; RefMet, when it votes, anchors a different node.
KG_IDS = {"CHEBI:bmp": ["a", "b"], "CHEBI:refmet": ["RM:1"]}
REFMET_VOTE = {REFMET_ANNOTATOR: {"CHEBI:refmet": ["RM:1"]}}
# NO_MATCH and UNAVAILABLE both surface as an EMPTY RefMet vote (the status lives elsewhere).
EMPTY_REFMET_VOTE = {REFMET_ANNOTATOR: {}}


def _resolver(conn: bool | None) -> Resolver:
    r = Resolver(linker=MagicMock(), biolink_client=MagicMock())
    r.structure_resolver = MagicMock()
    r.structure_resolver.connectivity_match.return_value = conn
    r._is_small_molecule = lambda category: category == CATEGORY
    return r


def test_r6a_no_match_row_resolves_exactly_as_today():
    """An empty RefMet vote (genuine no-match, or unavailable) falls to the majority, unflagged."""
    chosen, flag = _resolver(False)._choose_best_kg_id(KG_IDS, EMPTY_REFMET_VOTE, CATEGORY)
    assert (chosen, flag) == ("CHEBI:bmp", None)


def test_r6b_rescued_row_shifts_choice_toward_the_refmet_node():
    """A row UNAVAILABLE-then-VOTED gains a RefMet vote, which the resolver source-weights toward.

    Asserts the DIRECTION (majority -> RefMet-anchored node), not invariance: this is the intended
    coverage restoration, and it does change ``chosen_kg_id`` on rescued rows.
    """
    unrescued, _ = _resolver(False)._choose_best_kg_id(KG_IDS, EMPTY_REFMET_VOTE, CATEGORY)
    rescued, flag = _resolver(False)._choose_best_kg_id(KG_IDS, REFMET_VOTE, CATEGORY)
    assert unrescued == "CHEBI:bmp"
    assert rescued == "CHEBI:refmet"
    assert flag == "divergent_refmet"


def test_route_armed_deadline_survives_nested_fetch_all():
    # The API /batch route arms the shared deadline once, then each entity re-enters _fetch_all via
    # build_availability_cache. A nested _fetch_all must NOT disarm the outer deadline, or the whole
    # -batch bound would be wiped after the first entity (Greptile round-2 finding).
    ann = MetabolomicsWorkbenchAnnotator(batch_deadline_s=999.0)
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM"}
    assert ann.arm_batch_deadline() is True  # outer arm (the route)
    outer_deadline = ann._batch_deadline
    ann.build_availability_cache({"name": "x"}, "name")  # inner _fetch_all: arms->False, no disarm
    assert ann._batch_deadline == outer_deadline  # outer deadline survived the nested call
    assert ann.arm_batch_deadline() is False  # still armed
    ann.disarm_batch_deadline()
    assert ann._batch_deadline is None


# --------------------------------------------------------------------------------------------
# Kept vote (source ``kept_live_no_match``): the label flows annotator -> engine ->
# mapper -> certificate -> batch counts with no change in shape.
# --------------------------------------------------------------------------------------------

KEPT = "kept_live_no_match"


def test_engine_surfaces_a_kept_vote_as_voted_with_the_kept_label(monkeypatch, tmp_path):
    from biomapper2.core.annotators import refmet_store

    monkeypatch.setenv("REFMET_STORE_PATH", str(tmp_path / "store.sqlite"))
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    refmet_store.upsert("docosahexaenoate (dha; 22:6n3)", "voted", refmet_id="RM0153631")
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: None  # live /match now says no-match

    engine = _engine()
    engine.annotator_registry[REFMET_ANNOTATOR] = ann
    result = engine.annotate(
        item={"name": "docosahexaenoate (dha; 22:6n3)"},
        name_field="name",
        provided_id_fields=[],
        category=CATEGORY,
        prefixes=[],
        mode="all",
        annotators=[REFMET_ANNOTATOR],
    )
    assert isinstance(result, pd.Series)
    assert result["annotator_availability"][REFMET_ANNOTATOR] == "voted"
    assert result["annotator_source"][REFMET_ANNOTATOR] == KEPT
    assert result["assigned_ids"][REFMET_ANNOTATOR] == {"refmet_id": {"RM0153631": {}}}


def _offline_mapper_with_refmet_source(refmet_source: str):
    """Real ``map_entity_to_kg`` + real Resolver over mocked stages (as in test_source_weighting_e2e).

    The engine reports a RefMet vote with the given source; the linker puts that vote on its own node.
    """
    from biomapper2.mapper import Mapper

    m = Mapper.__new__(Mapper)
    m.biolink_client = MagicMock()
    m.biolink_client.standardize_entity_type = lambda t: CATEGORY
    m.biolink_client.get_descendants = lambda c: {CATEGORY}
    m.annotation_engine = MagicMock()
    m.annotation_engine.annotate = lambda **k: pd.Series(
        {
            "annotator_availability": {REFMET_ANNOTATOR: "voted"},
            "annotator_source": {REFMET_ANNOTATOR: refmet_source},
        }
    )
    m.normalizer = MagicMock()
    m.normalizer.get_standard_prefix = lambda v: None
    m.normalizer.normalize = lambda **k: pd.Series(dtype=object)
    m.linker = MagicMock()
    m.linker.link = lambda s: pd.Series({"kg_ids": KG_IDS, "kg_ids_provided": {}, "kg_ids_assigned": REFMET_VOTE})
    m.linker.get_equivalent_ids_checked = lambda ids, prefixes=None: ({}, True)
    m.tier_b = None
    m.resolver = Resolver(linker=m.linker, biolink_client=m.biolink_client)
    m.resolver.structure_resolver = MagicMock()
    m.resolver.structure_resolver.connectivity_match.return_value = True  # same molecule: no flag
    return m


def _map_one(refmet_source: str) -> dict:
    out = _offline_mapper_with_refmet_source(refmet_source).map_entity_to_kg(
        {"name": "docosahexaenoate (dha; 22:6n3)"}, name_field="name", provided_id_fields=[], entity_type="metabolite"
    )
    assert isinstance(out, dict)
    return out


def test_kept_vote_reaches_the_result_and_certificate_and_resolves_like_a_live_vote():
    kept = _map_one(KEPT)
    live = _map_one("live_api")

    assert kept["refmet_availability"] == "voted"
    assert kept["refmet_source"] == KEPT
    certificate = kept["resolution_certificate"]
    assert certificate["refmet_availability"] == "voted"
    assert certificate["refmet_source"] == KEPT
    # A kept vote is an ordinary RefMet vote to the resolver: same node and same (absent) review flag
    # as the same row served live, and a certificate that differs in the source label only.
    assert kept["chosen_kg_id"] == live["chosen_kg_id"] == "CHEBI:refmet"
    assert kept["chosen_kg_id_review"] is None and live["chosen_kg_id_review"] is None
    live_certificate = dict(live["resolution_certificate"], refmet_source=KEPT)
    assert certificate == live_certificate


def test_kept_vote_certificate_has_no_snapshot_version_like_a_freeze_backup_row(monkeypatch):
    from biomapper2.core.annotators import refmet_snapshot

    # Even with a freeze loaded: the mapper stamps the version by source label, and neither label is
    # in the snapshot-attributable set.
    monkeypatch.setattr(refmet_snapshot, "version", lambda: "mw_match_fixture")
    kept = _map_one(KEPT)
    backup = _map_one("freeze_backup")
    assert kept["refmet_snapshot_version"] is None and backup["refmet_snapshot_version"] is None
    assert kept["resolution_certificate"]["refmet_snapshot_version"] is None
    assert backup["resolution_certificate"]["refmet_snapshot_version"] is None
    assert _map_one("local_snapshot")["refmet_snapshot_version"] == "mw_match_fixture"  # positive control


def test_batch_summary_counts_a_kept_vote_under_its_own_label():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from biomapper2.api.main import app

    sources = ["live_api", KEPT, "live_api"]
    mapper = MagicMock()
    mapper.map_entity_to_kg.side_effect = [
        {"name": f"m{i}", "chosen_kg_id": "CHEBI:1", "refmet_availability": "voted", "refmet_source": source}
        for i, source in enumerate(sources)
    ]
    with TestClient(app) as client:
        fa = client.app
        assert isinstance(fa, FastAPI)
        original = fa.state.mapper
        fa.state.mapper = mapper
        try:
            response = client.post(
                "/api/v1/map/batch",
                json={"entities": [{"name": f"m{i}", "entity_type": "metabolite"} for i in range(len(sources))]},
            )
        finally:
            fa.state.mapper = original
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["refmet_source_counts"] == {"live_api": 2, KEPT: 1}
    assert [r["refmet_source"] for r in body["results"]] == sources
