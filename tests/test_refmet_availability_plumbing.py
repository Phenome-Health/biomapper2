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
