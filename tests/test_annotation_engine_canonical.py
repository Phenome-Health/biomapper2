"""Tests for the engine's canonical-namespace policy resolution (prefer_canonical gating).

Uses a MagicMock BiolinkClient so no bmt/network init is needed: get_descendants returns a controllable
descendant set. A spy annotator records the preferred_prefixes the engine resolves and passes down.
"""

from unittest.mock import MagicMock

import pandas as pd

from biomapper2.core.annotation_engine import AnnotationEngine
from biomapper2.core.annotators.base import BaseAnnotator


class _SpyAnnotator(BaseAnnotator):
    slug = "spy"

    def __init__(self):
        self.received = []

    def get_annotations(
        self,
        entity,
        name_field,
        category,
        prefixes=None,
        prefer_human=True,
        preferred_prefixes=None,
        accepted_categories=None,
        cache=None,
    ):
        self.received.append(preferred_prefixes)
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
    ):
        self.received.append(preferred_prefixes)
        return pd.Series([{self.slug: {}} for _ in range(len(entities))], index=entities.index)


def _engine(descendants_side_effect=None):
    bc = MagicMock()
    bc.get_descendants.side_effect = descendants_side_effect or (lambda c: {c})
    engine = AnnotationEngine(biolink_client=bc)
    spy = _SpyAnnotator()
    engine.annotator_registry["spy"] = spy
    return engine, spy


def _resolve(engine, spy, category, prefer_canonical=True):
    engine.annotate(
        item={"name": "x"},
        name_field="name",
        provided_id_fields=[],
        category=category,
        prefixes=[],
        mode="all",
        annotators=["spy"],
        prefer_canonical=prefer_canonical,
    )
    return spy.received[-1]


class TestCanonicalPolicyResolution:
    def test_preferred_prefixes_map_expands_via_descendants(self):
        engine, _ = _engine()
        assert engine._category_preferred_prefixes["biolink:SmallMolecule"] == {"CHEBI", "HMDB", "RM"}
        assert engine._category_preferred_prefixes["biolink:Disease"] == {"MONDO"}

    def test_metabolite_receives_canonical_set(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule") == {"CHEBI", "HMDB", "RM"}

    def test_disease_descendant_inherits_policy(self):
        # A disease subcategory is a descendant of biolink:Disease -> inherits {MONDO}.
        engine, spy = _engine(
            descendants_side_effect=lambda c: {"biolink:Disease", "biolink:Mass"} if c == "biolink:Disease" else {c}
        )
        assert _resolve(engine, spy, "biolink:Mass") == {"MONDO"}

    def test_gene_receives_none(self):
        # biolink:Gene is human-applicable -> prefer_human wins, no canonical set.
        engine, spy = _engine(descendants_side_effect=lambda c: {"biolink:Gene"} if c == "biolink:Gene" else {c})
        assert _resolve(engine, spy, "biolink:Gene") is None

    def test_prefer_canonical_false_yields_none(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule", prefer_canonical=False) is None

    def test_unconfigured_category_yields_none(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:OrganismTaxon") is None

    def test_precedence_human_wins_when_category_in_both_sets(self):
        # A category that is BOTH a gene descendant AND in a preferred-namespace key resolves to None
        # (prefer_human precedence) — locks the `not effective_prefer_human` ordering.
        def descendants(c):
            if c == "biolink:Gene":
                return {"biolink:Gene", "biolink:Overlap"}
            if c == "biolink:SmallMolecule":
                return {"biolink:SmallMolecule", "biolink:Overlap"}
            return {c}

        engine, spy = _engine(descendants_side_effect=descendants)
        assert _resolve(engine, spy, "biolink:Overlap") is None
