"""Tests for the engine's category-acceptance policy resolution (the accepted_categories path).

Mirrors ``test_annotation_engine_canonical.py``: a MagicMock BiolinkClient supplies a controllable
``get_descendants`` (no bmt init, no network) and a spy annotator records what the engine resolved
and passed down.

The distinguishing property versus ``preferred_prefixes``: category acceptance is a **correctness
guard, not a re-ranking preference**, so it is independent of both ``prefer_canonical`` and
``prefer_human`` — it must not inherit the canonical policy's kill switch.
"""

from unittest.mock import MagicMock

import pandas as pd

from biomapper2.core.annotation_engine import AnnotationEngine
from biomapper2.core.annotators.base import BaseAnnotator

# ``get_descendants('biolink:ChemicalEntity')`` under Biolink 4.2.5 (n=12, verified against bmt).
CHEMICAL = {
    "biolink:ChemicalEntity",
    "biolink:ChemicalMixture",
    "biolink:ComplexMolecularMixture",
    "biolink:Drug",
    "biolink:EnvironmentalFoodContaminant",
    "biolink:Food",
    "biolink:FoodAdditive",
    "biolink:MolecularEntity",
    "biolink:MolecularMixture",
    "biolink:NucleicAcidEntity",
    "biolink:ProcessedMaterial",
    "biolink:SmallMolecule",
}


class _SpyAnnotator(BaseAnnotator):
    slug = "spy"

    def __init__(self):
        self.received: list[set[str] | None] = []

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
        self.received.append(accepted_categories)
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
        self.received.append(accepted_categories)
        return pd.Series([{self.slug: {}} for _ in range(len(entities))], index=entities.index)


def _descendants(category):
    """Reflexive by default, with the real ChemicalEntity subtree for the acceptance root."""
    return CHEMICAL if category == "biolink:ChemicalEntity" else {category}


def _engine(descendants_side_effect=None):
    bc = MagicMock()
    bc.get_descendants.side_effect = descendants_side_effect or _descendants
    engine = AnnotationEngine(biolink_client=bc)
    spy = _SpyAnnotator()
    engine.annotator_registry["spy"] = spy
    return engine, spy


def _resolve(engine, spy, category, item=None, **kwargs):
    engine.annotate(
        item={"name": "x"} if item is None else item,
        name_field="name",
        provided_id_fields=[],
        category=category,
        prefixes=[],
        mode="all",
        annotators=["spy"],
        **kwargs,
    )
    return spy.received[-1]


class TestAcceptanceMapResolution:
    def test_small_molecule_maps_to_the_chemical_subtree(self):
        engine, _ = _engine()
        assert engine._category_accepted_categories["biolink:SmallMolecule"] == CHEMICAL

    def test_configured_key_is_expanded_via_descendants(self):
        """Subcategories of the configured key inherit the acceptance root, like the prefix policy."""
        engine, _ = _engine(
            descendants_side_effect=lambda c: (
                CHEMICAL
                if c == "biolink:ChemicalEntity"
                else {"biolink:SmallMolecule", "biolink:Subtype"} if c == "biolink:SmallMolecule" else {c}
            )
        )
        assert engine._category_accepted_categories["biolink:Subtype"] == CHEMICAL

    def test_metabolite_receives_the_acceptance_set(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule") == CHEMICAL

    def test_independent_of_prefer_canonical(self):
        """A correctness guard must not inherit the re-ranking policy's kill switch.

        This also closes the ``limit`` question raised by the pool-filter design: because the
        validator runs on the already-committed node rather than on the candidate window, decoupling
        from ``prefer_canonical`` cannot produce a one-row-window coverage cliff.
        """
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule", prefer_canonical=False) == CHEMICAL

    def test_independent_of_prefer_human(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule", prefer_human=False) == CHEMICAL

    def test_gene_receives_none(self):
        """Gene/protein are intentionally absent from the map (nearly all hgnc commits are off-category)."""
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:Gene", prefer_human=True) is None

    def test_unconfigured_category_receives_none(self):
        """Unmapped categories are unfiltered — including the ``biolink:NamedThing`` fallback that
        ``standardize_entity_type`` returns for an unrecognized entity type."""
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:Disease") is None
        assert _resolve(engine, spy, "biolink:NamedThing") is None

    def test_bulk_path_forwards_the_acceptance_set(self):
        engine, spy = _engine()
        assert _resolve(engine, spy, "biolink:SmallMolecule", item=pd.DataFrame({"name": ["a", "b"]})) == CHEMICAL
