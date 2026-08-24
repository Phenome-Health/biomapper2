"""Goslin lipid-shorthand annotator (Biolink SmallMolecule route, parse-success = lipid detector).

Two internal stages keep measurement honest:
  1. PARSE/NORMALIZE (offline, deterministic): pygoslin turns a messy/dialect shorthand into a
     canonical shorthand + formula + mass + dialect. A parse MISS returns ``{}`` (fail-soft) so a
     SmallMolecule non-lipid falls through to the other annotators unchanged.
  2. IDENTIFIER BINDING (lookup): the canonical name is handed to the existing name binder (the
     RefMet /match annotator), which previously failed on the RAW shorthand and now matches. The
     bound ids are re-keyed under this annotator's slug and each carries the Goslin metadata.

LIPID MAPS REST enrichment is an INJECTED, OFF-BY-DEFAULT seam (the circular path vs LMSD): only when
a caller supplies ``enrichment`` are LM_ID/InChIKey filled from it, and the metadata records that it
fired so any number it touched is flagged as coverage, not independent accuracy.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from ...utils import AssignedIDsDict
from .base import BaseAnnotator
from .goslin_grammar import LipidGrammar, LipidParse
from .lipidmaps_rest import LipidEnricher
from .metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class GoslinLipidAnnotator(BaseAnnotator):
    """Normalize lipid shorthand with Goslin, then bind the canonical name to KG-native ids."""

    slug = "goslin-lipid"

    def __init__(
        self,
        grammar: LipidGrammar | None = None,
        binder: BaseAnnotator | None = None,
        enrichment: LipidEnricher | None = None,
    ) -> None:
        self._grammar = grammar if grammar is not None else LipidGrammar()
        # Default binder is the RefMet /match annotator; the canonical name is what it receives.
        self._binder = binder if binder is not None else MetabolomicsWorkbenchAnnotator()
        self._enrichment = enrichment  # None => LIPID MAPS REST enrichment OFF (accuracy config)

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations. Returns ``{}`` for non-lipids (fail-soft)."""
        name = entity.get(name_field)
        if not name:
            return {}

        parsed = self._grammar.parse(str(name))
        if parsed is None:
            # Not lipid shorthand — fall through unchanged (the parse-success lipid detector).
            return {}

        # Stage 2: bind the CANONICAL name via the existing name binder.
        rewritten = self._with_name(entity, name_field, parsed.canonical_name)
        bound = self._binder.get_annotations(
            rewritten,
            name_field,
            category,
            prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
        )
        inner: dict[str, dict[str, dict[str, Any]]] = dict(bound.get(self._binder.slug, {}))

        enrichment_fired = False
        if self._enrichment is not None:
            for vocab, id_ in self._enrichment.enrich(parsed.canonical_name).items():
                enrichment_fired = True
                inner.setdefault(vocab, {}).setdefault(id_, {})

        metadata = self._metadata(parsed, enrichment_fired)
        for id_map in inner.values():
            for meta in id_map.values():
                meta.update(metadata)

        if not inner:
            # Parsed as a lipid but nothing bound: still fail-soft (no wrong commit), report empty.
            return {}
        return {self.slug: inner}

    def get_annotations_bulk(
        self,
        entities: pd.DataFrame,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
    ) -> pd.Series:
        """Implements BaseAnnotator.get_annotations_bulk (rowwise; the binder handles its own cache)."""
        col = entities.apply(
            self.get_annotations,
            axis=1,
            name_field=name_field,
            category=category,
            prefixes=prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
        )
        return cast(pd.Series, col)

    @staticmethod
    def _with_name(entity: dict | pd.Series, name_field: str, canonical: str) -> dict:
        """Copy of the entity with ``name_field`` rewritten to the canonical shorthand."""
        base = dict(entity) if isinstance(entity, dict) else entity.to_dict()
        base[name_field] = canonical
        return base

    @staticmethod
    def _metadata(parsed: LipidParse, enrichment_fired: bool) -> dict[str, Any]:
        return {
            "goslin_canonical": parsed.canonical_name,
            "goslin_formula": parsed.sum_formula,
            "goslin_mass": parsed.monoisotopic_mass,
            "goslin_dialect": parsed.dialect,
            "goslin_level": parsed.level,
            "lipidmaps_rest_enrichment_fired": enrichment_fired,
        }
