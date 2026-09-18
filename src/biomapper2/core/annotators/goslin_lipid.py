"""Goslin lipid-shorthand annotator (Biolink SmallMolecule route, parse-success = lipid detector).

Two internal stages keep measurement honest:
  1. PARSE/NORMALIZE (offline, deterministic): pygoslin turns a messy/dialect shorthand into a
     canonical shorthand + formula + mass + dialect. A parse MISS returns ``{}`` (fail-soft) so a
     SmallMolecule non-lipid falls through to the other annotators unchanged.
  2. IDENTIFIER BINDING (lookup): the parse exposes a name at every lipid level pygoslin can render,
     and the binder loop CASCADES those names from the input's effective query level down to species
     across the sources (RefMet /match, LIPID MAPS enrichment, Kestrel hybrid search). The first
     level that yields a hit per source wins, and each resulting vote records the level it matched at.
     Bound ids are re-keyed under this annotator's slug and each carries the Goslin metadata.

The effective query level follows the sn-position trust policy (config.LIPID_TRUST_SN_POSITION,
default OFF): shorthand written with "/" claims a proven sn-position, but vendors use it loosely, so
by default an sn-position input is DOWNGRADED to molecular-species FOR QUERYING. Both the asserted
level (the input's real level) and the effective level (after the policy) are recorded.

LIPID MAPS REST enrichment is an INJECTED, OFF-BY-DEFAULT seam (the circular path vs LMSD): only when
a caller supplies ``enrichment`` are LM_ID/InChIKey filled from it, and the metadata records that it
fired so any number it touched is flagged as coverage, not independent accuracy. The Kestrel source is
likewise injected: when present it is queried only for cascade levels no other source already hit
(D3), staying within the caller's ``candidate_limit``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from ... import config
from ...utils import AssignedIDsDict
from .base import BaseAnnotator
from .goslin_grammar import LipidGrammar, LipidParse
from .lipidmaps_rest import LipidEnricher
from .metabolomics_workbench import MetabolomicsWorkbenchAnnotator

# A single inner vote block: ``{vocab: {local_or_curie_id: metadata_dict}}``.
_Votes = dict[str, dict[str, dict[str, Any]]]

# Lipid shorthand levels ordered MOST specific -> LEAST specific. The cascade walks this order from the
# effective query level down to species, so a more specific match is tried before a broader one. Only
# species and finer levels appear here; CATEGORY / CLASS are coarser than species and never queried.
_LEVEL_ORDER_MOST_TO_LEAST: tuple[str, ...] = (
    "COMPLETE_STRUCTURE",
    "FULL_STRUCTURE",
    "STRUCTURE_DEFINED",
    "SN_POSITION",
    "MOLECULAR_SPECIES",
    "SPECIES",
)
_SPECIES = "SPECIES"
_SN_POSITION = "SN_POSITION"
_MOLECULAR_SPECIES = "MOLECULAR_SPECIES"


class GoslinLipidAnnotator(BaseAnnotator):
    """Normalize lipid shorthand with Goslin, then cascade the level names to KG-native ids."""

    slug = "goslin-lipid"

    def __init__(
        self,
        grammar: LipidGrammar | None = None,
        binder: BaseAnnotator | None = None,
        enrichment: LipidEnricher | None = None,
        kestrel: BaseAnnotator | None = None,
        trust_sn_position: bool | None = None,
    ) -> None:
        self._grammar = grammar if grammar is not None else LipidGrammar()
        # Default binder is the RefMet /match annotator; each level name is what it receives.
        self._binder = binder if binder is not None else MetabolomicsWorkbenchAnnotator()
        self._enrichment = enrichment  # None => LIPID MAPS REST enrichment OFF (accuracy config)
        # Injected Kestrel hybrid-search source (D3). None => the cascade never calls Kestrel; the
        # engine's separately registered Kestrel annotator (raw name) is unaffected either way.
        self._kestrel = kestrel
        # Trust policy for "/" (D1). None defers to the config flag captured at import; a test injects
        # a bool directly. When False, an sn-position input is downgraded to molecular-species FOR
        # QUERYING (see _effective_level).
        self._trust_sn = config.LIPID_TRUST_SN_POSITION if trust_sn_position is None else trust_sn_position

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
        candidate_limit: int | None = None,
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

        asserted = self._asserted_level(parsed)
        effective = self._effective_level(asserted)
        cascade = self._cascade_levels(parsed, effective)

        # Source 1: RefMet /match via the injected name binder. First cascade level with a vote wins.
        refmet_votes, refmet_level = self._cascade_source(
            cascade,
            lambda level_name: self._binder_query(
                entity,
                name_field,
                level_name,
                category,
                prefixes,
                prefer_human,
                preferred_prefixes,
                accepted_categories,
                candidate_limit,
            ),
        )

        # Source 2: LIPID MAPS enrichment (off unless injected). First cascade level with a candidate.
        enrich_votes: _Votes = {}
        enrich_level: str | None = None
        if self._enrichment is not None:
            enrich_votes, enrich_level = self._cascade_source(cascade, self._enrichment_query)

        # Source 3: Kestrel hybrid search (off unless injected), ONLY for levels no other source hit (D3).
        # Fail-soft: Kestrel is the last source and the least essential (RefMet is the accuracy path), so
        # any error in its cascade degrades to no Kestrel vote rather than discarding the RefMet / LIPID
        # MAPS votes already gathered above. This annotator's whole contract is fail-soft.
        already_hit = {level for level in (refmet_level, enrich_level) if level is not None}
        kestrel_votes: _Votes = {}
        kestrel_level: str | None = None
        if self._kestrel is not None:
            try:
                kestrel_votes, kestrel_level = self._cascade_source(
                    cascade,
                    lambda level_name: self._kestrel_query(
                        entity,
                        name_field,
                        level_name,
                        category,
                        prefixes,
                        prefer_human,
                        preferred_prefixes,
                        accepted_categories,
                        candidate_limit,
                    ),
                    skip_levels=already_hit,
                )
            except Exception:  # noqa: BLE001 — Kestrel is best-effort; a failure must not lose other votes
                logging.warning("goslin-lipid Kestrel cascade failed for %r (ignored, other votes kept)", name)
                kestrel_votes, kestrel_level = {}, None

        base_meta = self._base_metadata(parsed, asserted, effective, enrichment_fired=bool(enrich_votes))

        inner: _Votes = {}
        self._merge_votes(inner, refmet_votes, refmet_level, base_meta)
        self._merge_votes(inner, enrich_votes, enrich_level, base_meta)
        self._merge_votes(inner, kestrel_votes, kestrel_level, base_meta)

        if not inner:
            # Parsed as a lipid but nothing bound at any level: still fail-soft (no wrong commit).
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
        candidate_limit: int | None = None,
    ) -> pd.Series:
        """Implements BaseAnnotator.get_annotations_bulk (rowwise; each source handles its own cache).

        Call volume, stated rather than hidden: the cascade runs PER ROW, so a lipid panel costs on the
        order of rows x levels x sources network calls. This is inherent to level-aware matching and is
        the ratified behavior (D3: more lookups to try more specific levels), bounded by stop-at-first-hit
        and skip-already-hit. Request batching across rows is deferred to a later plan, not done here.

        Fail-soft PER ROW: one row's source failure degrades that row to an empty vote (like a non-lipid),
        it never aborts the whole batch. So a single bad name in a dataset job cannot lose every other
        row's mapping.
        """
        col = entities.apply(
            self._row_annotations_fail_soft,
            axis=1,
            name_field=name_field,
            category=category,
            prefixes=prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
            # Re-dispatches into get_annotations, which forwards it to every source — omitting it would
            # drop the candidate window on every lipid dataset job.
            candidate_limit=candidate_limit,
        )
        return cast(pd.Series, col)

    def _row_annotations_fail_soft(self, entity: pd.Series, **kwargs: Any) -> AssignedIDsDict:
        """One row's ``get_annotations``, but any exception degrades to an empty vote so the batch
        survives. The single-entity path calls ``get_annotations`` directly and is unaffected."""
        try:
            return self.get_annotations(entity, **kwargs)
        except Exception:  # noqa: BLE001 — one bad row must not abort the whole lipid panel
            logging.warning("goslin-lipid row annotation failed for %r (ignored)", entity.get(kwargs["name_field"]))
            return {}

    # ---------------------------------------- Cascade helpers ---------------------------------------- #

    @staticmethod
    def _cascade_source(
        cascade: list[tuple[str, str]],
        query: Callable[[str], _Votes],
        skip_levels: frozenset[str] | set[str] = frozenset(),
    ) -> tuple[_Votes, str | None]:
        """Walk the cascade (most specific first) and return the FIRST non-empty vote block, plus the
        level (LipidLevel name) it matched at. ``skip_levels`` are passed over without a query (D3)."""
        for level, rendered in cascade:
            if level in skip_levels:
                continue
            votes = query(rendered)
            if votes:
                return votes, level
        return {}, None

    def _binder_query(
        self,
        entity: dict | pd.Series,
        name_field: str,
        level_name: str,
        category: str,
        prefixes: list[str] | None,
        prefer_human: bool,
        preferred_prefixes: set[str] | None,
        accepted_categories: set[str] | None,
        candidate_limit: int | None,
    ) -> _Votes:
        rewritten = self._with_name(entity, name_field, level_name)
        bound = self._binder.get_annotations(
            rewritten,
            name_field,
            category,
            prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
            candidate_limit=candidate_limit,
        )
        return dict(bound.get(self._binder.slug, {}))

    def _kestrel_query(
        self,
        entity: dict | pd.Series,
        name_field: str,
        level_name: str,
        category: str,
        prefixes: list[str] | None,
        prefer_human: bool,
        preferred_prefixes: set[str] | None,
        accepted_categories: set[str] | None,
        candidate_limit: int | None,
    ) -> _Votes:
        assert self._kestrel is not None  # guarded by the caller
        rewritten = self._with_name(entity, name_field, level_name)
        bound = self._kestrel.get_annotations(
            rewritten,
            name_field,
            category,
            prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
            candidate_limit=candidate_limit,
        )
        return dict(bound.get(self._kestrel.slug, {}))

    def _enrichment_query(self, level_name: str) -> _Votes:
        """LIPID MAPS candidates for one level name, as a vote block. Every candidate contributes its
        LM_ID and InChIKey (order-independent, multi-row aware per the resolver's set contract)."""
        assert self._enrichment is not None  # guarded by the caller
        candidates, ok = self._enrichment.candidates_checked(level_name)
        if not ok or not candidates:
            return {}
        votes: _Votes = {}
        for cand in candidates:
            lm_id = cand.get("lm_id")
            inchikey = cand.get("inchi_key")
            if lm_id:
                votes.setdefault("LIPIDMAPS", {}).setdefault(lm_id, {})
            if inchikey:
                votes.setdefault("INCHIKEY", {}).setdefault(inchikey, {})
        return votes

    @staticmethod
    def _merge_votes(inner: _Votes, votes: _Votes, matched_level: str | None, base_meta: dict[str, Any]) -> None:
        """Merge one source's votes into ``inner``, stamping the shared Goslin metadata plus the level
        THIS source matched at. A source's own per-vote metadata (e.g. a Kestrel score) is preserved."""
        meta = dict(base_meta)
        if matched_level is not None:
            meta["matched_level"] = matched_level.lower()
        for vocab, id_map in votes.items():
            dest = inner.setdefault(vocab, {})
            for id_, vote_meta in id_map.items():
                existing = dest.setdefault(id_, {})
                existing.update(vote_meta)  # keep the source's own metadata (score, resolved_via, ...)
                existing.update(meta)  # then the Goslin + matched-level metadata

    # ---------------------------------------- Level policy ------------------------------------------- #

    def _asserted_level(self, parsed: LipidParse) -> str:
        """The input's real level (LipidLevel name). Prefers the parsed grammar level; when that is
        unavailable or coarser than species, falls back to the most specific level pygoslin rendered
        (which equals the input level, since a coarser input cannot render a finer name)."""
        name = self._normalize_level(parsed.level)
        if name in _LEVEL_ORDER_MOST_TO_LEAST:
            return name
        for level in _LEVEL_ORDER_MOST_TO_LEAST:
            if level in parsed.level_names:
                return level
        return _SPECIES

    def _effective_level(self, asserted: str) -> str:
        """Apply the sn-position trust policy (D1): with trust OFF, an input asserted at sn-position OR
        FINER is capped at molecular-species for querying (vendors use "/" loosely, and the finer levels
        STRUCTURE_DEFINED / FULL_STRUCTURE / COMPLETE_STRUCTURE also carry the "/" sn claim). Capping,
        not just an sn-position equality check, so a slash-bearing structurally-detailed name never
        escapes the policy. MOLECULAR_SPECIES and SPECIES stay as-is. With trust ON, effective=asserted."""
        if self._trust_sn:
            return asserted
        order = _LEVEL_ORDER_MOST_TO_LEAST
        try:
            asserted_index = order.index(asserted)
        except ValueError:
            return asserted
        # Lower index == more specific; sn-position or finer is at/above SN_POSITION's specificity.
        if asserted_index <= order.index(_SN_POSITION):
            return _MOLECULAR_SPECIES
        return asserted

    @staticmethod
    def _cascade_levels(parsed: LipidParse, effective: str) -> list[tuple[str, str]]:
        """The ``(level_name, rendered_name)`` pairs to query, most specific first: every level at or
        below the effective specificity that pygoslin actually rendered. Species is always included so
        the cascade never empties (the canonical name backstops a parse with no rendered species)."""
        order = _LEVEL_ORDER_MOST_TO_LEAST
        try:
            start = order.index(effective)
        except ValueError:
            start = order.index(_SPECIES)
        levels: list[tuple[str, str]] = []
        for level in order[start:]:
            rendered = parsed.level_names.get(level)
            if rendered:
                levels.append((level, rendered))
        if not levels:
            levels = [(_SPECIES, parsed.canonical_name)]
        return levels

    @staticmethod
    def _normalize_level(raw: str | None) -> str | None:
        """``"LipidLevel.SN_POSITION"`` -> ``"SN_POSITION"``; ``None`` -> ``None``."""
        if not raw:
            return None
        return str(raw).rsplit(".", 1)[-1].strip().upper()

    # ---------------------------------------- Assembly ----------------------------------------------- #

    @staticmethod
    def _with_name(entity: dict | pd.Series, name_field: str, level_name: str) -> dict:
        """Copy of the entity with ``name_field`` rewritten to a level-specific shorthand."""
        base = dict(entity) if isinstance(entity, dict) else entity.to_dict()
        base[name_field] = level_name
        return base

    @staticmethod
    def _base_metadata(parsed: LipidParse, asserted: str, effective: str, enrichment_fired: bool) -> dict[str, Any]:
        return {
            "goslin_canonical": parsed.canonical_name,
            "goslin_formula": parsed.sum_formula,
            "goslin_mass": parsed.monoisotopic_mass,
            "goslin_dialect": parsed.dialect,
            "goslin_level": parsed.level,
            "goslin_input_level": parsed.level,
            "goslin_level_names": dict(parsed.level_names),
            "goslin_chains": list(parsed.chains),
            "query_lipid_level_asserted": asserted.lower(),
            "query_lipid_level_effective": effective.lower(),
            "lipidmaps_rest_enrichment_fired": enrichment_fired,
        }
