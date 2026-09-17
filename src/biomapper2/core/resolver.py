"""
One-to-many resolution module for selecting single KG nodes.

Resolves cases where multiple KG nodes match an entity by selecting the best candidate.
For small-molecule ChEBI conflicts the naive curie-count vote is source-weighted toward the
RefMet (``metabolomics-workbench``) annotator under an InChIKey-connectivity guard, so the
common protonation/stereo variant conflict is fixed while genuine divergences are flagged
for review rather than committed silently.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..config import CATEGORY_PREFERRED_NAMESPACES
from .certificate import LipidStructureEvidence, compare_lipid_composition, structural_agree
from .structure_resolver import StructureResolver

# The annotator whose vote is authoritative for small-molecule ChEBI conflicts: it queries the
# RefMet /match endpoint by name and emits the RefMet-anchored node.
REFMET_ANNOTATOR = "metabolomics-workbench"

# The lipid-shorthand annotator whose votes carry the ``matched_level`` and effective-query-level
# metadata the Unit 4 level-aware tie-break reads.
GOSLIN_LIPID_ANNOTATOR = "goslin-lipid"

# Review hint set when a lipid tie could only be resolved to a node BROADER than the effective query
# level. Deliberately a NEW, additive resolver-output field, NOT a ``selection_conflict`` value: that
# channel is a closed, small-molecule-only whitelist enforced by the certificate, so widening it would
# be a certificate-contract change. This hint rides its own field and never touches selection_conflict.
LIPID_GENERALIZED_HINT = "lipid_generalized"

# Lipid shorthand levels, MOST specific -> LEAST specific, lowercased to match the goslin annotator's
# stamped ``matched_level`` metadata. Kept local (not imported from core.annotators.goslin_lipid) to
# avoid an annotators -> resolver import edge; mirrors GoslinLipidAnnotator._LEVEL_ORDER_MOST_TO_LEAST.
_LIPID_LEVEL_ORDER: tuple[str, ...] = (
    "complete_structure",
    "full_structure",
    "structure_defined",
    "sn_position",
    "molecular_species",
    "species",
)


def _lipid_level_rank(level: str | None) -> int:
    """Specificity rank of a lipid level: LOWER means MORE specific. Unknown/None sorts as broadest."""
    if level is None:
        return len(_LIPID_LEVEL_ORDER)
    try:
        return _LIPID_LEVEL_ORDER.index(level)
    except ValueError:
        return len(_LIPID_LEVEL_ORDER)


def _raw_level_for_curie(curie: str, level_by_raw: dict[str, str]) -> str | None:
    """A curie's matched level from a map keyed by the annotator's RAW id, reconciling the two forms.

    Normalization rewrites a raw annotator id into a curie in one of two shapes: a vocab prefix glued
    onto the value is split off behind a colon (so stripping the colon rebuilds the raw id -- RefMet and
    LIPID MAPS), or no prefix was present and the curie's local part already equals the raw id (an
    InChIKey). Trying the full curie, the colon-stripped curie, and the local part covers both, so the
    level survives the raw-id vs curie representation gap that a plain local-part join silently drops.
    """
    for candidate in (curie, curie.replace(":", ""), curie.rsplit(":", 1)[-1]):
        level = level_by_raw.get(candidate)
        if level is not None:
            return level
    return None


def _lipid_level_context(entity: "pd.Series | dict[str, Any]") -> tuple[dict[str, str], str | None]:
    """Best-effort ``{kg_id: matched_level}`` plus the effective query level, joined from goslin votes.

    The matched level lives on the goslin-lipid votes' metadata (``assigned_ids``), keyed by the
    annotator's RAW id; the linked KG node ids live in ``kg_ids_assigned``, keyed by the NORMALIZED
    curie that same raw id became. Those two representations differ, so the join reconciles them via
    ``_raw_level_for_curie`` rather than assuming the curie's local part equals the raw id. Returns
    ``({}, None)`` for a non-lipid row (no goslin votes) or when either structure is absent, so the
    tie-break stays inert off the lipid path.
    """
    assigned_ids = entity.get("assigned_ids") or {}
    kg_ids_assigned = entity.get("kg_ids_assigned") or {}
    goslin_meta = assigned_ids.get(GOSLIN_LIPID_ANNOTATOR) or {}
    goslin_kg = kg_ids_assigned.get(GOSLIN_LIPID_ANNOTATOR) or {}
    if not goslin_meta or not goslin_kg:
        return {}, None

    # raw metadata id -> matched_level, unioned across vocabs. The key is the annotator's RAW id (what
    # goslin stamps the metadata under), NOT yet a curie.
    level_by_raw: dict[str, str] = {}
    effective: str | None = None
    for vocab_map in goslin_meta.values():
        if not isinstance(vocab_map, dict):
            continue
        for raw_id, meta in vocab_map.items():
            if not isinstance(meta, dict):
                continue
            matched = meta.get("matched_level")
            if isinstance(matched, str):
                level_by_raw[raw_id] = matched
            if effective is None and isinstance(meta.get("query_lipid_level_effective"), str):
                effective = meta["query_lipid_level_effective"]

    # Each KG node inherits the MOST specific level among the goslin votes its curies reconcile to.
    levels: dict[str, str] = {}
    for kg_id, curies in goslin_kg.items():
        found = [lvl for c in curies if (lvl := _raw_level_for_curie(c, level_by_raw)) is not None]
        if found:
            levels[kg_id] = min(found, key=_lipid_level_rank)
    return levels, effective


def _lipid_generalized_hint(
    chosen_kg_id: str | None, lipid_levels: dict[str, str], effective_level: str | None
) -> str | None:
    """``lipid_generalized`` when the committed lipid node is BROADER than the effective query level -- a
    generalization the resolver could not avoid (R8). None for non-lipid rows and exact/finer matches."""
    if not chosen_kg_id or not lipid_levels or not effective_level:
        return None
    chosen_level = lipid_levels.get(chosen_kg_id)
    if chosen_level is None:
        return None
    if _lipid_level_rank(chosen_level) > _lipid_level_rank(effective_level):
        return LIPID_GENERALIZED_HINT
    return None


# SKOS mapping relation between the COMMITTED lipid node's matched level and the effective query level.
# ``exact`` when they coincide, ``broad`` when the node is coarser (the ``lipid_generalized`` case),
# ``narrow`` when it is finer (should not happen under Decision 2's at-or-above policy, but is
# represented honestly if seen), ``unknown`` when either level is missing.
LIPID_RELATION_EXACT = "exact"
LIPID_RELATION_BROAD = "broad"
LIPID_RELATION_NARROW = "narrow"
LIPID_RELATION_UNKNOWN = "unknown"
_LIPID_RELATION_PREDICATE: dict[str, str] = {
    LIPID_RELATION_EXACT: "skos:exactMatch",
    LIPID_RELATION_BROAD: "skos:broadMatch",
    LIPID_RELATION_NARROW: "skos:narrowMatch",
}

# ``query_transformed`` values. A trust-off downgrade (asserted finer than effective) means the input's
# slash-bearing name was queried in its underscore form; otherwise the query rode goslin's canonical
# species-level rendering.
QUERY_TRANSFORMED_SLASH_TO_UNDERSCORE = "slash_to_underscore"
QUERY_TRANSFORMED_GOSLIN_CANONICAL = "goslin_species_canonical"

# Ordered keys of the additive ``lipid_resolution`` object, reused for the API model and the dataset's
# ``lipid_``-prefixed flat columns so the two surfaces cannot drift apart.
LIPID_RESOLUTION_FIELDS: tuple[str, ...] = (
    "query_lipid_level_asserted",
    "query_lipid_level_effective",
    "matched_lipid_level",
    "mapping_relation",
    "mapping_predicate",
    "query_transformed",
    "ambiguous",
    "candidate_structure_count",
    "ambiguity_basis",
    "goslin_dialect",
    "goslin_formula",
    "goslin_mass",
)


def lipid_mapping_relation(matched_level: str | None, effective_level: str | None) -> tuple[str, str | None]:
    """The (relation, SKOS predicate) for a committed lipid node's matched level vs the effective level.

    Compared by specificity RANK, not name equality, so the relation survives any level whose name the
    two sources spell differently. ``unknown`` (predicate None) when either level is missing.
    """
    if matched_level is None or effective_level is None:
        return LIPID_RELATION_UNKNOWN, None
    rank_matched = _lipid_level_rank(matched_level)
    rank_effective = _lipid_level_rank(effective_level)
    if rank_matched == rank_effective:
        relation = LIPID_RELATION_EXACT
    elif rank_matched > rank_effective:
        # Higher rank == LESS specific == the committed node is broader than the query asked for.
        relation = LIPID_RELATION_BROAD
    else:
        relation = LIPID_RELATION_NARROW
    return relation, _LIPID_RELATION_PREDICATE.get(relation)


def _lipid_query_transformed(asserted_level: str | None, effective_level: str | None) -> str:
    """``slash_to_underscore`` when the trust-off policy downgraded the asserted level to a coarser
    effective one; ``goslin_species_canonical`` otherwise (no downgrade, or a level is unknown)."""
    if asserted_level is not None and effective_level is not None and asserted_level != effective_level:
        return QUERY_TRANSFORMED_SLASH_TO_UNDERSCORE
    return QUERY_TRANSFORMED_GOSLIN_CANONICAL


def _goslin_base_metadata(entity: "pd.Series | dict[str, Any]") -> dict[str, Any] | None:
    """Any goslin-lipid vote's shared metadata (dialect / formula / mass / asserted / effective).

    Every goslin vote is stamped with the same base metadata, so the first non-empty one is
    representative. ``None`` for a non-lipid row (no goslin votes), which is the signal the whole
    ``lipid_resolution`` object stays null off the lipid path.
    """
    assigned_ids = entity.get("assigned_ids") or {}
    goslin_meta = assigned_ids.get(GOSLIN_LIPID_ANNOTATOR) or {}
    for vocab_map in goslin_meta.values():
        if not isinstance(vocab_map, dict):
            continue
        for meta in vocab_map.values():
            if isinstance(meta, dict) and meta:
                return meta
    return None


class _UseEntityChosen:
    """Sentinel: ``build_lipid_resolution`` reads ``chosen_kg_id`` from the entity itself.

    A distinct type (not ``None``) because ``None`` is a real committed value -- "no node" -- that a
    caller may pass to describe an unmapped row.
    """


_USE_ENTITY_CHOSEN = _UseEntityChosen()


def build_lipid_resolution(
    entity: "pd.Series | dict[str, Any]",
    chosen_kg_id: "str | None | _UseEntityChosen" = _USE_ENTITY_CHOSEN,
) -> dict[str, Any] | None:
    """Assemble the additive ``lipid_resolution`` object for a mapped row, or ``None`` for a non-lipid.

    Pulls together what the earlier units already produced: the goslin metadata on the goslin-lipid
    votes (Unit 2/3) and the committed node's matched level joined via ``_lipid_level_context`` (the
    same raw-id vs curie reconciliation Unit 4's tie-break uses). The relation between that matched
    level and the effective query level is computed here and SUBSUMES the ``chosen_kg_id_lipid_hint``
    flag: ``mapping_relation == "broad"`` is exactly the ``lipid_generalized`` case.

    ``chosen_kg_id`` defaults to the value on ``entity``; a caller may pass it explicitly to describe a
    DIFFERENT committed node than the one on the row. Re-resolution relies on this so the object always
    tracks the node the certificate actually commits, never a node a later swap replaced.

    ``ambiguous`` / ``candidate_structure_count`` / ``ambiguity_basis`` describe the LIPID MAPS / Tier B
    candidate set (plan Units for lipids). That ambiguity signal does not reach this row yet, so they
    default to a non-ambiguous, count-unknown reading rather than fabricating one.
    """
    meta = _goslin_base_metadata(entity)
    if meta is None:
        return None
    levels, effective_from_context = _lipid_level_context(entity)
    node = entity.get("chosen_kg_id") if isinstance(chosen_kg_id, _UseEntityChosen) else chosen_kg_id
    matched_level = levels.get(node) if isinstance(node, str) else None
    asserted_level = meta.get("query_lipid_level_asserted")
    effective_level = meta.get("query_lipid_level_effective") or effective_from_context
    relation, predicate = lipid_mapping_relation(matched_level, effective_level)
    return {
        "query_lipid_level_asserted": asserted_level,
        "query_lipid_level_effective": effective_level,
        "matched_lipid_level": matched_level,
        "mapping_relation": relation,
        "mapping_predicate": predicate,
        "query_transformed": _lipid_query_transformed(asserted_level, effective_level),
        "ambiguous": False,
        "candidate_structure_count": None,
        "ambiguity_basis": None,
        "goslin_dialect": meta.get("goslin_dialect"),
        "goslin_formula": meta.get("goslin_formula"),
        "goslin_mass": meta.get("goslin_mass"),
    }


def _lipid_head_group(species_name: str | None) -> str | None:
    """The lipid class (head group) from a species-level shorthand: the token before the first space
    (e.g. ``PC 34:1`` -> ``PC``). None for an empty name. A fallback used only when a parse exposes no
    explicit CLASS-level rendering."""
    if not species_name or not str(species_name).strip():
        return None
    return str(species_name).strip().split(" ", 1)[0] or None


def build_lipid_structure_evidence(
    query_meta: dict[str, Any] | None,
    node_parse: Any | None,
) -> LipidStructureEvidence | None:
    """The STRUCTURE-FREE lipid verdict for a committed node, from the query's goslin metadata and the
    committed node's parsed name (a :class:`LipidParse`). Both are produced UPSTREAM; this only reads
    fields and delegates the comparison to the pure ``compare_lipid_composition``.

    ``None`` when the node name did not parse as a lipid (no structure-free comparison is possible) or
    the query is not a lipid, which the certificate reads as ``out_of_scope`` rather than a verdict.
    """
    if node_parse is None or not query_meta:
        return None
    query_species = query_meta.get("goslin_canonical")
    query_level_names = query_meta.get("goslin_level_names") or {}
    query_class = query_level_names.get("CLASS") or _lipid_head_group(query_species)
    node_level_names = getattr(node_parse, "level_names", {}) or {}
    node_species = node_level_names.get("SPECIES") or getattr(node_parse, "canonical_name", None)
    node_class = node_level_names.get("CLASS") or _lipid_head_group(node_species)
    return compare_lipid_composition(
        query_class=query_class,
        query_species=query_species,
        node_class=node_class,
        node_species=node_species,
    )


def lipid_flat_columns(lipid_resolution: dict[str, Any] | None) -> dict[str, Any]:
    """The ``lipid_resolution`` object as ``lipid_``-prefixed flat scalar columns for the dataset TSV.

    Every key is always emitted so the column set is stable across rows; a non-lipid row (``None``)
    gets ``None`` in every ``lipid_`` column.
    """
    if lipid_resolution is None:
        return {f"lipid_{field}": None for field in LIPID_RESOLUTION_FIELDS}
    return {f"lipid_{field}": lipid_resolution.get(field) for field in LIPID_RESOLUTION_FIELDS}


def _curie_sort_key(curie: str) -> tuple[int, int, str]:
    """Total order over CURIEs preferring a lower NUMERIC local id, without implying canonicality.

    String sort is not numeric (``CHEBI:983`` > ``CHEBI:12777`` lexically); this orders numeric locals by
    value and falls back to the full CURIE for non-numeric locals. Determinism only — the winner on a
    genuine tie is arbitrary and is logged as a WARNING by the caller.
    """
    local = curie.rsplit(":", 1)[-1]
    if local.isdigit():
        return (0, int(local), curie)
    return (1, 0, curie)


class Resolver:
    """Resolves one-to-many KG mappings to single chosen nodes."""

    def __init__(self, linker: Any = None, biolink_client: Any = None, lipid_resolver: Any = None) -> None:
        """
        Args:
            linker: Linker used by the connectivity test (get-nodes InChIKey enrichment). When
                None, source-weighting is inactive and resolution is the plain majority vote.
            biolink_client: Used to test small-molecule category membership.
            lipid_resolver: Shared LipidStructureResolver so a lipid CANDIDATE node resolves to a
                structural key during re-resolution (KTD6). When None, the candidate side has no
                lipid hop. The same instance also feeds the query-side Tier B lookup.
        """
        self.linker = linker
        self.biolink_client = biolink_client
        self.structure_resolver = (
            StructureResolver(linker, lipid_resolver=lipid_resolver) if linker is not None else None
        )

    def resolve(
        self, item: pd.Series | dict[str, Any] | pd.DataFrame, category: str | None = None
    ) -> pd.Series | pd.DataFrame:
        """
        Resolve one-to-many KG mappings to single chosen node.

        Args:
            item: Entity or entities with kg_ids fields
            category: Standardized Biolink category (e.g. 'biolink:SmallMolecule'); enables
                source-weighting for small-molecule ChEBI conflicts. None falls through to voting.

        Returns:
            Named Series for single entity or DataFrame for multiple entities, containing fields:
            chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned, chosen_kg_id_review
        """
        logging.debug("Beginning one-to-many resolution step..")

        if isinstance(item, pd.DataFrame):
            return item.apply(lambda row: self._resolve_entity(row, category=category), axis=1, result_type="expand")
        else:
            return self._resolve_entity(item, category=category)

    def _resolve_entity(self, entity: pd.Series | dict[str, Any], category: str | None = None) -> pd.Series:
        """
        Resolve one-to-many KG mappings for a single entity.

        Args:
            entity: Entity with kg_ids fields
            category: Standardized Biolink category (threaded from the mapper)

        Returns:
            Named Series with fields: chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned,
            chosen_kg_id_review
        """
        # Lipid level context for the main tie-break (Unit 4). Empty/None off the lipid path, so the
        # provided- and assigned-combined resolutions below deliberately stay level-blind (today's
        # behavior); only the primary ``kg_ids`` choice is made level-aware.
        lipid_levels, effective_level = _lipid_level_context(entity)

        chosen_kg_id_provided, _ = self._choose_best_kg_id(entity["kg_ids_provided"])
        chosen_kg_id, chosen_kg_id_review = self._choose_best_kg_id(
            entity["kg_ids"],
            kg_ids_assigned=entity["kg_ids_assigned"],
            category=category,
            lipid_levels=lipid_levels,
            effective_level=effective_level,
        )

        # Combine all annotators' KG IDs dict into one to choose preferred 'assigned' KG ID
        kg_ids_assigned_combined = defaultdict(list)
        for annotator_kg_ids_assigned in entity["kg_ids_assigned"].values():
            for kg_id, curies in annotator_kg_ids_assigned.items():
                kg_ids_assigned_combined[kg_id].extend(curies)
        chosen_kg_id_assigned, _ = self._choose_best_kg_id(kg_ids_assigned_combined)

        return pd.Series(
            {
                "chosen_kg_id": chosen_kg_id,
                "chosen_kg_id_provided": chosen_kg_id_provided,
                "chosen_kg_id_assigned": chosen_kg_id_assigned,
                "chosen_kg_id_review": chosen_kg_id_review,
                # New, additive channel (separate from the closed selection_conflict whitelist).
                "chosen_kg_id_lipid_hint": _lipid_generalized_hint(chosen_kg_id, lipid_levels, effective_level),
            }
        )

    def _stable_majority(
        self,
        kg_ids_dict: dict[str, list[str]],
        category: str | None,
        lipid_levels: dict[str, str] | None = None,
        effective_level: str | None = None,
    ) -> str:
        """Majority vote by supporting-curie count with a DETERMINISTIC tie-break.

        A strict count winner is returned unchanged (byte-identical to the old
        ``max(kg_ids_dict, key=len)``). On a genuine 2+ count tie the pick is (1) for a lipid tie whose
        candidates carry known levels, the node whose matched level EQUALS the effective query level
        (Unit 4), then (2) a category-preferred namespace when one is configured, then (3)
        ``_curie_sort_key`` as a total-order fallback, and a WARNING is logged so a coin-flip resolution
        stays visible in run logs instead of riding silently on API-response order. Determinism, not
        chemical correctness.

        ``lipid_levels`` / ``effective_level`` are supplied ONLY for the primary lipid-capable
        resolution; when absent (every non-lipid tie, and the provided/assigned-combined resolutions)
        the pool stays the full tie set and the pick is byte-identical to the pre-Unit-4 behavior.
        """
        max_count = max(len(curies) for curies in kg_ids_dict.values())
        tied = [kg_id for kg_id, curies in kg_ids_dict.items() if len(curies) == max_count]
        if len(tied) == 1:
            return tied[0]
        # Unit 4: for a lipid tie, prefer the node whose matched level equals the EFFECTIVE query level
        # before today's namespace/numeric tie-break. Only narrows the pool when at least one tied node
        # matches exactly; otherwise (and for every non-lipid tie, where lipid_levels is empty) the pool
        # is the full tie set and the downstream pick is unchanged.
        pool_for_pref = tied
        if lipid_levels and effective_level:
            exact = [kg_id for kg_id in tied if lipid_levels.get(kg_id) == effective_level]
            if exact:
                pool_for_pref = exact
        # Prefer a canonical namespace for the row's category when configured, inheriting a configured
        # ancestor's policy through the Biolink hierarchy (so e.g. biolink:Drug inherits SmallMolecule's
        # CHEBI/HMDB/RM), matching the annotation path. The numeric fallback keeps ties deterministic when
        # no policy applies.
        preferred = self._preferred_prefixes(category)
        pool = [kg_id for kg_id in pool_for_pref if kg_id.split(":", 1)[0] in preferred] if preferred else []
        chosen = min(pool or pool_for_pref, key=_curie_sort_key)
        logging.warning(
            "majority tie among %s (category=%s); picked %s by deterministic tie-break — arbitrary, no "
            "chemical preference",
            sorted(tied),
            category,
            chosen,
        )
        return chosen

    def _choose_best_kg_id(
        self,
        kg_ids_dict: dict[str, list[str]],
        kg_ids_assigned: dict[str, dict[str, list[str]]] | None = None,
        category: str | None = None,
        lipid_levels: dict[str, str] | None = None,
        effective_level: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Select a single KG ID from multiple candidates.

        The default is a majority vote by count of supporting curies. For small-molecule ChEBI
        conflicts (RefMet annotator disagreeing with the majority) the choice is source-weighted
        toward RefMet under a three-way InChIKey-connectivity rule:
        - same connectivity  -> RefMet, no flag (same molecule, no accuracy loss)
        - different connectivity -> RefMet, flag 'divergent_refmet' (error-prone bucket)
        - InChIKey unavailable -> majority, flag 'conflict_no_structure'
        Non-metabolite, no-RefMet-vote, and no-conflict cases fall through to today's behavior.

        Args:
            kg_ids_dict: Dictionary mapping KG IDs to supporting curies
            kg_ids_assigned: Per-annotator {kg_id: [curies]}, used to find the RefMet node
            category: Standardized Biolink category; source-weighting applies only to small molecules
            lipid_levels: Optional {kg_id: matched_level} for lipid candidates; enables the Unit 4
                level-aware tie-break. Omitted (empty) off the lipid path, leaving the tie-break unchanged.
            effective_level: The effective lipid query level the tie-break prefers a candidate to match.

        Returns:
            (chosen_kg_id, review_flag) — review_flag is None unless the choice should be
            surfaced for human review; (None, None) if there are no candidates.
        """
        if not kg_ids_dict:
            return None, None

        # Deterministic majority vote (stable tie-break on count ties; warning logged on a genuine tie).
        # Lipid level context, when supplied, makes the tie-break prefer the effective-query-level node.
        majority = self._stable_majority(kg_ids_dict, category, lipid_levels, effective_level)

        # Source-weighting applies ONLY to small-molecule ChEBI conflicts.
        if not (kg_ids_assigned and category and self._is_small_molecule(category)):
            return majority, None

        # Deterministic pick. RefMet contributing >1 node is itself a signal, so the choice must not
        # ride on dict insertion order (which follows API response order). This is a provable no-op on
        # today's data -- no baseline row has a multi-node RefMet vote (artifact field
        # refmet_multi_node_rate, regenerated by studies/analysis/off_category_audit.py). Note
        # this is a *determinism* fix, not a correctness one — lexicographic order is still chemically
        # arbitrary, so warn to surface the case if it ever appears and needs a real tiebreak rule.
        refmet_nodes = sorted(kg_ids_assigned.get(REFMET_ANNOTATOR, {}))
        if len(refmet_nodes) > 1:
            logging.warning(
                "RefMet contributed %d KG nodes (%s); picking %s lexicographically — "
                "no chemical tiebreak rule exists for this case",
                len(refmet_nodes),
                refmet_nodes,
                refmet_nodes[0],
            )
        # Agreement is membership, not first-element equality. When RefMet votes for several nodes and
        # one of them IS the majority, RefMet agrees — testing only refmet_nodes[0] would miss that,
        # override the majority with a different RefMet node, and emit a spurious 'divergent_refmet'.
        # Unreachable on current data (see refmet_multi_node_rate), but the deterministic sort above
        # exists precisely to make that case well-defined, so it must be correct.
        if not refmet_nodes or majority in refmet_nodes:
            return majority, None  # RefMet had no say, or already agrees

        refmet_node = refmet_nodes[0]
        same = self._connectivity_match(refmet_node, majority)
        if same is True:
            return refmet_node, None  # same molecule -> RefMet, silent
        if same is False:
            return refmet_node, "divergent_refmet"  # different molecule -> RefMet, FLAG
        return majority, "conflict_no_structure"  # no InChIKey -> majority, FLAG

    def is_small_molecule(self, category: str | None) -> bool:
        """Public form of the small-molecule subtree test, for callers outside resolution.

        The resolution certificate needs it to tell a metabolite row (where "the graph asserts no
        structure" is a meaningful refusal) from a gene row (where it is a category error). Kept as
        a thin delegate so both answers come from one definition.
        """
        return bool(category) and self._is_small_molecule(str(category))

    def _is_small_molecule(self, category: str) -> bool:
        """True when the category is ``biolink:SmallMolecule`` or a descendant.

        Mirrors ``AnnotationEngine._is_human_applicable_category`` / ``_category_preferred_prefixes``
        (subtree membership) — the same population that carries the ChEBI/RM canonical-namespace policy.
        """
        if self.biolink_client is None:
            return False
        return category in self.biolink_client.get_descendants("biolink:SmallMolecule")

    def _preferred_prefixes(self, category: str | None) -> set[str]:
        """Preferred namespaces for a category, inheriting configured Biolink ancestors' policy.

        Mirrors ``AnnotationEngine._category_preferred_prefixes``: each configured key in
        ``CATEGORY_PREFERRED_NAMESPACES`` applies to all its Biolink descendants, so a descendant category
        (e.g. ``biolink:Drug``) inherits ``biolink:SmallMolecule``'s prefixes. Returns the union on overlap;
        empty set when no policy applies or ``biolink_client`` is unavailable. Called only on a genuine tie.
        """
        if not category or self.biolink_client is None:
            return set()
        preferred: set[str] = set()
        for configured, prefixes in CATEGORY_PREFERRED_NAMESPACES.items():
            if category in self.biolink_client.get_descendants(configured):
                preferred |= prefixes
        return preferred

    def _connectivity_match(self, node_a: str, node_b: str) -> bool | None:
        """Delegate the InChIKey-connectivity test to the StructureResolver (None if unavailable)."""
        if self.structure_resolver is None:
            return None
        return self.structure_resolver.connectivity_match(node_a, node_b)

    def reresolve_on_contradiction(
        self,
        *,
        candidates: Iterable[str],
        query_independent_inchikey: str | None,
        committed_kg_id: str | None,
    ) -> tuple[str | None, str]:
        """Pick the distinct candidate whose structure matches the query's independent structure.

        This is the structure-guided correction: when a certificate is CONTRADICTED, the conflated
        vote committed a node whose structure disagrees with the query's INDEPENDENT structure. Among
        the one-to-many candidates (the vote's losers), find the unique DISTINCT candidate whose own
        structure matches that independent anchor and switch to it.

        KTD5 — the anchor is ALWAYS ``query_independent_inchikey`` (the query's independent
        structure), never the committed node's own InChIKey. The committed node is excluded from the
        match search, so its structure is never even consulted here.

        Two topologies (design fact): across-node conflation has a distinct matching candidate to
        switch to; within-node conflation does not, and this refuses rather than fabricates (L2).

        Returns ``(chosen_kg_id, reason)``:
          * ``(candidate, "reresolved")`` — a unique distinct candidate matched.
          * ``(committed, "reresolution_ambiguous")`` — several distinct candidates matched.
          * ``(committed, "reresolution_refused_no_match")`` — none matched / no anchor / disabled deps.
          * ``(committed, "reresolution_disabled")`` — the flag is off (defensive; the Mapper gates too).
        """
        import biomapper2.config as config

        if not config.RERESOLUTION_ENABLED:
            return committed_kg_id, "reresolution_disabled"
        anchor = query_independent_inchikey
        if not anchor or self.structure_resolver is None:
            return committed_kg_id, "reresolution_refused_no_match"

        # Never the committed node: KTD5 forbids reading its key as the anchor, and there is no point
        # matching it to itself. Only the DISTINCT other candidates are considered.
        others = [c for c in dict.fromkeys(candidates) if c and c != committed_kg_id]
        if not others:
            return committed_kg_id, "reresolution_refused_no_match"

        records = self.linker.get_node_records(others) if self.linker is not None else {}
        matches: set[str] = set()
        for candidate in others:
            name = (records.get(candidate) or {}).get("name")
            # A candidate may carry several graph-asserted InChIKeys; accept a match against ANY of
            # them, so the correct candidate is not missed when its match is a non-first key.
            candidate_keys = self.structure_resolver.structural_inchikeys(candidate, name, records)
            if any(structural_agree(anchor, ck) for ck in candidate_keys):
                matches.add(candidate)

        if len(matches) == 1:
            return matches.pop(), "reresolved"
        if len(matches) > 1:
            # Two distinct candidates both match the independent structure: the KG is conflated on
            # the other side too. Refuse rather than pick arbitrarily (L2).
            return committed_kg_id, "reresolution_ambiguous"
        return committed_kg_id, "reresolution_refused_no_match"
