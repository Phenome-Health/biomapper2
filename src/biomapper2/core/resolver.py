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

from .certificate import structural_agree
from .structure_resolver import StructureResolver

# The annotator whose vote is authoritative for small-molecule ChEBI conflicts: it queries the
# RefMet /match endpoint by name and emits the RefMet-anchored node.
REFMET_ANNOTATOR = "metabolomics-workbench"


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
        chosen_kg_id_provided, _ = self._choose_best_kg_id(entity["kg_ids_provided"])
        chosen_kg_id, chosen_kg_id_review = self._choose_best_kg_id(
            entity["kg_ids"], kg_ids_assigned=entity["kg_ids_assigned"], category=category
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
            }
        )

    def _choose_best_kg_id(
        self,
        kg_ids_dict: dict[str, list[str]],
        kg_ids_assigned: dict[str, dict[str, list[str]]] | None = None,
        category: str | None = None,
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

        Returns:
            (chosen_kg_id, review_flag) — review_flag is None unless the choice should be
            surfaced for human review; (None, None) if there are no candidates.
        """
        if not kg_ids_dict:
            return None, None

        # Unchanged default: majority vote by number of supporting curies.
        majority = max(kg_ids_dict, key=lambda k: len(kg_ids_dict[k]))

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
