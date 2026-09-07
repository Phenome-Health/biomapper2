"""The resolution certificate: what the graph asserts about a committed node, and what refutes it.

Why this module exists
----------------------
Before it, the resolver computed a structural verdict inside ``Resolver._choose_best_kg_id`` and
discarded it. The only thing that escaped was ``chosen_kg_id_review``, a flag string whose ``None``
covered several distinct situations, so a consumer could not tell "the graph asserts a structure for
this answer" from "the graph asserts nothing and we cannot check". The certificate makes that
distinction a first-class, emitted object.

The one rule this module must never get wrong (L21)
---------------------------------------------------
``structure_absent`` is **unverifiable**, not wrong. A committed node the graph lists no InChIKey for
cannot be ``contradicted``, because there is nothing to contradict it with; it is ``unavailable``.
Publishing a precision gain for refusing that bucket would be a claim no oracle can support -- see
the artifact field ``sparsity_control.n_absent_oracle_could_fire`` in
``studies/analysis/certificate_state_audit.py``, which is the admissibility test for any such claim.
``issue`` enforces the rule structurally (``contradicted`` is reachable only from
``structure_present``) and ``tests/test_resolution_certificate.py`` asserts it over the full input
cross-product.

Two tiers
---------
**Tier A** is the default and is **zero-I/O**. It reads ``kg_equivalent_ids["INCHIKEY"]`` -- the
structure the *graph* asserts for the node the pipeline already committed -- and nothing else. It
deliberately does NOT use ``StructureResolver.inchikey_blocks``: that helper falls through to
Metabolomics Workbench and PubChem *by name* when the KG lists no key, i.e. on exactly the
``structure_absent`` population, which would fire an external request per absent row and silently
reclassify some of them as ``structure_present`` from a non-KG source. The two look interchangeable
in a diff and are not: ``inchikey_blocks`` answers "what structure can I find for this node by any
means", Tier A answers "what structure does the GRAPH assert for this node". Only the second is a
self-certificate, and only the second is free.

**Tier B** is opt-in and default-off. It resolves the *query name* against an independent registry
and can refine ``structure_present`` into ``corroborated`` / ``contradicted``. It never moves a row
off ``unavailable`` or ``not_applicable`` -- so it is neither consulted on those rows (the producer
side, in ``Mapper._issue_certificate``) nor reported on them (``issue`` below), which keeps the
rate-limited lookups, the resolution-rate denominator, and the emitted evidence fields all scoped to
the one population a structural comparison happens in.

``issue`` is pure and I/O-free on purpose, so the state table can be tested without a network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# The annotator whose vote the resolver source-weights toward. Named here as well as in
# ``resolver`` because Tier B's independence claim is defined against it (L26).
REFMET_ANNOTATOR = "metabolomics-workbench"

# Tier B sources, and the annotator each one is NOT independent of. PubChem is independent of every
# annotator the resolver source-weights toward; MW is the RefMet registry itself. LIPID MAPS binds a
# Goslin-canonical lipid shorthand to a structure; it is KG-independent (valid for the cross-cohort
# certificate) but CIRCULAR against the LMSD benchmark arm, so a lipidmaps-sourced structure carries
# this tag and the benchmark axis excludes those rows from its LMSD lipid oracle.
TIER_B_SOURCE_MW = "metabolomics-workbench"
TIER_B_SOURCE_PUBCHEM = "pubchem"
TIER_B_SOURCE_LIPIDMAPS = "lipidmaps"
_SOURCE_TO_DEPENDENT_ANNOTATOR = {TIER_B_SOURCE_MW: REFMET_ANNOTATOR}

# Identifier of the rule that produced the verdict (L20). Seeded to the semantics PR #47 shipped:
# ``connectivity_match`` intersects the SET of InChIKey first blocks. D3's tightening, when it
# lands, introduces a second value here rather than silently changing what this one means.
COMPARISON_RULE_FIRST_BLOCK_SET_INTERSECTION = "inchikey_first_block_set_intersection/v1"

INCHIKEY_PREFIX = "INCHIKEY"

# Length of the InChIKey stereo layer prefix folded into the structural key. The second block
# encodes stereochemistry, isotopes, protonation and charge in a fixed layout; its first 8
# characters are the stereo/skeleton-refinement layer, and comparing only those keeps the key robust
# to the protonation/charge suffix while still separating stereoisomers. This is the same
# ``block1 + block2[:8]`` seam the benchmark scorers use (studies/.../necs_gold_repair.py).
STRUCTURAL_KEY_BLOCK2_LEN = 8


def structural_key_parts(inchikey: str | None) -> tuple[str | None, str | None]:
    """Split an InChIKey (full OR first-block-only) into ``(block1, block2[:8] | None)``.

    ``block1`` is the connectivity skeleton; ``block2`` (truncated) is the stereo layer, or ``None``
    when the input carries only a first block (MW/PubChem emit first-block only). Upper-cased so an
    external service's casing never decides a verdict.
    """
    if not inchikey or not str(inchikey).strip():
        return None, None
    parts = str(inchikey).strip().upper().split("-")
    block1 = parts[0] or None
    block2 = parts[1][:STRUCTURAL_KEY_BLOCK2_LEN] if len(parts) > 1 and parts[1] else None
    return block1, block2


def structural_agree(a: str | None, b: str | None) -> bool:
    """True when two InChIKeys agree at the structural key (block1, then block2[:8] if BOTH carry it).

    Connectivity is compared on block1. Stereo is compared on block2[:8] ONLY when both operands
    carry a second block; if either side is first-block-only the agreement is connectivity-only and
    stereo is deliberately NOT asserted (never a silent stereo pass). A missing/empty key never
    agrees.
    """
    a1, a2 = structural_key_parts(a)
    b1, b2 = structural_key_parts(b)
    if a1 is None or b1 is None or a1 != b1:
        return False
    if a2 is not None and b2 is not None:
        return a2 == b2
    return True


# Cache provenance. The confound that motivated recording this (a cold cache returning a wrong
# node) lives in the KESTREL store, which expires; the structure store does not expire at all. Both
# are recorded because a certificate read months later cannot otherwise tell which one it depended
# on.
KESTREL_CACHE_STORE = "kestrel_http"
KESTREL_CACHE_EXPIRY = "1h"
STRUCTURE_CACHE_STORE = "structure_http"
STRUCTURE_CACHE_EXPIRY = "never"

# The legacy review-flag values the resolver actually returns. The field is tri-valued: these two
# plus None. Exported so the derivation test enumerates the real domain instead of a hand-copied one.
SELECTION_CONFLICT_VALUES = ("divergent_refmet", "conflict_no_structure", None)


class CertificateState(str, Enum):
    """What independent evidence says about the committed node."""

    CORROBORATED = "corroborated"
    UNCORROBORATED = "uncorroborated"
    CONTRADICTED = "contradicted"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class StructureStatus(str, Enum):
    """What the GRAPH asserts about the committed node's structure. Never a network answer."""

    STRUCTURE_PRESENT = "structure_present"
    STRUCTURE_ABSENT = "structure_absent"
    NOT_APPLICABLE = "not_applicable"


class TierBOutcome(str, Enum):
    """What happened when (and whether) an independent source was consulted.

    ``LOOKUP_FAILED`` is kept distinct from ``UNRESOLVABLE`` deliberately: a rate-limited or
    unreachable service is a property of the network, and collapsing it into "the name has no known
    structure" would turn an operating curve into an artifact of the run.
    """

    OFF = "off"
    RESOLVED = "resolved"
    UNRESOLVABLE = "unresolvable"
    LOOKUP_FAILED = "lookup_failed"


@dataclass(frozen=True)
class TierBResult:
    """One independent structure lookup for a QUERY NAME. Produced by the Tier B module, never here."""

    source: str | None
    inchikey_block: str | None
    outcome: TierBOutcome
    cache_state: str | None = None  # 'hit' | 'miss' | 'process_memo' | None


@dataclass(frozen=True)
class ResolutionCertificate:
    """The emitted certificate. Frozen: a consumer must not be able to edit a verdict in place."""

    state: CertificateState
    structure_status: StructureStatus
    node_inchikey_blocks: list[str]
    comparison_rule: str
    equivalent_ids_lookup_ok: bool
    selection_conflict: str | None = None
    independent_source: str | None = None
    independent_inchikey_block: str | None = None
    independent_of_selection: bool | None = None
    tier_b_outcome: TierBOutcome = TierBOutcome.OFF
    # Reserved so the schema does not change shape when the refusal-reason follow-up lands (L28).
    # Until it does, a consumer cannot distinguish an off-category refusal from a no-match, and
    # refusal must not be described as observable in the released artifact (L30).
    refusal_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        """Nested, plain-JSON representation for the API surface.

        Plain types are not a nicety here: pydantic rejects a raw dataclass, and the streaming
        endpoint builds an unvalidated dict that is ``json.dumps``'d outside its try/except, so a
        dataclass there raises mid-stream after a 200 has already been sent.
        """
        return {
            "state": self.state.value,
            "structure_status": self.structure_status.value,
            "node_inchikey_blocks": list(self.node_inchikey_blocks),
            "comparison_rule": self.comparison_rule,
            "equivalent_ids_lookup_ok": self.equivalent_ids_lookup_ok,
            "selection_conflict": self.selection_conflict,
            "independent_source": self.independent_source,
            "independent_inchikey_block": self.independent_inchikey_block,
            "independent_of_selection": self.independent_of_selection,
            "tier_b_outcome": self.tier_b_outcome.value,
            "refusal_reason": self.refusal_reason,
            "provenance": dict(self.provenance),
        }

    def to_flat_columns(self) -> dict[str, Any]:
        """Flat scalar columns for the mapped TSV.

        Scalars only, and the ``certificate_`` prefix is load-bearing: ``chosen_kg_id_provided`` and
        ``chosen_kg_id_assigned`` are separately emitted columns that never receive a category and
        get no certificate, so a bare ``state`` column would be read as covering them. These columns
        describe ``chosen_kg_id`` and nothing else.
        """
        flat: dict[str, Any] = {
            "certificate_state": self.state.value,
            "certificate_structure_status": self.structure_status.value,
            "certificate_node_inchikey_blocks": "|".join(self.node_inchikey_blocks),
            "certificate_comparison_rule": self.comparison_rule,
            "certificate_equivalent_ids_lookup_ok": self.equivalent_ids_lookup_ok,
            "certificate_selection_conflict": self.selection_conflict,
            "certificate_independent_source": self.independent_source,
            "certificate_independent_inchikey_block": self.independent_inchikey_block,
            "certificate_independent_of_selection": self.independent_of_selection,
            "certificate_tier_b_outcome": self.tier_b_outcome.value,
            "certificate_refusal_reason": self.refusal_reason,
        }
        for key, value in self.provenance.items():
            flat[f"certificate_provenance_{key}"] = value
        return flat


def derive_chosen_kg_id_review(certificate: ResolutionCertificate) -> str | None:
    """The legacy ``chosen_kg_id_review`` flag, derived from the certificate (C4 / L20).

    Identical to today's value for one release; the deprecation is filed as a follow-up. The
    derivation rests on an invariant ``issue`` enforces: a non-None ``selection_conflict`` always
    co-occurs with a committed node and a small-molecule category.
    """
    return certificate.selection_conflict


def node_blocks_from_equivalent_ids(kg_equivalent_ids: Mapping[str, Any] | None) -> list[str]:
    """Sorted InChIKey first blocks the GRAPH asserts for the committed node. Zero I/O.

    Reads ``kg_equivalent_ids[INCHIKEY]`` only. Do not "simplify" this to
    ``StructureResolver.inchikey_blocks``: that helper reaches MW/PubChem by name when the KG lists
    no key, which is the entire ``structure_absent`` population, and would shift the state
    distribution with no test going red.
    ``tests/test_certificate_emission.py::test_tier_a_makes_no_structure_lookup`` (G6) asserts on
    the StructureResolver fetchers specifically for that reason.
    """
    if not kg_equivalent_ids:
        return []
    keys = kg_equivalent_ids.get(INCHIKEY_PREFIX) or []
    if isinstance(keys, str):
        keys = [keys]
    # Upper-cased at the producer so the EMITTED column is canonical, not just the comparison.
    # InChIKey blocks are conventionally upper-case but nothing upstream enforces it: these come
    # from the KG, and Tier B's come from two external services.
    blocks = {k.split("-")[0].upper() for k in keys if isinstance(k, str) and k.strip()}
    return sorted(blocks)


def node_full_inchikeys_from_equivalent_ids(kg_equivalent_ids: Mapping[str, Any] | None) -> list[str]:
    """FULL InChIKeys the GRAPH asserts for the committed node, upper-cased. Zero I/O.

    Companion to ``node_blocks_from_equivalent_ids`` (which strips to first blocks for the emitted
    column). The full keys are kept only for the structural-key comparison, so a stereo-layer
    difference against a full-key Tier B result is detectable instead of being flattened away.
    """
    if not kg_equivalent_ids:
        return []
    keys = kg_equivalent_ids.get(INCHIKEY_PREFIX) or []
    if isinstance(keys, str):
        keys = [keys]
    return sorted({k.strip().upper() for k in keys if isinstance(k, str) and k.strip()})


def _default_provenance(tier_b: TierBResult | None) -> dict[str, Any]:
    return {
        "tier_b_enabled": tier_b is not None and tier_b.outcome is not TierBOutcome.OFF,
        "tier_b_cache_state": tier_b.cache_state if tier_b else None,
        "kestrel_cache_store": KESTREL_CACHE_STORE,
        "kestrel_cache_expiry": KESTREL_CACHE_EXPIRY,
        "structure_cache_store": STRUCTURE_CACHE_STORE,
        "structure_cache_expiry": STRUCTURE_CACHE_EXPIRY,
    }


def _independent_of_selection(tier_b: TierBResult, committed_node_sources: Iterable[str] | None) -> bool | None:
    """L26. False when the Tier B source IS the registry that supplied the committed node.

    Tier B via MW is not independent of the selector on the rows where it matters most: the resolver
    source-weights toward the RefMet annotator, which queries MW's fuzzy ``/refmet/match`` with the
    query name to produce the candidate that then wins, and Tier B's first hop is MW's
    ``/refmet/name`` -- the same registry, keyed on the same query name. Asking RefMet whether
    RefMet was right is circular, so independence is claimed only on the subset where it holds.
    """
    if tier_b.outcome is not TierBOutcome.RESOLVED or not tier_b.source:
        return None
    dependent_annotator = _SOURCE_TO_DEPENDENT_ANNOTATOR.get(tier_b.source)
    if dependent_annotator is None:
        return True
    return dependent_annotator not in set(committed_node_sources or ())


def issue(
    *,
    chosen_kg_id: str | None,
    is_small_molecule: bool,
    kg_equivalent_ids: Mapping[str, Any] | None,
    equivalent_ids_lookup_ok: bool,
    selection_conflict: str | None = None,
    tier_b: TierBResult | None = None,
    committed_node_sources: Iterable[str] | None = None,
    comparison_rule: str = COMPARISON_RULE_FIRST_BLOCK_SET_INTERSECTION,
    refusal_reason: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ResolutionCertificate:
    """Issue a certificate for one committed answer. Pure: no network, no cache, no clock.

    Args:
        chosen_kg_id: The committed node, or None when the pipeline committed nothing.
        is_small_molecule: Whether the row's category is in the small-molecule subtree. Rows outside
            it get ``not_applicable`` rather than ``unavailable`` -- see below.
        kg_equivalent_ids: The committed node's equivalent ids, as the pipeline already emits them.
        equivalent_ids_lookup_ok: False when the /get-nodes enrichment call failed. A failed lookup
            must not be read as "the graph asserts no structure".
        selection_conflict: The resolver's intra-KG review flag, on a different axis from ``state``.
        tier_b: An independent lookup for the query name, when Tier B is enabled.
        committed_node_sources: Annotator slugs that supplied the committed node (L26).
    """
    if selection_conflict is not None and (chosen_kg_id is None or not is_small_molecule):
        # The resolver only reaches the flagging branch inside the small-molecule guard and only
        # after a candidate exists, so this combination means a caller assembled the certificate
        # from mismatched pieces. Failing loudly protects the legacy derivation.
        raise ValueError(
            "selection_conflict is set on a row with no committed node or a non-small-molecule "
            "category; the resolver cannot produce that combination"
        )

    blocks = node_blocks_from_equivalent_ids(kg_equivalent_ids)
    tier_b_outcome = tier_b.outcome if tier_b else TierBOutcome.OFF

    if not is_small_molecule:
        # Not defensive padding. ``unavailable`` means "we looked for a structure and the graph has
        # none" -- a meaningful statement about a metabolite and a meaningless one about a gene.
        # Deleting this state to simplify the enum reintroduces L21's error in a new population.
        structure_status = StructureStatus.NOT_APPLICABLE
        state = CertificateState.NOT_APPLICABLE
    elif chosen_kg_id is None:
        # Nothing was committed, so there is no answer to certify. Not ``structure_absent``: that
        # would assert something about a node that does not exist.
        structure_status = StructureStatus.NOT_APPLICABLE
        state = CertificateState.UNAVAILABLE
    elif not equivalent_ids_lookup_ok:
        # A transient /get-nodes failure returns {} and only logs, so without this branch an outage
        # would silently mark an entire run ``structure_absent`` and an offline rerun on the
        # resulting TSV could never detect it. Unknown is not absent.
        structure_status = StructureStatus.NOT_APPLICABLE
        state = CertificateState.UNAVAILABLE
    elif not blocks:
        structure_status = StructureStatus.STRUCTURE_ABSENT
        state = CertificateState.UNAVAILABLE
    else:
        structure_status = StructureStatus.STRUCTURE_PRESENT
        state = CertificateState.UNCORROBORATED
        if tier_b is not None and tier_b.outcome is TierBOutcome.RESOLVED and tier_b.inchikey_block:
            # Structural-key comparison (block1 + block2[:8]), folded over BOTH operands so a full-key
            # Tier B result (the lipid source emits one) is not tested for membership in a set of
            # first blocks -- which would never match and would flip every RESOLVED row to
            # CONTRADICTED. The node side keeps its FULL keys here (the emitted column stays first
            # blocks) so a stereoisomer difference is detectable; MW/PubChem still emit first-block
            # only, degrading their rows to a connectivity-only agreement, never a silent stereo
            # pass. This generalizes the old first-block set-intersection: when every operand is
            # first-block-only the verdict is identical, so default (Tier-A) output is unchanged.
            node_full_keys = node_full_inchikeys_from_equivalent_ids(kg_equivalent_ids)
            agrees = any(structural_agree(tier_b.inchikey_block, nk) for nk in node_full_keys)
            state = CertificateState.CORROBORATED if agrees else CertificateState.CONTRADICTED

    # Independent-evidence fields belong ONLY to rows the evidence was actually weighed on -- i.e.
    # ``structure_present``. A row that is out of scope, or that committed no node, has nothing for
    # this evidence to be about: attaching a structural block to a gene declares the entity outside
    # the population and then describes its structure in the same breath, and
    # ``_independent_of_selection`` is vacuously True with no committed node -- asserting
    # independence from a selection that never happened.
    #
    # ``structure_absent`` is excluded for the same reason and is not a lesser case (L21): the state
    # stays ``unavailable`` no matter what Tier B returns, because there is no node structure to
    # compare against. Emitting ``independent_source``/``independent_inchikey_block`` there would
    # publish an independent block beside a verdict that never used it, which reads as corroborating
    # evidence the certificate declined to act on. The producer side of the same rule lives in
    # ``Mapper._issue_certificate``, which does not spend the lookup at all; this branch is what
    # makes a direct ``issue()`` caller behave identically.
    in_population = structure_status is StructureStatus.STRUCTURE_PRESENT
    if tier_b is None or not in_population:
        independent_source = None
        independent_block = None
        independence = None
    else:
        independent_source = tier_b.source if tier_b.outcome is not TierBOutcome.OFF else None
        independent_block = tier_b.inchikey_block
        independence = _independent_of_selection(tier_b, committed_node_sources)

    certificate = ResolutionCertificate(
        state=state,
        structure_status=structure_status,
        node_inchikey_blocks=blocks,
        comparison_rule=comparison_rule,
        equivalent_ids_lookup_ok=equivalent_ids_lookup_ok,
        selection_conflict=selection_conflict,
        independent_source=independent_source,
        independent_inchikey_block=independent_block,
        independent_of_selection=independence,
        tier_b_outcome=tier_b_outcome,
        refusal_reason=refusal_reason,
        provenance=dict(provenance) if provenance is not None else _default_provenance(tier_b),
    )

    # G3, asserted at the point of construction as well as in the suite. ``contradicted`` is
    # reachable only from ``structure_present`` above; this makes a future branch that reintroduces
    # it fail here rather than in a figure.
    if certificate.state is CertificateState.CONTRADICTED and not certificate.node_inchikey_blocks:
        raise AssertionError("contradicted issued for a node the graph asserts no structure for (L21)")
    return certificate
