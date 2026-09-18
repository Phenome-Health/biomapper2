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

# The rule that produces the graded ``resolution_level``. It refines the state's agreement into
# exact/structural/connectivity using the SAME operand and block1/block2 comparison, so it never
# disagrees with the state; recorded in provenance when a level was computed.
COMPARISON_RULE_INCHIKEY_LADDER = "inchikey_ladder/v2"

# The rule that produces a STRUCTURE-FREE lipid verdict: a Goslin name-composition comparison between
# the committed node's name and the query, used when the committed node carries no graph InChIKey (the
# common case for species-level lipid nodes). It is a DIFFERENT axis from the InChIKey-block rules
# above; a verdict it produces is graded on ``lipid_resolution_level`` (parallel to ``resolution_level``)
# and never touches the InChIKey block comparison.
COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION = "goslin_level_composition/v1"

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


class ResolutionLevel(str, Enum):
    """How deeply the committed node and the independent lookup agree, refining the binary
    corroborated/contradicted state. Derived from the SAME block1/block2 comparison as
    ``structural_agree``; a block-1 disagreement is ``contradicted`` here, and the structure
    escalation (a later unit) can reclaim some of those to a same-molecule level.
    """

    EXACT_INCHIKEY = "exact_inchikey"  # full keys identical
    STRUCTURAL = "structural"  # block1 and block2[:8] agree (both operands carry a stereo layer)
    CONNECTIVITY = "connectivity"  # block1 agrees; stereo not compared (a side is first-block-only)
    CONTRADICTED = "contradicted"  # block1 disagrees, or block1 agrees but block2[:8] differs
    UNAVAILABLE = "unavailable"  # no comparison was possible


_LEVEL_RANK = {
    ResolutionLevel.EXACT_INCHIKEY: 4,
    ResolutionLevel.STRUCTURAL: 3,
    ResolutionLevel.CONNECTIVITY: 2,
    ResolutionLevel.CONTRADICTED: 1,
    ResolutionLevel.UNAVAILABLE: 0,
}


class LipidResolutionLevel(str, Enum):
    """How deeply a committed LIPID node and the query agree, on a PARALLEL axis to ``ResolutionLevel``.

    ``ResolutionLevel`` grades an InChIKey-block comparison (EXACT_INCHIKEY / STRUCTURAL / CONNECTIVITY);
    a lipid composition match is a different kind of agreement, so it gets its own axis rather than
    overloading the InChIKey enum. It is produced by the structure-free Goslin name-composition check
    for lipid nodes the graph lists no InChIKey for, and stays ``UNAVAILABLE`` on every non-lipid and
    every InChIKey-bearing row (those keep the ``ResolutionLevel`` axis).
    """

    LIPID_SPECIES = "lipid_species"  # same lipid class and same sum composition (species level)
    CONTRADICTED = "contradicted"  # different lipid class or different sum composition
    UNAVAILABLE = "unavailable"  # no structure-free comparison was possible


# The mapping relation a structure-free comparison assigns when it corroborates. ``exact`` when the
# node expresses the same species composition the query does; ``broad`` when the node is a coarser
# reading (class only) than the query's species.
LIPID_STRUCTURE_RELATION_EXACT = "exact"
LIPID_STRUCTURE_RELATION_BROAD = "broad"


@dataclass(frozen=True)
class LipidStructureEvidence:
    """A STRUCTURE-FREE lipid verdict for a committed node, pre-computed UPSTREAM (``issue`` never parses).

    Assembled by comparing the committed node's Goslin parse to the query's; the state machine consumes
    it as plain data. ``level`` grades the composition agreement on the ``LipidResolutionLevel`` axis;
    ``mapping_relation`` is ``broad`` when the node is coarser than the query, ``exact`` when it matches
    the query's species, and ``None`` when contradicted or unavailable.
    """

    level: LipidResolutionLevel
    mapping_relation: str | None = None
    comparison_rule: str = COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION


def _norm_lipid_token(value: str | None) -> str | None:
    """Case/space-normalized lipid token for comparison, or None when empty."""
    if value is None:
        return None
    text = " ".join(str(value).strip().split()).upper()
    return text or None


def compare_lipid_composition(
    *,
    query_class: str | None,
    query_species: str | None,
    node_class: str | None,
    node_species: str | None,
) -> LipidStructureEvidence:
    """Pure structure-free comparison of a committed lipid node against the query.

    Both operands are parsed UPSTREAM (this never calls Goslin). ``*_class`` is the head group (e.g.
    ``PC``); ``*_species`` is the sum-composition species name (e.g. ``PC 34:1``) or None when the parse
    is coarser than species. Same class and same sum composition corroborate at ``LIPID_SPECIES``; a
    node that is class-only (coarser than the query's species) corroborates ``broad``; a different class
    or a different composition contradicts; a missing class on either side is unavailable (not a lipid).
    """
    qc, nc = _norm_lipid_token(query_class), _norm_lipid_token(node_class)
    if qc is None or nc is None:
        return LipidStructureEvidence(level=LipidResolutionLevel.UNAVAILABLE)
    if qc != nc:
        return LipidStructureEvidence(level=LipidResolutionLevel.CONTRADICTED)
    qs, ns = _norm_lipid_token(query_species), _norm_lipid_token(node_species)
    if ns is None or qs is None:
        # Class agrees but at least one side has no species composition: the node is a coarser reading
        # than the query. Corroborate at the class level, flagged broad (a generalization).
        return LipidStructureEvidence(
            level=LipidResolutionLevel.LIPID_SPECIES, mapping_relation=LIPID_STRUCTURE_RELATION_BROAD
        )
    if qs == ns:
        return LipidStructureEvidence(
            level=LipidResolutionLevel.LIPID_SPECIES, mapping_relation=LIPID_STRUCTURE_RELATION_EXACT
        )
    return LipidStructureEvidence(level=LipidResolutionLevel.CONTRADICTED)


def _pair_level(node_key: str | None, independent_key: str | None) -> ResolutionLevel:
    """The level of one node key against one independent key, consistent with ``structural_agree``.

    A block-1 match with a differing block2[:8] is ``contradicted`` (never a silent stereo pass), the
    same way ``structural_agree`` returns False for that case.
    """
    n1, n2 = structural_key_parts(node_key)
    i1, i2 = structural_key_parts(independent_key)
    if n1 is None or i1 is None:
        return ResolutionLevel.UNAVAILABLE
    if n1 != i1:
        return ResolutionLevel.CONTRADICTED
    if n2 is None or i2 is None:
        # At least one side is first-block-only (MW/PubChem, or a graph key with no stereo layer):
        # only connectivity was compared. Two identical truncated strings are NOT exact -- no full
        # key or stereo layer was ever seen -- so never grant exact/structural here.
        return ResolutionLevel.CONNECTIVITY
    if str(node_key).strip().upper() == str(independent_key).strip().upper():
        return ResolutionLevel.EXACT_INCHIKEY
    return ResolutionLevel.STRUCTURAL if n2 == i2 else ResolutionLevel.CONTRADICTED


def resolve_level(
    node_keys: Iterable[str | None] | None, independent_key: str | None
) -> tuple[ResolutionLevel, ResolutionLevel]:
    """Best and worst resolution level of ``independent_key`` against the node's key list.

    Set-based over the full multi-valued node keys (never index 0). ``best`` is the strongest
    agreement any node key reaches; ``worst`` surfaces an internally inconsistent node (one key
    matching exactly while another contradicts) so it cannot silently grade ``exact_inchikey``.
    """
    keys = [k for k in (node_keys or []) if k and str(k).strip()]
    if not keys or not independent_key or not str(independent_key).strip():
        return ResolutionLevel.UNAVAILABLE, ResolutionLevel.UNAVAILABLE
    levels = [_pair_level(k, independent_key) for k in keys]
    best = max(levels, key=lambda lv: _LEVEL_RANK[lv])
    worst = min(levels, key=lambda lv: _LEVEL_RANK[lv])
    return best, worst


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
    # The registry returned several candidates with DISTINCT connectivity (e.g. a lipid species name
    # that maps to more than one first block). Not RESOLVED (no single structure to compare) and not
    # UNRESOLVABLE (the name is known); the candidate keys are carried on the result.
    AMBIGUOUS = "ambiguous"
    # Tier B is ENABLED for the run but this row is not a case an independent-registry lookup can
    # adjudicate (a non-small-molecule, an uncommitted row, or a committed node whose structure Tier B
    # never looked up). Kept DISTINCT from OFF: OFF means Tier B is disabled, OUT_OF_SCOPE means it is
    # on and simply does not apply here, so a reader cannot mistake an out-of-scope row for a disabled
    # run.
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class TierBResult:
    """One independent structure lookup for a QUERY NAME. Produced by the Tier B module, never here."""

    source: str | None
    inchikey_block: str | None
    outcome: TierBOutcome
    cache_state: str | None = None  # 'hit' | 'miss' | 'process_memo' | 'frozen' | None
    # Candidate InChIKeys when ``outcome is AMBIGUOUS`` (empty otherwise). A tuple so the frozen
    # dataclass stays hashable; sorted so it is order-independent.
    candidate_inchikeys: tuple[str, ...] = ()
    # Version of the Tier B freeze that served this result, when it came from the freeze (see
    # tier_b_snapshot); None for a live (non-frozen) result. Surfaced in the certificate provenance
    # as ``tier_b_snapshot_version`` so frozen evidence is auditable, mirroring RefMet's snapshot
    # version. Appended after the existing defaulted fields so the dataclass shape stays additive.
    version: str | None = None


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
    # RefMet (Metabolomics Workbench) availability for the row, on the same axis as
    # ``equivalent_ids_lookup_ok``: a runtime input about whether a source answered, NOT a verdict.
    # ``voted`` | ``no_match`` | ``unavailable`` | ``not_queried``. ``not_queried`` when RefMet was
    # not selected for the row, so the field is total. Appended after the existing defaulted fields
    # so the dataclass shape stays additive.
    refmet_availability: str = "not_queried"
    # WHICH RefMet source served the row, on a parallel axis to ``refmet_availability`` (a provenance
    # tag, not a verdict): ``local_snapshot`` | ``not_in_snapshot`` | ``live_api`` | ``unavailable`` |
    # ``not_queried``. With a pinned freeze present the default path is ``local_snapshot`` /
    # ``not_in_snapshot`` (the circuit breaker is OUT of it); without one it is ``live_api`` /
    # ``unavailable``, exactly as before. ``refmet_snapshot_version`` pins WHICH freeze produced a
    # snapshot-served vote (set only when the freeze served or was consulted for the row), else None.
    refmet_source: str = "not_queried"
    refmet_snapshot_version: str | None = None
    # Graded structural agreement between the committed node and the independent lookup, refining the
    # binary state. Derived from the SAME operand and block1/block2 comparison the state uses, so the
    # two never disagree: a corroborated row is exact/structural/connectivity, a contradicted row is
    # contradicted, and a row with no independent comparison is unavailable. ``_worst`` surfaces an
    # internally inconsistent multi-key node. Appended after the existing defaulted fields so the
    # dataclass shape stays additive.
    resolution_level: ResolutionLevel = ResolutionLevel.UNAVAILABLE
    resolution_level_worst: ResolutionLevel = ResolutionLevel.UNAVAILABLE
    # Graded agreement of a STRUCTURE-FREE lipid comparison, on an axis PARALLEL to ``resolution_level``
    # (which is InChIKey-block specific). ``UNAVAILABLE`` on every non-lipid and every InChIKey-bearing
    # row, so the two axes never overlap. Appended after the existing defaulted fields so the dataclass
    # shape stays additive.
    lipid_resolution_level: LipidResolutionLevel = LipidResolutionLevel.UNAVAILABLE
    # Version of the Tier B freeze that produced a frozen independent result (set only when the freeze
    # served the row), else None for a live result. A first-class field mirroring
    # ``refmet_snapshot_version`` so frozen Tier B evidence is auditable on the same footing: it is a
    # flat ``certificate_tier_b_snapshot_version`` column and an API response field, not only a
    # provenance entry. Appended after the existing defaulted fields so the dataclass shape stays
    # additive.
    tier_b_snapshot_version: str | None = None
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
            "resolution_level": self.resolution_level.value,
            "resolution_level_worst": self.resolution_level_worst.value,
            "lipid_resolution_level": self.lipid_resolution_level.value,
            "equivalent_ids_lookup_ok": self.equivalent_ids_lookup_ok,
            "selection_conflict": self.selection_conflict,
            "independent_source": self.independent_source,
            "independent_inchikey_block": self.independent_inchikey_block,
            "independent_of_selection": self.independent_of_selection,
            "tier_b_outcome": self.tier_b_outcome.value,
            "refusal_reason": self.refusal_reason,
            "refmet_availability": self.refmet_availability,
            "refmet_source": self.refmet_source,
            "refmet_snapshot_version": self.refmet_snapshot_version,
            "tier_b_snapshot_version": self.tier_b_snapshot_version,
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
            "certificate_resolution_level": self.resolution_level.value,
            "certificate_resolution_level_worst": self.resolution_level_worst.value,
            "certificate_lipid_resolution_level": self.lipid_resolution_level.value,
            "certificate_equivalent_ids_lookup_ok": self.equivalent_ids_lookup_ok,
            "certificate_selection_conflict": self.selection_conflict,
            "certificate_independent_source": self.independent_source,
            "certificate_independent_inchikey_block": self.independent_inchikey_block,
            "certificate_independent_of_selection": self.independent_of_selection,
            "certificate_tier_b_outcome": self.tier_b_outcome.value,
            "certificate_refusal_reason": self.refusal_reason,
            "certificate_refmet_availability": self.refmet_availability,
            "certificate_refmet_source": self.refmet_source,
            "certificate_refmet_snapshot_version": self.refmet_snapshot_version,
            "certificate_tier_b_snapshot_version": self.tier_b_snapshot_version,
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


def _default_provenance(tier_b: TierBResult | None, tier_b_enabled: bool | None = None) -> dict[str, Any]:
    # ``tier_b_enabled`` reports whether Tier B is ENABLED for the run, NOT whether a lookup happened
    # to run on this row. When the caller passes the config flag (the Mapper does), it is authoritative
    # so an out-of-scope row under an enabled run reads ``True`` rather than the misleading ``False``.
    # When None (a direct caller that does not know the flag), fall back to the historical derivation so
    # existing behaviour is byte-identical.
    enabled = (
        tier_b_enabled
        if tier_b_enabled is not None
        else (tier_b is not None and tier_b.outcome is not TierBOutcome.OFF)
    )
    return {
        "tier_b_enabled": enabled,
        "tier_b_cache_state": tier_b.cache_state if tier_b else None,
        # The freeze version is NOT recorded here: it is a FIRST-CLASS certificate field
        # (``tier_b_snapshot_version``, a flat column and an API field), mirroring RefMet's
        # ``refmet_snapshot_version`` which is likewise not carried in provenance.
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
    refmet_availability: str = "not_queried",
    refmet_source: str = "not_queried",
    refmet_snapshot_version: str | None = None,
    lipid_structure: LipidStructureEvidence | None = None,
    tier_b_enabled: bool | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
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
        refmet_availability: Whether the RefMet source answered for this row (voted / no_match /
            unavailable / not_queried). A runtime availability input like ``equivalent_ids_lookup_ok``,
            recorded on the certificate; it does not affect the state machine.
        refmet_source: WHICH RefMet source served the row (local_snapshot / not_in_snapshot /
            live_api / unavailable / not_queried). A provenance tag on a parallel axis to
            ``refmet_availability``; like it, recorded but not part of the state machine.
        refmet_snapshot_version: Version of the freeze that served (or was consulted for) the row,
            else None. Recorded, not part of the state machine.
        lipid_structure: A STRUCTURE-FREE lipid verdict, pre-computed upstream (this function never
            parses). It applies ONLY to a committed lipid node the graph lists no InChIKey for, where it
            can refine the ``unavailable`` state into ``corroborated`` / ``contradicted`` on the Goslin
            name-composition axis. None on every other path, so those paths are unchanged.
        tier_b_enabled: Whether Tier B is ENABLED for the run (the config flag), not whether a lookup
            ran. Passed by the Mapper so an out-of-scope row under an enabled run reports
            ``tier_b_outcome=out_of_scope`` and ``provenance.tier_b_enabled=True`` instead of the
            misleading ``off`` / ``False``. None preserves the historical behaviour for direct callers.
        extra_provenance: Extra provenance keys MERGED onto the (default or supplied) provenance,
            so a caller can mirror an out-of-band signal such as the lipid ``mapping_relation`` and
            ``ambiguous`` onto the certificate without replacing the default cache/Tier B provenance.
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
    # OUT_OF_SCOPE (Tier B on, this row not adjudicable) is DISTINCT from OFF (Tier B disabled). It is
    # reported only when the run enabled Tier B (``tier_b_enabled``) and no lookup ran for the row.
    if tier_b is not None:
        tier_b_outcome = tier_b.outcome
    elif tier_b_enabled:
        tier_b_outcome = TierBOutcome.OUT_OF_SCOPE
    else:
        tier_b_outcome = TierBOutcome.OFF
    resolution_level = ResolutionLevel.UNAVAILABLE
    resolution_level_worst = ResolutionLevel.UNAVAILABLE
    resolution_level_rule: str | None = None
    lipid_resolution_level = LipidResolutionLevel.UNAVAILABLE
    effective_comparison_rule = comparison_rule
    candidate_structure_count: int | None = None

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
        if lipid_structure is not None and lipid_structure.level is not LipidResolutionLevel.UNAVAILABLE:
            # STRUCTURE-FREE lipid check. The graph asserts no InChIKey (``structure_absent`` is honest
            # and stays), but the committed node's NAME and the query parse to a comparable Goslin
            # composition, so a real corroborated / contradicted verdict IS possible on a DIFFERENT axis
            # from the InChIKey block comparison. This does NOT reopen L21: L21 forbids contradicting a
            # node on the ground that it carries no InChIKey block; a Goslin composition mismatch is a
            # positive disagreement between two parsed names, tagged with its own comparison rule and
            # graded on ``lipid_resolution_level`` so it is never conflated with a block-level verdict.
            lipid_resolution_level = lipid_structure.level
            effective_comparison_rule = lipid_structure.comparison_rule
            if lipid_structure.level is LipidResolutionLevel.CONTRADICTED:
                state = CertificateState.CONTRADICTED
            else:
                state = CertificateState.CORROBORATED
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
            # Same operand and comparison as the state above, graded rather than binary, so the level
            # never disagrees with the state.
            resolution_level, resolution_level_worst = resolve_level(node_full_keys, tier_b.inchikey_block)
            resolution_level_rule = COMPARISON_RULE_INCHIKEY_LADDER
        elif tier_b is not None and tier_b.outcome is TierBOutcome.AMBIGUOUS and tier_b.candidate_inchikeys:
            # SET-BASED structure check. The registry returned several candidate structures for the query
            # name (a lipid species that pins no single connectivity) and the committed node DOES carry
            # graph InChIKeys, so corroborate when the two SETS of first blocks intersect. A
            # non-intersection is deliberately NOT a contradiction: the node may be a valid variant the
            # candidate set does not enumerate, so it stays ``uncorroborated``. The candidate count is
            # recorded either way.
            candidate_structure_count = len(tier_b.candidate_inchikeys)
            candidate_blocks = {str(k).split("-")[0].upper() for k in tier_b.candidate_inchikeys if k}
            if candidate_blocks & set(blocks):
                state = CertificateState.CORROBORATED
                resolution_level = ResolutionLevel.CONNECTIVITY
                resolution_level_worst = ResolutionLevel.CONNECTIVITY

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

    _prov = dict(provenance) if provenance is not None else _default_provenance(tier_b, tier_b_enabled)
    if resolution_level_rule is not None:
        _prov["resolution_level_rule"] = resolution_level_rule
    if candidate_structure_count is not None:
        _prov["candidate_structure_count"] = candidate_structure_count
    if lipid_structure is not None and lipid_structure.mapping_relation is not None:
        # The structure-free comparison's own relation (broad when the node is coarser than the query),
        # under a dedicated key so it never collides with the level-based ``mapping_relation`` the Mapper
        # mirrors from ``build_lipid_resolution`` via ``extra_provenance``.
        _prov["lipid_composition_relation"] = lipid_structure.mapping_relation
    if extra_provenance:
        _prov.update(extra_provenance)

    certificate = ResolutionCertificate(
        state=state,
        structure_status=structure_status,
        node_inchikey_blocks=blocks,
        comparison_rule=effective_comparison_rule,
        resolution_level=resolution_level,
        resolution_level_worst=resolution_level_worst,
        lipid_resolution_level=lipid_resolution_level,
        equivalent_ids_lookup_ok=equivalent_ids_lookup_ok,
        selection_conflict=selection_conflict,
        independent_source=independent_source,
        independent_inchikey_block=independent_block,
        independent_of_selection=independence,
        tier_b_outcome=tier_b_outcome,
        refusal_reason=refusal_reason,
        refmet_availability=refmet_availability,
        refmet_source=refmet_source,
        refmet_snapshot_version=refmet_snapshot_version,
        # Read the freeze version off the passed-in result (None for a live result). issue() stays
        # pure: it neither loads the freeze nor knows how the version was derived.
        tier_b_snapshot_version=(tier_b.version if tier_b is not None else None),
        provenance=_prov,
    )

    # G3, asserted at the point of construction as well as in the suite. On the INCHIKEY-BLOCK axis
    # ``contradicted`` is reachable only from ``structure_present``; this makes a future branch that
    # reintroduces a block-level contradiction on an absent structure fail here rather than in a figure.
    # The ONE deliberate exception is the structure-free lipid check, which contradicts on the Goslin
    # name-composition axis (two parsed names positively disagree) and is tagged with its own comparison
    # rule -- L21's "no InChIKey block, so nothing to contradict" reasoning does not apply to it.
    if (
        certificate.state is CertificateState.CONTRADICTED
        and not certificate.node_inchikey_blocks
        and certificate.comparison_rule != COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION
    ):
        raise AssertionError("contradicted issued for a node the graph asserts no structure for (L21)")
    return certificate
