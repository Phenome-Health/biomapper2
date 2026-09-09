"""The certificate's state machine, exercised as a table.

Why this file exists
--------------------
The preprint's central claim is that a name-input resolution carries a *structural certificate* and
that the resolver refuses when one cannot be issued. Until this module existed, the code computed a
structural verdict inside ``Resolver._choose_best_kg_id`` and threw it away; the only thing that
escaped was ``chosen_kg_id_review``, whose ``None`` was overloaded.

The one thing these tests must not let regress is L21: ``structure_absent`` is **unverifiable**, not
wrong. A node the graph asserts no structure for can never be ``contradicted``, because there is
nothing to contradict it with. ``test_structure_absent_is_never_contradicted`` is that guard (G3).
"""

from __future__ import annotations

import dataclasses

import pytest

from biomapper2.core.certificate import (
    COMPARISON_RULE_FIRST_BLOCK_SET_INTERSECTION,
    SELECTION_CONFLICT_VALUES,
    CertificateState,
    ResolutionCertificate,
    StructureStatus,
    TierBOutcome,
    TierBResult,
    derive_chosen_kg_id_review,
    issue,
)

NODE = "CHEBI:15365"
BLOCKS = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "BSYNRYMUTXBXSQ-UHFFFAOYSA-M"]}
OTHER_BLOCK = "QQQQQQQQQQQQQQ"


def _tier_b(outcome: TierBOutcome, block: str | None = None, source: str | None = None) -> TierBResult:
    return TierBResult(source=source, inchikey_block=block, outcome=outcome)


def _issue(**overrides):
    """Tier-A defaults: a small molecule with a committed node whose graph record carries a key."""
    kwargs = dict(
        chosen_kg_id=NODE,
        is_small_molecule=True,
        kg_equivalent_ids=BLOCKS,
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
    )
    kwargs.update(overrides)
    return issue(**kwargs)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------------
# Tier A — the default path
# --------------------------------------------------------------------------------------------


def test_certificate_is_frozen_and_carries_the_comparison_rule() -> None:
    cert = _issue()
    assert isinstance(cert, ResolutionCertificate)
    assert cert.comparison_rule == COMPARISON_RULE_FIRST_BLOCK_SET_INTERSECTION
    with pytest.raises(dataclasses.FrozenInstanceError):
        cert.state = CertificateState.CONTRADICTED  # type: ignore[misc]


def test_structure_present_is_uncorroborated_with_tier_b_off() -> None:
    cert = _issue()
    assert cert.structure_status is StructureStatus.STRUCTURE_PRESENT
    assert cert.state is CertificateState.UNCORROBORATED
    assert cert.node_inchikey_blocks == ["BSYNRYMUTXBXSQ"]  # sorted, de-duplicated to first blocks


def test_structure_absent_is_unavailable_never_contradicted() -> None:
    cert = _issue(kg_equivalent_ids={"HMDB": ["HMDB0001879"]})
    assert cert.structure_status is StructureStatus.STRUCTURE_ABSENT
    assert cert.state is CertificateState.UNAVAILABLE
    assert cert.node_inchikey_blocks == []


def test_non_small_molecule_is_not_applicable() -> None:
    """Nothing at stage 5 gates on category, so without this state every gene row reads
    ``structure_absent`` -> ``unavailable``: the strongest negative claim about the one population
    the certificate was never designed to judge."""
    cert = _issue(is_small_molecule=False, kg_equivalent_ids={"HGNC": ["1097"]})
    assert cert.structure_status is StructureStatus.NOT_APPLICABLE
    assert cert.state is CertificateState.NOT_APPLICABLE


def test_no_committed_node_is_unavailable_not_absent() -> None:
    """A row with no committed node has no answer to certify. That is unavailable, and its
    structure status is not_applicable -- calling it ``structure_absent`` would assert something
    about a node that does not exist."""
    cert = _issue(chosen_kg_id=None, kg_equivalent_ids={})
    assert cert.state is CertificateState.UNAVAILABLE
    assert cert.structure_status is StructureStatus.NOT_APPLICABLE


def test_kestrel_outage_refuses_a_tier_a_verdict_and_is_not_structure_absent() -> None:
    """``Linker.get_equivalent_ids`` returns {} on any exception and only logs, so a transient
    /get-nodes failure would otherwise mark an entire run ``unavailable`` via ``structure_absent``
    and an offline rerun on the resulting TSV could never tell the two apart."""
    cert = _issue(kg_equivalent_ids={}, equivalent_ids_lookup_ok=False)
    assert cert.state is CertificateState.UNAVAILABLE
    assert cert.structure_status is StructureStatus.NOT_APPLICABLE
    assert cert.equivalent_ids_lookup_ok is False

    absent = _issue(kg_equivalent_ids={}, equivalent_ids_lookup_ok=True)
    assert absent.structure_status is StructureStatus.STRUCTURE_ABSENT
    assert absent.equivalent_ids_lookup_ok is True


# --------------------------------------------------------------------------------------------
# Tier B — opt-in refinement of ``structure_present`` only
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier_b", "expected_state"),
    [
        (None, CertificateState.UNCORROBORATED),
        (_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"), CertificateState.CORROBORATED),
        (_tier_b(TierBOutcome.RESOLVED, OTHER_BLOCK, "pubchem"), CertificateState.CONTRADICTED),
        (_tier_b(TierBOutcome.UNRESOLVABLE, None, "pubchem"), CertificateState.UNCORROBORATED),
        (_tier_b(TierBOutcome.LOOKUP_FAILED, None, "pubchem"), CertificateState.UNCORROBORATED),
    ],
)
def test_tier_b_refines_structure_present(tier_b: TierBResult | None, expected_state: CertificateState) -> None:
    cert = _issue(tier_b=tier_b)
    assert cert.state is expected_state


def test_a_failed_lookup_is_distinguishable_from_an_unresolvable_name() -> None:
    """Otherwise a rate-limited PubChem turns the figure into a network artifact that nothing
    records. Both stay ``uncorroborated`` -- neither is evidence -- but they are not the same event."""
    failed = _issue(tier_b=_tier_b(TierBOutcome.LOOKUP_FAILED, None, "pubchem"))
    unresolvable = _issue(tier_b=_tier_b(TierBOutcome.UNRESOLVABLE, None, "pubchem"))
    assert failed.tier_b_outcome is TierBOutcome.LOOKUP_FAILED
    assert unresolvable.tier_b_outcome is TierBOutcome.UNRESOLVABLE
    assert failed.state is unresolvable.state is CertificateState.UNCORROBORATED


@pytest.mark.parametrize("tier_b_outcome", list(TierBOutcome))
def test_structure_absent_stays_unavailable_under_every_tier_b_outcome(tier_b_outcome: TierBOutcome) -> None:
    """L21, stated as a table: there is nothing to compare an absent structure against, so no
    Tier-B result can move the row off ``unavailable``."""
    cert = _issue(
        kg_equivalent_ids={"HMDB": ["HMDB0001879"]},
        tier_b=_tier_b(tier_b_outcome, "BSYNRYMUTXBXSQ", "pubchem"),
    )
    assert cert.state is CertificateState.UNAVAILABLE


@pytest.mark.parametrize(
    ("is_small_molecule", "equiv", "lookup_ok", "chosen"),
    [
        (False, {"HGNC": ["1097"]}, True, NODE),
        (True, {}, False, NODE),
        (True, BLOCKS, True, None),
    ],
)
def test_tier_b_cannot_promote_a_row_that_has_no_tier_a_structure(
    is_small_molecule: bool, equiv: dict, lookup_ok: bool, chosen: str | None
) -> None:
    cert = _issue(
        is_small_molecule=is_small_molecule,
        kg_equivalent_ids=equiv,
        equivalent_ids_lookup_ok=lookup_ok,
        chosen_kg_id=chosen,
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"),
    )
    assert cert.state in {CertificateState.NOT_APPLICABLE, CertificateState.UNAVAILABLE}


# --------------------------------------------------------------------------------------------
# G3 — the invariant that makes the paper's claim safe
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("is_small_molecule", [True, False])
@pytest.mark.parametrize("lookup_ok", [True, False])
@pytest.mark.parametrize("chosen", [NODE, None])
@pytest.mark.parametrize("equiv", [BLOCKS, {"HMDB": ["HMDB0001879"]}, {}])
@pytest.mark.parametrize("selection_conflict", SELECTION_CONFLICT_VALUES)
@pytest.mark.parametrize(
    "tier_b",
    [
        None,
        _tier_b(TierBOutcome.OFF),
        _tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"),
        _tier_b(TierBOutcome.RESOLVED, OTHER_BLOCK, "metabolomics-workbench"),
        _tier_b(TierBOutcome.UNRESOLVABLE, None, "pubchem"),
        _tier_b(TierBOutcome.LOOKUP_FAILED, None, "pubchem"),
    ],
)
def test_structure_absent_is_never_contradicted(
    is_small_molecule: bool,
    lookup_ok: bool,
    chosen: str | None,
    equiv: dict,
    selection_conflict: str | None,
    tier_b: TierBResult | None,
) -> None:
    """G3. ``contradicted`` implies the committed node carried at least one InChIKey block.

    Exhaustive over the full input cross-product rather than spot-checked: the failure mode this
    guards is a future branch that assigns ``contradicted`` from some *other* signal, and a
    hand-picked example would not see it. Combinations the resolver cannot produce (a review flag
    without a committed small-molecule node) are excluded rather than asserted on -- ``issue``
    rejects those, and a separate test covers the rejection.
    """
    if selection_conflict is not None and (chosen is None or not is_small_molecule):
        pytest.skip("resolver cannot emit a selection_conflict without a committed small-molecule node")
    cert = issue(
        chosen_kg_id=chosen,
        is_small_molecule=is_small_molecule,
        kg_equivalent_ids=equiv,
        equivalent_ids_lookup_ok=lookup_ok,
        selection_conflict=selection_conflict,
        tier_b=tier_b,
    )
    if cert.state is CertificateState.CONTRADICTED:
        assert cert.node_inchikey_blocks, "contradicted issued for a node with no InChIKey block"
        assert cert.structure_status is StructureStatus.STRUCTURE_PRESENT


def test_state_and_selection_conflict_are_different_axes() -> None:
    """``divergent_refmet`` is an intra-KG selection conflict -- both sides come from the graph.
    Folding it into ``contradicted`` would restate a KG-internal disagreement as independent
    refutation, which is the same class of error as L21."""
    cert = _issue(selection_conflict="divergent_refmet")
    assert cert.selection_conflict == "divergent_refmet"
    assert cert.state is CertificateState.UNCORROBORATED


def test_conflict_no_structure_may_co_occur_with_structure_present() -> None:
    """``connectivity_match`` returns None if *either* node is unresolvable, so the committed node
    may still carry a key. The legacy flag name is misleading; it is kept unchanged for
    compatibility and documented rather than renamed."""
    cert = _issue(selection_conflict="conflict_no_structure")
    assert cert.structure_status is StructureStatus.STRUCTURE_PRESENT
    assert cert.selection_conflict == "conflict_no_structure"


# --------------------------------------------------------------------------------------------
# L26 — independence is claimed only where it holds
# --------------------------------------------------------------------------------------------


def test_mw_tier_b_is_not_independent_when_refmet_supplied_the_committed_node() -> None:
    """Tier B's first hop is MW's /refmet/name, keyed on the same query name the RefMet annotator
    used to produce the candidate that won. On those rows Tier B asks RefMet whether RefMet was
    right, so a ``corroborated`` verdict is circular."""
    cert = _issue(
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "metabolomics-workbench"),
        committed_node_sources={"metabolomics-workbench"},
    )
    assert cert.state is CertificateState.CORROBORATED
    assert cert.independent_of_selection is False


def test_mw_tier_b_is_independent_when_another_annotator_supplied_the_node() -> None:
    cert = _issue(
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "metabolomics-workbench"),
        committed_node_sources={"kestrel-hybrid"},
    )
    assert cert.independent_of_selection is True


def test_pubchem_tier_b_is_independent_of_the_refmet_selector() -> None:
    cert = _issue(
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"),
        committed_node_sources={"metabolomics-workbench"},
    )
    assert cert.independent_of_selection is True


@pytest.mark.parametrize(
    ("label", "is_small_molecule", "equiv", "lookup_ok", "chosen"),
    [
        ("structure_absent", True, {"HMDB": ["HMDB0001879"]}, True, NODE),
        ("non_small_molecule", False, {"HGNC": ["1097"]}, True, NODE),
        ("kestrel_outage", True, {}, False, NODE),
        ("no_committed_node", True, BLOCKS, True, None),
    ],
)
def test_independent_evidence_is_reported_only_where_a_comparison_happened(
    label: str, is_small_molecule: bool, equiv: dict, lookup_ok: bool, chosen: str | None
) -> None:
    """A certificate must not show an independent block beside a state that never used it.

    ``structure_absent`` is the case this adds: the state stays ``unavailable`` under every Tier B
    outcome (L21), so publishing ``independent_source``/``independent_inchikey_block`` there reads as
    corroborating evidence the certificate declined to act on -- and a consumer stratifying on
    ``independent_source`` would pull those rows into a source curve they were never adjudicated in.
    ``Mapper`` does not even spend the lookup; this pins the same rule for a direct caller.
    """
    cert = issue(
        chosen_kg_id=chosen,
        is_small_molecule=is_small_molecule,
        kg_equivalent_ids=equiv,
        equivalent_ids_lookup_ok=lookup_ok,
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"),
    )
    assert cert.independent_source is None, label
    assert cert.independent_inchikey_block is None, label
    assert cert.independent_of_selection is None, label
    # The OUTCOME still travels: whether a lookup was made is a fact about the run, not a verdict.
    assert cert.tier_b_outcome is TierBOutcome.RESOLVED, label


def test_independent_evidence_is_reported_when_a_comparison_did_happen() -> None:
    """Guards the guard above: a rule that blanks the fields everywhere would leave Panel B with no
    stratification and every gate trivially unmeasurable."""
    cert = _issue(tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"))
    assert cert.structure_status is StructureStatus.STRUCTURE_PRESENT
    assert cert.independent_source == "pubchem"
    assert cert.independent_inchikey_block == "BSYNRYMUTXBXSQ"
    assert cert.independent_of_selection is True


def test_independence_is_unclaimed_when_tier_b_is_off() -> None:
    """None, not True. An unmade comparison is not an independent one."""
    assert _issue().independent_of_selection is None
    assert _issue(tier_b=_tier_b(TierBOutcome.LOOKUP_FAILED, None, "pubchem")).independent_of_selection is None


# --------------------------------------------------------------------------------------------
# Legacy derivation (C4 / L20) and the invariant it rests on
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", SELECTION_CONFLICT_VALUES)
def test_legacy_review_flag_is_derived_identically(flag: str | None) -> None:
    """Three values, not four. The brainstorm's 'four states' counts situations collapsed into
    ``None``, which is a different thing -- ``resolver.py`` returns exactly these three."""
    cert = _issue(selection_conflict=flag)
    assert derive_chosen_kg_id_review(cert) == flag


def test_legacy_derivation_rests_on_a_stated_invariant() -> None:
    """A non-None flag always co-occurs with a committed node and a small-molecule category:
    the resolver only reaches the flagging branch inside that guard."""
    with pytest.raises(ValueError, match="selection_conflict"):
        issue(
            chosen_kg_id=None,
            is_small_molecule=True,
            kg_equivalent_ids={},
            equivalent_ids_lookup_ok=True,
            selection_conflict="divergent_refmet",
        )
    with pytest.raises(ValueError, match="selection_conflict"):
        issue(
            chosen_kg_id=NODE,
            is_small_molecule=False,
            kg_equivalent_ids={},
            equivalent_ids_lookup_ok=True,
            selection_conflict="divergent_refmet",
        )


# --------------------------------------------------------------------------------------------
# Serialization — plain types only, at both surfaces
# --------------------------------------------------------------------------------------------


def test_api_dict_is_plain_json_types() -> None:
    import json

    cert = _issue(
        selection_conflict="divergent_refmet",
        tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"),
        committed_node_sources={"metabolomics-workbench"},
    )
    payload = cert.to_api_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["state"] == "corroborated"
    assert payload["structure_status"] == "structure_present"
    assert payload["tier_b_outcome"] == "resolved"
    assert payload["node_inchikey_blocks"] == ["BSYNRYMUTXBXSQ"]
    assert payload["refusal_reason"] is None  # reserved; ships in the follow-up PR (L28)


def test_flat_columns_are_scalars_and_namespaced_to_the_committed_answer() -> None:
    """No repr'd dicts: an object column through ``df.to_csv`` emits ``ResolutionCertificate(...)``,
    reintroducing exactly the ``ast.literal_eval``-only column this design exists to eliminate."""
    cert = _issue(tier_b=_tier_b(TierBOutcome.RESOLVED, "BSYNRYMUTXBXSQ", "pubchem"))
    flat = cert.to_flat_columns()
    assert all(key.startswith("certificate_") for key in flat)
    assert all(value is None or isinstance(value, (str, bool, int, float)) for value in flat.values())
    assert flat["certificate_state"] == "uncorroborated" or flat["certificate_state"] == "corroborated"
    assert flat["certificate_node_inchikey_blocks"] == "BSYNRYMUTXBXSQ"


def test_flat_columns_join_multiple_blocks_on_a_pipe() -> None:
    cert = _issue(kg_equivalent_ids={"INCHIKEY": ["AAAAAAAAAAAAAA-X-N", "ZZZZZZZZZZZZZZ-Y-N"]})
    assert cert.to_flat_columns()["certificate_node_inchikey_blocks"] == "AAAAAAAAAAAAAA|ZZZZZZZZZZZZZZ"


def test_provenance_records_tier_b_state_and_both_cache_stores() -> None:
    cert = _issue()
    provenance = cert.to_api_dict()["provenance"]
    assert provenance["tier_b_enabled"] is False
    # Both stores, because the confound that motivated this lives in the Kestrel store (1-hour
    # expiry), not the structure store (no expiry).
    assert "kestrel_cache_store" in provenance
    assert "structure_cache_store" in provenance
    flat = cert.to_flat_columns()
    assert flat["certificate_provenance_tier_b_enabled"] is False


# ----------------------------------------------------------------------------------------------
# Case folding on the SHIPPED comparison.
#
# The study module gained this fold twice (its `_first_block`, then both operands of its
# independence control) while the production comparison that actually assigns
# corroborated/contradicted folded nothing. The instrument was hardened against a defect the
# instrument does not have; the code being measured kept it. Worse, the two disagree in the
# direction that hides: the audit folds, so it scores a case-mismatched row as AGREEING while
# `issue()` scored it as REFUTED -- an inverted Panel B that every gate passes.
# ----------------------------------------------------------------------------------------------

_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def _case_tier_b(block: str) -> TierBResult:
    return TierBResult(source="pubchem", inchikey_block=block, outcome=TierBOutcome.RESOLVED)


def _issue_case(kg_key: str, tier_b_block: str):
    return issue(
        chosen_kg_id="CHEBI:15365",
        is_small_molecule=True,
        kg_equivalent_ids={"INCHIKEY": [kg_key]},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        tier_b=_case_tier_b(tier_b_block),
        committed_node_sources=set(),
    )


@pytest.mark.parametrize(
    "kg_key,tier_b_block",
    [
        (_KEY, "bsynrymutxbxsq"),  # KG upper, Tier B lower
        (_KEY.lower(), "BSYNRYMUTXBXSQ"),  # KG lower, Tier B upper
        (_KEY.lower(), "bsynrymutxbxsq"),  # both lower
        (_KEY, "BSYNRYMUTXBXSQ"),  # both upper (the ordinary case)
    ],
)
def test_the_same_structure_corroborates_whatever_its_casing(kg_key, tier_b_block):
    """`contradicted` asserts independent evidence REFUTES the node -- the strongest claim here.

    Two spellings of one molecule must never produce it. Both operands come from sources that do
    not constrain casing: the KG on one side, MW and PubChem on the other.
    """
    assert _issue_case(kg_key, tier_b_block).state is CertificateState.CORROBORATED


def test_a_genuine_structural_disagreement_still_contradicts():
    """Guards the guard: folding must not turn the comparison into one that cannot refuse."""
    assert _issue_case(_KEY, "ZZZZZZZZZZZZZZ").state is CertificateState.CONTRADICTED


def test_the_emitted_blocks_are_canonical_regardless_of_input_casing():
    """Folded at the PRODUCER too, so the published column is canonical and not merely compared as if."""
    certificate = _issue_case(_KEY.lower(), "bsynrymutxbxsq")
    assert certificate.node_inchikey_blocks == ["BSYNRYMUTXBXSQ"]
