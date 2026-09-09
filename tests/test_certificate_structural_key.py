"""Unit 3: the certificate's structural-key comparison (block1 + block2[:8]).

Tier B may now carry a FULL InChIKey (the lipid independent-structure source emits one), not only a
first block. The corroboration comparison must fold BOTH operands to ``block1 + block2[:8]`` so that:

  * a full-key Tier B result that agrees at block1+block2[:8] CORROBORATES (the regression this
    fixes: a naive first-block set-membership test would never find a full key in a set of first
    blocks and would flip every RESOLVED row to CONTRADICTED),
  * a genuine connectivity difference (block1 differs) still CONTRADICTS,
  * two stereoisomers (block1 same, block2[:8] differ, BOTH present) CONTRADICT — the new stereo
    tightening,
  * a first-block-only side degrades to a connectivity-only agreement, never a silent stereo pass.

Every case runs through the pure ``issue`` function; no network.
"""

from __future__ import annotations

from biomapper2.core.certificate import (
    CertificateState,
    TierBOutcome,
    TierBResult,
    structural_agree,
    structural_key_parts,
)
from biomapper2.core.certificate import issue as _issue_cert

# A full node InChIKey the graph asserts for the committed node (block1 + block2 + protonation).
NODE_FULL = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
NODE_BLOCK1 = "BSYNRYMUTXBXSQ"


def _issue(tier_b: TierBResult):
    return _issue_cert(
        chosen_kg_id="CHEBI:15365",
        is_small_molecule=True,
        kg_equivalent_ids={"INCHIKEY": [NODE_FULL]},
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        tier_b=tier_b,
        committed_node_sources=set(),
    )


def _resolved(block: str, source: str = "lipidmaps") -> TierBResult:
    return TierBResult(source=source, inchikey_block=block, outcome=TierBOutcome.RESOLVED)


# --- the pure helper ------------------------------------------------------------------------


def test_structural_key_parts_splits_block1_and_block2_8():
    b1, b2 = structural_key_parts("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert b1 == "BSYNRYMUTXBXSQ"
    assert b2 == "UHFFFAOY"  # block2 truncated to 8


def test_structural_key_parts_first_block_only_has_no_block2():
    b1, b2 = structural_key_parts("BSYNRYMUTXBXSQ")
    assert b1 == "BSYNRYMUTXBXSQ"
    assert b2 is None


def test_structural_agree_full_keys_matching():
    # Same block1, same block2[:8], differing protonation suffix and casing -> agree.
    assert structural_agree("BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "bsynrymutxbxsq-uhfffaoysa-m".upper()) is True


def test_structural_agree_connectivity_differs_is_false():
    assert structural_agree("AAAAAAAAAAAAAA-UHFFFAOYSA-N", "BBBBBBBBBBBBBB-UHFFFAOYSA-N") is False


def test_structural_agree_stereo_differs_when_both_present_is_false():
    # Same connectivity, different stereo layer, both sides carry block2.
    assert structural_agree("BSYNRYMUTXBXSQ-AAAAAAAASA-N", "BSYNRYMUTXBXSQ-BBBBBBBBSA-N") is False


def test_structural_agree_first_block_only_side_is_connectivity_only():
    # One side has no block2: agreement is connectivity-only, never a silent stereo pass.
    assert structural_agree("BSYNRYMUTXBXSQ", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N") is True


# --- through issue() ------------------------------------------------------------------------


def test_full_key_tier_b_agreeing_corroborates_not_contradicts():
    # The regression: a full key must not flip RESOLVED -> CONTRADICTED.
    cert = _issue(_resolved(NODE_FULL))
    assert cert.state is CertificateState.CORROBORATED


def test_connectivity_difference_contradicts():
    cert = _issue(_resolved("QQQQQQQQQQQQQQ-UHFFFAOYSA-N"))
    assert cert.state is CertificateState.CONTRADICTED


def test_stereoisomer_full_keys_contradict():
    # block1 same as the node, block2 differs, both present -> CONTRADICTED (stereo tightening).
    cert = _issue(_resolved(NODE_BLOCK1 + "-ZZZZZZZZSA-N"))
    assert cert.state is CertificateState.CONTRADICTED


def test_first_block_only_tier_b_matches_on_connectivity():
    # MW/PubChem still emit first-block only; against a full node key this is a connectivity match.
    cert = _issue(_resolved(NODE_BLOCK1, source="pubchem"))
    assert cert.state is CertificateState.CORROBORATED
