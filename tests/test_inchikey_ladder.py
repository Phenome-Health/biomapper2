"""The graded resolution level, derived from the same block1/block2 comparison ``structural_agree``
already uses. Pure and IO-free: a fixture table over key pairs, no network.

The level refines the existing binary agreement into exact / structural / connectivity, keeps
``contradicted`` for a block-1 disagreement (Unit 3's escalation reclaims some of those later), and
reports worst-case alongside best-case so an internally inconsistent multi-key node cannot grade
``exact``.
"""

from __future__ import annotations

from biomapper2.core.certificate import ResolutionLevel, resolve_level

# Two full keys sharing block1 and block2[:8] but differing in the final (charge/version) char.
GLUCOSE_A = "WQZGKKKJIJFFOK-GASJEMHNSA-N"
GLUCOSE_A_ANION = "WQZGKKKJIJFFOK-GASJEMHNSA-M"
# Same block1, different block2 (a stereoisomer): mannose shares glucose connectivity here for the test.
GLUCOSE_STEREO = "WQZGKKKJIJFFOK-QTVWNMPRSA-N"
# A different connectivity skeleton entirely.
DIFFERENT = "ATHGHQPFGPMSJY-UHFFFAOYSA-N"
FIRST_BLOCK_ONLY = "WQZGKKKJIJFFOK"


def test_identical_full_keys_are_exact() -> None:
    assert resolve_level([GLUCOSE_A], GLUCOSE_A) == (ResolutionLevel.EXACT_INCHIKEY, ResolutionLevel.EXACT_INCHIKEY)


def test_same_structural_key_different_final_char_is_structural() -> None:
    # block1 and block2[:8] agree; only the protonation/version char differs -> structural, not exact.
    best, worst = resolve_level([GLUCOSE_A], GLUCOSE_A_ANION)
    assert best == ResolutionLevel.STRUCTURAL and worst == ResolutionLevel.STRUCTURAL


def test_first_block_only_independent_is_connectivity() -> None:
    # MW/PubChem historically emit first-block-only; agreement is connectivity, never a silent stereo pass.
    best, worst = resolve_level([GLUCOSE_A], FIRST_BLOCK_ONLY)
    assert best == ResolutionLevel.CONNECTIVITY and worst == ResolutionLevel.CONNECTIVITY


def test_block1_match_block2_mismatch_is_contradicted() -> None:
    # Same connectivity, different stereo layer: structural_agree is False, so the state stays
    # contradicted and the level matches it (no silent stereo pass).
    best, _ = resolve_level([GLUCOSE_A], GLUCOSE_STEREO)
    assert best == ResolutionLevel.CONTRADICTED


def test_block1_disagreement_is_contradicted() -> None:
    assert resolve_level([GLUCOSE_A], DIFFERENT) == (ResolutionLevel.CONTRADICTED, ResolutionLevel.CONTRADICTED)


def test_two_identical_first_block_only_keys_are_connectivity_not_exact() -> None:
    # Regression (Greptile PR #77): a node first-block-only key equal to an MW/PubChem first-block
    # key only proves connectivity; grading it exact would publish an overstated certificate.
    assert resolve_level([FIRST_BLOCK_ONLY], FIRST_BLOCK_ONLY) == (
        ResolutionLevel.CONNECTIVITY,
        ResolutionLevel.CONNECTIVITY,
    )


def test_empty_sides_are_unavailable() -> None:
    assert resolve_level([], GLUCOSE_A) == (ResolutionLevel.UNAVAILABLE, ResolutionLevel.UNAVAILABLE)
    assert resolve_level([GLUCOSE_A], None) == (ResolutionLevel.UNAVAILABLE, ResolutionLevel.UNAVAILABLE)


def test_match_not_at_index_zero_is_found() -> None:
    # Guards the multi-valued [0] landmine: the exact match is the second node key.
    best, _ = resolve_level([DIFFERENT, GLUCOSE_A], GLUCOSE_A)
    assert best == ResolutionLevel.EXACT_INCHIKEY


def test_internally_inconsistent_node_reports_worst_case() -> None:
    # One node key matches exactly, another contradicts: best is exact, worst surfaces the conflict.
    best, worst = resolve_level([GLUCOSE_A, DIFFERENT], GLUCOSE_A)
    assert best == ResolutionLevel.EXACT_INCHIKEY
    assert worst == ResolutionLevel.CONTRADICTED
