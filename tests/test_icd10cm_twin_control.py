"""Unit 1: the ICD-10-CM twin control gate's pure comparison logic.

The study (``studies/analysis/icd10cm_twin_control.py``) decides whether a provided ``ICD10CM:X`` that
KRAKEN cannot link may be committed through its ``ICD10:X`` twin as an EXACT match or only as a BROAD
one. Every number it emits feeds a shipped config constant, so the arithmetic is pinned here on tiny
fixtures, with no network.

Gate (decided at Checkpoint 2): independent CDC-title vs KRAKEN-label agreement, at or above
``GATE_THRESHOLD`` on rows with a label and with at least ``GATE_MIN_N`` of them. Mondo SSSOM
agreement is reported as Babel-Mondo CONSISTENCY only, never as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.analysis import icd10cm_twin_control as study

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------------------------------------
# CDC order-file parsing
# ------------------------------------------------------------------------------------------------
_ORDER_LINES = [
    "00001 A00     0 Cholera                                                      Cholera",
    "00002 A000    1 Cholera due to Vibrio cholerae 01, biovar cholerae           "
    "Cholera due to Vibrio cholerae 01, biovar cholerae",
    "00005 E1165   1 Type 2 diabetes mellitus with hyperglycemia                  "
    "Type 2 diabetes mellitus with hyperglycemia",
    "00006 T360X1A 1 Poisoning by penicillins, accidental (unintentional), init   "
    "Poisoning by penicillins, accidental (unintentional), initial encounter",
]


def test_order_file_rows_are_redotted_and_flagged() -> None:
    rows = study.parse_order_file("\r\n".join(_ORDER_LINES) + "\r\n")
    assert [r.code for r in rows] == ["A00", "A00.0", "E11.65", "T36.0X1A"]
    assert [r.billable for r in rows] == [False, True, True, True]
    # The LONG title is the comparison text; the short one is truncated by CDC.
    assert rows[3].title == "Poisoning by penicillins, accidental (unintentional), initial encounter"


def test_redot_leaves_three_character_codes_alone() -> None:
    assert study.redot("I10") == "I10"
    assert study.redot("E1165") == "E11.65"


# ------------------------------------------------------------------------------------------------
# Title agreement (the GATE)
# ------------------------------------------------------------------------------------------------
def test_title_matching_label_agrees() -> None:
    verdict, score = study.title_verdict("Fabry disease", ["Fabry disease", "Angiokeratoma corporis diffusum"])
    assert verdict == study.TITLE_AGREE
    assert score == pytest.approx(1.0)


def test_unspecified_boilerplate_does_not_cause_a_disagreement() -> None:
    verdict, _ = study.title_verdict("Essential (primary) hypertension, unspecified", ["essential hypertension"])
    assert verdict == study.TITLE_AGREE
    verdict, _ = study.title_verdict("Cholera, NOS", ["cholera"])
    assert verdict == study.TITLE_AGREE


def test_genuinely_different_concept_disagrees() -> None:
    verdict, score = study.title_verdict("Fabry disease", ["Gaucher disease type 1"])
    assert verdict == study.TITLE_DISAGREE
    assert score is not None and score < study.TITLE_MATCH_CUTOFF


def test_best_label_wins_over_synonyms() -> None:
    # A node whose primary name is terse but a synonym carries the CDC wording still agrees.
    verdict, _ = study.title_verdict("Type 2 diabetes mellitus", ["T2DM", "type 2 diabetes mellitus"])
    assert verdict == study.TITLE_AGREE


def test_node_with_no_label_is_counted_separately() -> None:
    verdict, score = study.title_verdict("Fabry disease", [])
    assert verdict == study.TITLE_NO_LABEL
    assert score is None


# ------------------------------------------------------------------------------------------------
# Gate decision
# ------------------------------------------------------------------------------------------------
def test_gate_passes_at_threshold_with_enough_rows() -> None:
    gate = study.gate_decision(agree=190, disagree=10, threshold=0.95, min_n=200)
    assert gate["status"] == study.GATE_PASS
    assert gate["n"] == 200
    assert gate["rate"] == pytest.approx(0.95)


def test_gate_fails_below_threshold() -> None:
    gate = study.gate_decision(agree=189, disagree=11, threshold=0.95, min_n=200)
    assert gate["status"] == study.GATE_FAIL


def test_gate_reports_insufficient_n_rather_than_pass_or_fail() -> None:
    gate = study.gate_decision(agree=150, disagree=0, threshold=0.95, min_n=200)
    assert gate["status"] == study.GATE_INSUFFICIENT_N


def test_twin_relation_is_broad_until_a_reviewed_pass() -> None:
    passing = study.gate_decision(agree=1000, disagree=1, threshold=0.95, min_n=200)
    assert study.twin_relation_for(passing, reviewed=False) == "broad"
    assert study.twin_relation_for(passing, reviewed=True) == "exact"
    failing = study.gate_decision(agree=10, disagree=1000, threshold=0.95, min_n=200)
    assert study.twin_relation_for(failing, reviewed=True) == "broad"
    thin = study.gate_decision(agree=5, disagree=0, threshold=0.95, min_n=200)
    assert study.twin_relation_for(thin, reviewed=True) == "broad"


# ------------------------------------------------------------------------------------------------
# Mondo SSSOM consistency (REPORTED, not gating)
# ------------------------------------------------------------------------------------------------
_SSSOM = "\n".join(
    [
        "# curie_map:",
        "#   MONDO: http://purl.obolibrary.org/obo/MONDO_",
        "subject_id\tsubject_label\tpredicate_id\tobject_id\tobject_label\tmapping_justification",
        "MONDO:1\tcholera\tskos:exactMatch\tICD10CM:A00\tCholera\tsemapv:UnspecifiedMatching",
        "MONDO:2\tintestinal infectious disease\tskos:exactMatch\tICD10CM:A00-A09\tx\tsemapv:UnspecifiedMatching",
        "MONDO:3\tgastric ulcer\tskos:exactMatch\tICD10CM:K28\tx\tsemapv:UnspecifiedMatching",
        "MONDO:3\tgastric ulcer\tskos:exactMatch\tICD10CM:K28.0\tx\tsemapv:UnspecifiedMatching",
        "MONDO:4\tsomething\tskos:broadMatch\tICD10CM:B00\tx\tsemapv:UnspecifiedMatching",
        "MONDO:5\tother\tskos:exactMatch\tDOID:4\tx\tsemapv:UnspecifiedMatching",
    ]
)


def test_sssom_parse_keeps_icd10cm_exact_rows_and_flags_ranges() -> None:
    rows = study.parse_sssom_icd10cm_exact(_SSSOM)
    assert [(r.mondo, r.code, r.is_range) for r in rows] == [
        ("MONDO:1", "A00", False),
        ("MONDO:2", "A00-A09", True),
        ("MONDO:3", "K28", False),
        ("MONDO:3", "K28.0", False),
    ]


def test_consistency_counts_ranges_out_and_twin_null_separately() -> None:
    rows = study.parse_sssom_icd10cm_exact(_SSSOM)
    canonical: dict[str, str | None] = {
        "ICD10:A00": "MONDO:1",  # consistent
        "MONDO:1": "MONDO:1",
        "ICD10:K28": "MONDO:3",  # consistent
        "MONDO:3": "MONDO:3",
        "ICD10:K28.0": "MONDO:99",  # parent and child evaluated independently: this one disagrees
        "MONDO:2": "MONDO:2",
    }
    result = study.consistency_check(rows, canonical)
    assert result["counts"] == {"consistent": 2, "inconsistent": 1, "twin_null": 0, "range_excluded": 1}
    assert result["inconsistent_codes"] == ["K28.0"]
    by_code = {r["code"]: r["verdict"] for r in result["rows"]}
    assert by_code == {"A00": "consistent", "K28": "consistent", "K28.0": "inconsistent"}


def test_twin_null_is_not_a_disagreement() -> None:
    rows = study.parse_sssom_icd10cm_exact(_SSSOM)
    canonical = {"MONDO:1": "MONDO:1", "ICD10:A00": None, "ICD10:K28": None, "ICD10:K28.0": None}
    result = study.consistency_check(rows, canonical)
    assert result["counts"]["twin_null"] == 3
    assert result["counts"]["inconsistent"] == 0


# ------------------------------------------------------------------------------------------------
# Denylist + review sheet
# ------------------------------------------------------------------------------------------------
def test_denylist_is_the_sorted_union_of_both_disagreement_sets() -> None:
    assert study.build_denylist(["F40.0", "A00"], ["K28.0", "A00"]) == ["A00", "F40.0", "K28.0"]


def test_review_sample_is_seeded_and_capped() -> None:
    mismatches = [{"code": f"X{i:02d}"} for i in range(80)]
    first = study.sample_for_review(mismatches, k=50, seed=7)
    second = study.sample_for_review(mismatches, k=50, seed=7)
    assert first == second
    assert len(first) == 50
    # Fewer mismatches than the cap: all of them, in code order.
    assert study.sample_for_review(list(reversed(mismatches[:3])), k=50, seed=7) == mismatches[:3]


def test_denylist_file_is_written_in_the_shape_the_runtime_reads(tmp_path: Path) -> None:
    path = tmp_path / "denylist.json"
    study.write_denylist(path, ["F40.0", "A00"], source_artifact="x.json", kg_build="2.3.0")
    payload = json.loads(path.read_text())
    assert payload["codes"] == ["A00", "F40.0"]
    assert payload["source_artifact"] == "x.json"

    from biomapper2.core import icd10cm_fallback

    assert icd10cm_fallback.load_twin_denylist(path) == frozenset({"A00", "F40.0"})


# ------------------------------------------------------------------------------------------------
# Error path: a Kestrel failure aborts the run and writes NO artifact
# ------------------------------------------------------------------------------------------------
def test_kestrel_failure_aborts_without_writing_an_artifact(tmp_path: Path) -> None:
    order_rows = study.parse_order_file("\n".join(_ORDER_LINES))

    def _boom(curies: list[str]) -> dict[str, str | None]:
        raise RuntimeError("kestrel 503")

    with pytest.raises(RuntimeError, match="kestrel 503"):
        study.run_and_write(
            order_rows=order_rows,
            sssom_rows=[],
            canonicalize=_boom,
            get_nodes=lambda ids: {},
            inputs={},
            kg_build={},
            out_dir=tmp_path,
            stable_dir=tmp_path / "stable",
            denylist_path=tmp_path / "deny.json",
            review_path=tmp_path / "review.tsv",
        )
    assert list(tmp_path.rglob("*.json")) == []


def test_offline_run_produces_complete_artifact(tmp_path: Path) -> None:
    order_rows = study.parse_order_file("\n".join(_ORDER_LINES))
    canonical = {"ICD10:A00": "MONDO:1", "ICD10:A00.0": "MONDO:2", "ICD10:E11.65": None, "ICD10:T36.0X1A": None}

    nodes = {
        "MONDO:1": {"name": "cholera", "synonyms": [], "equivalent_ids": ["ICD10:A00"], "provided_by": ["babel"]},
        "MONDO:2": {"name": "plague", "synonyms": [], "equivalent_ids": ["ICD10:A00.0"], "provided_by": ["babel"]},
    }
    artifact = study.run_and_write(
        order_rows=order_rows,
        sssom_rows=[],
        canonicalize=lambda curies: {c: canonical.get(c) for c in curies},
        get_nodes=lambda ids: {i: nodes[i] for i in ids if i in nodes},
        inputs={"cdc_order_zip": {"sha256": "abc"}},
        kg_build={"kg_version": "2.3.0"},
        out_dir=tmp_path,
        stable_dir=tmp_path / "stable",
        denylist_path=tmp_path / "deny.json",
        review_path=tmp_path / "review.tsv",
        min_n=1,
    )
    assert artifact["title_gate"]["counts"] == {"agree": 1, "disagree": 1, "no_label": 0}
    assert artifact["regression"]["twin_hit_count"] == 2
    assert artifact["regression"]["codes"]["A00"] == {"kg_id": "MONDO:1", "route": "twin"}
    assert artifact["gate"]["reviewed"] is False
    assert artifact["gate"]["twin_relation"] == "broad"
    assert json.loads((tmp_path / "deny.json").read_text())["codes"] == ["A00.0"]
    assert (tmp_path / "stable" / "icd10cm_twin_control.json").exists()
    assert "A00.0" in (tmp_path / "review.tsv").read_text()
