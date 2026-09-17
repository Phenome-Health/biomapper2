"""Unit 6: a structure-free lipid check gives species-level lipid nodes a real corroborated /
contradicted verdict even when the committed KG node carries no InChIKey.

Two axes are exercised here, both kept off the InChIKey-block path:
  * the STRUCTURE-FREE composition check, which parses the committed node's NAME and the query with
    Goslin (upstream) and hands ``issue()`` a pre-computed verdict, graded on the parallel
    ``lipid_resolution_level`` axis so it never touches the block comparison; and
  * the SET-BASED check, where LIPID MAPS returns several candidate structures and an InChIKey-bearing
    node corroborates when the sets intersect.

The regression test pins the additive contract: with Tier B off (the default), a non-lipid row and an
InChIKey-bearing row are byte-identical to the pre-Unit-6 certificate.
"""

from __future__ import annotations

from biomapper2.core.certificate import (
    COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION,
    CertificateState,
    LipidResolutionLevel,
    LipidStructureEvidence,
    StructureStatus,
    TierBOutcome,
    TierBResult,
    compare_lipid_composition,
    issue,
)

NODE = "RM:0010728"
NO_KEY = {"HMDB": ["HMDB0010728"]}  # a committed lipid node the graph lists no InChIKey for
WITH_KEY = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]}


def _lipid_issue(evidence: LipidStructureEvidence | None, **overrides):
    kwargs = dict(
        chosen_kg_id=NODE,
        is_small_molecule=True,
        kg_equivalent_ids=NO_KEY,
        equivalent_ids_lookup_ok=True,
        selection_conflict=None,
        lipid_structure=evidence,
        tier_b_enabled=True,
    )
    kwargs.update(overrides)
    return issue(**kwargs)  # pyright: ignore[reportArgumentType]


def test_same_class_and_composition_corroborates_at_lipid_species() -> None:
    evidence = compare_lipid_composition(
        query_class="PC", query_species="PC 34:1", node_class="PC", node_species="PC 34:1"
    )
    assert evidence.level is LipidResolutionLevel.LIPID_SPECIES
    assert evidence.mapping_relation == "exact"
    cert = _lipid_issue(evidence)
    assert cert.state is CertificateState.CORROBORATED
    assert cert.structure_status is StructureStatus.STRUCTURE_ABSENT  # graph still asserts no InChIKey
    assert cert.lipid_resolution_level is LipidResolutionLevel.LIPID_SPECIES
    assert cert.comparison_rule == COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION


def test_different_composition_contradicts_without_an_inchikey_block() -> None:
    evidence = compare_lipid_composition(
        query_class="PC", query_species="PC 34:1", node_class="PC", node_species="PC 36:2"
    )
    assert evidence.level is LipidResolutionLevel.CONTRADICTED
    cert = _lipid_issue(evidence)
    # The one deliberate exception to L21: a Goslin composition mismatch is a positive disagreement, so
    # it contradicts a node with no InChIKey block, tagged with its own comparison rule.
    assert cert.state is CertificateState.CONTRADICTED
    assert cert.node_inchikey_blocks == []
    assert cert.comparison_rule == COMPARISON_RULE_GOSLIN_LEVEL_COMPOSITION


def test_a_node_coarser_than_the_query_corroborates_broad() -> None:
    evidence = compare_lipid_composition(query_class="PC", query_species="PC 34:1", node_class="PC", node_species=None)
    assert evidence.level is LipidResolutionLevel.LIPID_SPECIES
    assert evidence.mapping_relation == "broad"
    cert = _lipid_issue(evidence)
    assert cert.state is CertificateState.CORROBORATED
    assert cert.provenance["lipid_composition_relation"] == "broad"


def test_a_different_class_contradicts() -> None:
    evidence = compare_lipid_composition(
        query_class="PC", query_species="PC 34:1", node_class="PE", node_species="PE 34:1"
    )
    assert evidence.level is LipidResolutionLevel.CONTRADICTED


def test_an_intersecting_candidate_set_corroborates_an_inchikey_node() -> None:
    tier_b = TierBResult(
        source="lipidmaps",
        inchikey_block=None,
        outcome=TierBOutcome.AMBIGUOUS,
        candidate_inchikeys=("BSYNRYMUTXBXSQ-AAAAAAAAAA-N", "QQQQQQQQQQQQQQ-BBBBBBBBBB-N"),
    )
    cert = issue(
        chosen_kg_id=NODE,
        is_small_molecule=True,
        kg_equivalent_ids=WITH_KEY,
        equivalent_ids_lookup_ok=True,
        tier_b=tier_b,
        tier_b_enabled=True,
    )
    assert cert.state is CertificateState.CORROBORATED
    assert cert.provenance["candidate_structure_count"] == 2


def test_out_of_scope_is_distinct_from_off() -> None:
    # Tier B enabled, but nothing was looked up for this row (no tier_b result passed).
    on = issue(
        chosen_kg_id="HGNC:1",
        is_small_molecule=False,
        kg_equivalent_ids={"HGNC": ["1"]},
        equivalent_ids_lookup_ok=True,
        tier_b_enabled=True,
    )
    off = issue(
        chosen_kg_id="HGNC:1",
        is_small_molecule=False,
        kg_equivalent_ids={"HGNC": ["1"]},
        equivalent_ids_lookup_ok=True,
        tier_b_enabled=False,
    )
    assert on.tier_b_outcome is TierBOutcome.OUT_OF_SCOPE
    assert on.provenance["tier_b_enabled"] is True
    assert off.tier_b_outcome is TierBOutcome.OFF
    assert off.provenance["tier_b_enabled"] is False


def test_non_lipid_and_inchikey_rows_are_unchanged_with_tier_b_off() -> None:
    # No lipid_structure, no tier_b_enabled override: the pre-Unit-6 baseline, byte-identical.
    inchikey_row = issue(
        chosen_kg_id=NODE, is_small_molecule=True, kg_equivalent_ids=WITH_KEY, equivalent_ids_lookup_ok=True
    )
    gene_row = issue(
        chosen_kg_id="HGNC:1",
        is_small_molecule=False,
        kg_equivalent_ids={"HGNC": ["1"]},
        equivalent_ids_lookup_ok=True,
    )
    assert inchikey_row.state is CertificateState.UNCORROBORATED
    assert inchikey_row.tier_b_outcome is TierBOutcome.OFF
    assert inchikey_row.lipid_resolution_level is LipidResolutionLevel.UNAVAILABLE
    assert inchikey_row.provenance["tier_b_enabled"] is False
    assert gene_row.state is CertificateState.NOT_APPLICABLE
    assert gene_row.tier_b_outcome is TierBOutcome.OFF


def test_serializers_surface_the_lipid_resolution_level() -> None:
    evidence = compare_lipid_composition(
        query_class="PC", query_species="PC 34:1", node_class="PC", node_species="PC 34:1"
    )
    cert = _lipid_issue(evidence)
    assert cert.to_api_dict()["lipid_resolution_level"] == "lipid_species"
    assert cert.to_flat_columns()["certificate_lipid_resolution_level"] == "lipid_species"
