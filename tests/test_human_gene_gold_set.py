"""Live merge-gate validation: human genes/proteins must resolve into the correct human-gene clique.

Marked `integration` (hits the live Kestrel API). Runs the full pipeline via Mapper.map_entity_to_kg
with the default-on prefer_human behavior, asserts each gold entity resolves to a KG node whose
equivalent-id clique contains the expected human NCBIGene **and** an HGNC marker, and persists a
timestamped report (per the experiment-artifact SOP).

Two classes of positive case:
- Search-recoverable (TNFRSF1A/TNFRSF1B/LDLR): the human node is in the candidate set and `prefer_human`
  re-ranking selects it; `chosen_kg_id` is the exact NCBIGene. The TNFRSF1A/TNFRSF1B pair exercises the
  paralog guard.
- Drug-conflated (GH1/CALCA/POMC/CRH/CTLA4/GBA1): the human node is unreachable by any Kestrel search
  (conflated into a drug node), so it resolves only via the curated gene-symbol fallback bridge. The
  bridge assigns the right NCBIGene, but the KG's canonicalize step then collapses it into its
  over-merged clique whose *representative* is a drug/chemical node (CALCA -> CHEBI:3306 "calcitonin",
  GH1 -> UNII:NQX9KB6PCL "SOMATROPIN"). So `chosen_kg_id == NCBIGene:X` is unsatisfiable for these — the
  achievable, biologically-correct assertion is that the chosen node's clique *contains* the expected
  NCBIGene (verified live 2026-06-18: those clique nodes carry both the NCBIGene and an HGNC equivalent).
"""

import json
from datetime import datetime, timezone

import pytest

from biomapper2.config import PROJECT_ROOT

# All gold entities must resolve to the expected human NCBIGene carrying an HGNC marker.
POSITIVE_GOLD = {
    # Search-recoverable (prefer_human re-ranking); includes a paralog pair (R7 guard).
    "TNFRSF1A": "NCBIGene:7132",
    "TNFRSF1B": "NCBIGene:7133",
    "LDLR": "NCBIGene:3949",
    # Drug-conflated: unreachable by search, resolved via the curated symbol fallback bridge.
    "GH1": "NCBIGene:2688",
    "CALCA": "NCBIGene:796",
    "POMC": "NCBIGene:5443",
    "CRH": "NCBIGene:1392",
    "CTLA4": "NCBIGene:1493",
    "GBA1": "NCBIGene:2629",
}


# Slug of the annotator that owns the symbol-fallback bridge (see core/annotators/kestrel_hybrid.py).
HYBRID_SLUG = "kestrel-hybrid-search"


def _has_hgnc(result: dict) -> bool:
    """True if the resolved node carries an HGNC marker in its equivalent ids."""
    return "HGNC" in json.dumps(result.get("kg_equivalent_ids", {}))


def _clique_contains(result: dict, expected_curie: str) -> bool:
    """True if the chosen node's equivalent-id clique contains ``expected_curie``.

    ``kg_equivalent_ids`` is ``{prefix: [local_id, ...]}`` for the chosen node (see Linker.get_equivalent_ids).
    For a search-recoverable gene the chosen node *is* the expected NCBIGene (and lists itself); for a
    drug-conflated gene the chosen node is the clique's drug/chemical representative, which still carries
    the expected NCBIGene among its equivalents. Either way this membership check holds.
    """
    prefix, local_id = expected_curie.split(":", 1)
    return local_id in (result.get("kg_equivalent_ids", {}).get(prefix, []) or [])


def _resolved_via(result: dict) -> str | None:
    """Provenance marker for an assigned id, read from the preserved per-annotator ``assigned_ids``.

    The symbol-fallback bridge tags its assigned id with ``resolved_via='symbol_fallback'``. That
    marker is retained on the entity's ``assigned_ids`` (and surfaced in the API response) even though
    the normalized curie/resolution layers do not carry it — so the gold-set can measure *actual*
    bridge usage rather than inferring it from a chosen-id mismatch.
    """
    assigned = result.get("assigned_ids", {}) or {}
    for vocab_map in (assigned.get(HYBRID_SLUG, {}) or {}).values():
        for meta in (vocab_map or {}).values():
            if isinstance(meta, dict) and meta.get("resolved_via"):
                return str(meta["resolved_via"])
    return None


def _map_gene(shared_mapper, name: str) -> dict:
    return shared_mapper.map_entity_to_kg(
        item={"name": name},
        name_field="name",
        provided_id_fields=[],
        entity_type="gene",
    )


@pytest.fixture(scope="module")
def gold_set_run(shared_mapper):
    """Run the full gold set live once, persist a timestamped report, and return the per-entity results."""
    rows = []
    for name, expected in POSITIVE_GOLD.items():
        result = _map_gene(shared_mapper, name)
        chosen = result.get("chosen_kg_id")
        rows.append(
            {
                "name": name,
                "expected_human": expected,
                "chosen_kg_id": chosen,
                "is_exact": chosen == expected,
                # The achievable assertion: resolution landed in the expected gene's clique (exact for
                # search-recoverable genes; the drug-canonical representative for conflated ones).
                "resolved_to_clique": (chosen == expected) or _clique_contains(result, expected),
                "has_hgnc": _has_hgnc(result),
                "resolved_via": _resolved_via(result),
                "is_positive": name in POSITIVE_GOLD,
            }
        )

    positives = [r for r in rows if r["is_positive"]]
    # Fraction of entities resolved through the curated symbol-fallback bridge (read from the
    # preserved provenance marker, not inferred from a chosen-id mismatch) — the real R8 signal.
    via_bridge = [r for r in rows if r["resolved_via"] == "symbol_fallback"]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_human": True,
        "n_total": len(rows),
        "n_positive": len(positives),
        "n_resolved_to_clique": sum(r["resolved_to_clique"] for r in positives),
        "n_resolved_via_bridge": len(via_bridge),
        "fallback_fraction": round(len(via_bridge) / len(rows), 3),
        "rows": rows,
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"hgnc_gold_set_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[gold-set] report saved to {out_path} (fallback_fraction={report['fallback_fraction']})")

    return report


@pytest.mark.integration
@pytest.mark.requires_api
class TestHumanGeneGoldSet:
    @pytest.mark.parametrize("name", sorted(POSITIVE_GOLD))
    def test_positive_resolves_into_human_clique(self, gold_set_run, name):
        """Each positive gold entity resolves into the expected gene's clique, carrying an HGNC marker."""
        row = next(r for r in gold_set_run["rows"] if r["name"] == name)
        assert row[
            "resolved_to_clique"
        ], f"{name} -> {row['chosen_kg_id']} whose clique does not contain {POSITIVE_GOLD[name]}"
        assert row["has_hgnc"], f"{name} resolved node lacks an HGNC marker"

    def test_drug_conflated_resolves_into_clique(self, gold_set_run):
        """The drug-conflated genes resolve into the expected human-gene clique.

        Hard gate (correctness): each lands in the expected gene's clique (the KG canonicalizes these
        into a drug/chemical representative, so `chosen_kg_id` is e.g. CHEBI/UNII, not the NCBIGene) and
        carries an HGNC marker.

        The resolution *mechanism* is deliberately NOT asserted live. Two mechanisms both count as
        correct and which one fires depends on live Kestrel recall (a run-to-run variable):
          * the curated symbol->HGNC bridge (`resolved_via == "symbol_fallback"`), and
          * a direct search hit when the conflated node itself surfaces in the gene window — e.g.
            `CRH` ranks in via CHEBI:65307 carrying 'CRH' as a synonym, so `_select_result` picks it
            directly with `resolved_via = None`. That is a correct outcome, not a regression.
        The bridge mechanism is hard-verified deterministically by the offline unit tests
        (test_kestrel_hybrid_fallback.py, test_gene_symbol_resolver.py); this live gate asserts only
        the end-to-end clique resolution.
        """
        for name in ("GH1", "CALCA", "POMC", "CRH", "CTLA4", "GBA1"):
            row = next(r for r in gold_set_run["rows"] if r["name"] == name)
            assert row[
                "resolved_to_clique"
            ], f"{name} -> {row['chosen_kg_id']} whose clique does not contain {POSITIVE_GOLD[name]}"
            assert row["has_hgnc"], f"{name} resolved node lacks an HGNC marker"
            # Accept either a correct mechanism: the curated bridge, or a direct search hit
            # (resolved_via is None) when the conflated node surfaces directly. A *wrong* non-None
            # mechanism would still fail the clique/HGNC gates above.
            assert row["resolved_via"] in (
                None,
                "symbol_fallback",
            ), f"{name} resolved into the clique via an unexpected mechanism: {row['resolved_via']!r}"

    def test_resolution_and_bridge_usage_reported(self, gold_set_run):
        """The run quantifies clique resolution and how often the bridge fired (R8 observability)."""
        assert 0.0 <= gold_set_run["fallback_fraction"] <= 1.0
        # Hard gate: every gold gene resolves into its expected human-gene clique.
        assert gold_set_run["n_resolved_to_clique"] == gold_set_run["n_positive"]
        # Bridge usage is bounded by the curated set, not hard-pinned to 6: upstream de-conflation (the
        # desired fix) reduces it as genes become search-recoverable — that is success, not a regression.
        assert 0 <= gold_set_run["n_resolved_via_bridge"] <= 6
