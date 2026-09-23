"""Live merge-gate: metabolite/disease entities must resolve to the canonical-namespace node.

Marked `integration` (hits the live Kestrel API). Runs the full pipeline via Mapper.map_entity_to_kg with
the default-on prefer_canonical behavior, asserts each gold entity resolves to a canonical-namespace node
(CHEBI/HMDB/RM for metabolites, MONDO for disease) — not a same-text non-canonical node (UMLS/ICD/KEGG) —
and persists a timestamped report (per the experiment-artifact SOP). The report includes a
canonical-flip efficacy column so a silent no-op (canonical present but not selected) is observable.

Expected nodes are taken from a live probe 2026-06-18; the gate asserts the namespace prefix (robust to
node-id churn) and records the exact chosen CURIE. A gene case is included as a regression guard: the
canonical re-rank must not touch gene/protein resolution (that path uses prefer_human).
"""

import json
from datetime import datetime, timezone

import pytest

from biomapper2.config import PROJECT_ROOT

# Each gold entity must resolve to a node whose namespace is in the expected canonical set.
METABOLITE_GOLD = {
    "kynurenine": {"CHEBI", "HMDB", "RM"},
    "serotonin": {"CHEBI", "HMDB", "RM"},
}
DISEASE_GOLD = {
    "Parkinson disease": {"MONDO"},
    "chronic myeloid leukemia": {"MONDO"},
}
# Gene regression guard: must still resolve to a HUMAN GENE node via the prefer_human path (unaffected
# by the canonical re-rank). The specific committed equivalent id (NCBIGene vs HGNC) is NOT deterministic
# across live KG runs — the guard's intent is "still a human gene node", not a fixed namespace — so accept
# either. See issue ``#118`` (flaky live gene-namespace resolution).
GENE_REGRESSION = {"TNFRSF1A": {"NCBIGene", "HGNC"}}


def _prefix(curie):
    return str(curie).split(":", 1)[0] if curie else None


def _map(shared_mapper, name, entity_type):
    return shared_mapper.map_entity_to_kg(
        item={"name": name}, name_field="name", provided_id_fields=[], entity_type=entity_type
    )


@pytest.fixture(scope="module")
def gold_set_run(shared_mapper):
    """Run the gold set live once, persist a timestamped report, return per-entity results."""
    rows = []
    for name, expected in {**METABOLITE_GOLD, **DISEASE_GOLD}.items():
        etype = "metabolite" if name in METABOLITE_GOLD else "disease"
        result = _map(shared_mapper, name, etype)
        chosen = result.get("chosen_kg_id")
        rows.append(
            {
                "name": name,
                "entity_type": etype,
                "expected_prefixes": sorted(expected),
                "chosen_kg_id": chosen,
                "chosen_prefix": _prefix(chosen),
                "is_canonical": _prefix(chosen) in expected,
            }
        )

    n = len(rows)
    n_canonical = sum(r["is_canonical"] for r in rows)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prefer_canonical": True,
        "n_total": n,
        "n_resolved_canonical": n_canonical,
        # Efficacy: fraction resolved to a canonical node. A low value on entities known to have a
        # canonical node signals the selector is silently no-op'ing (the failure mode the probe guards).
        "canonical_fraction": round(n_canonical / n, 3) if n else 0.0,
        "rows": rows,
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"canonical_namespace_gold_set_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[canonical-gold-set] saved {out_path} (canonical_fraction={report['canonical_fraction']})")
    return report


@pytest.mark.integration
@pytest.mark.requires_api
class TestCanonicalNamespaceGoldSet:
    @pytest.mark.parametrize("name", sorted({**METABOLITE_GOLD, **DISEASE_GOLD}))
    def test_resolves_to_canonical_namespace(self, gold_set_run, name):
        """Each gold entity resolves to a node in the expected canonical namespace set."""
        row = next(r for r in gold_set_run["rows"] if r["name"] == name)
        expected = {**METABOLITE_GOLD, **DISEASE_GOLD}[name]
        assert row["chosen_prefix"] in expected, f"{name} -> {row['chosen_kg_id']} (expected one of {expected})"

    def test_canonical_fraction_reported(self, gold_set_run):
        """Every gold entity flips to canonical (efficacy metric persisted; guards the silent no-op)."""
        assert 0.0 <= gold_set_run["canonical_fraction"] <= 1.0
        assert gold_set_run["n_resolved_canonical"] == gold_set_run["n_total"]

    def test_gene_regression_unaffected(self, shared_mapper):
        """The canonical re-rank must not touch gene/protein resolution (prefer_human path)."""
        result = _map(shared_mapper, "TNFRSF1A", "gene")
        expected = GENE_REGRESSION["TNFRSF1A"]
        assert (
            _prefix(result.get("chosen_kg_id")) in expected
        ), f"TNFRSF1A -> {result.get('chosen_kg_id')} (expected a human gene node in {sorted(expected)})"
