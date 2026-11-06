"""Rebuild ``primary_curie_v121_rows.json`` from the recorded v1.2.1 prod benchmark caches.

Not run by the test suite (it reads run directories outside the repo). Kept so the fixture is
reproducible. Each recorded client row carries ``primary_curie`` (= the API's ``curies[0]``),
``chosen_kg_id``, ``kg_equivalent_ids`` and the flattened annotator ``identifiers``. The API's sorted
``curies`` list is rebuilt by normalizing those identifiers with this repo's ``Normalizer``; the build
asserts that ``sorted(curies)[0]`` reproduces the recorded ``primary_curie`` on EVERY row, so the
rebuilt list is the list the API actually sorted. ``kg_ids`` was not recorded, so the fixture omits it
(the rule then uses ``{chosen_kg_id}`` + ``kg_equivalent_ids``, a subset of what the API sees).
``kg_equivalent_ids`` is trimmed to the prefixes that occur in ``curies``, which cannot change any
membership test the rule performs, and identical recorded results are stored once with a ``panels`` list.

Usage: uv run python tests/fixtures/build_primary_curie_fixture.py
"""

import json
import logging
from pathlib import Path

from biomapper2.core.normalizer.normalizer import Normalizer

RUNS = Path.home() / "external_benchmark_runs"
CROSS = "cross_cohort_1ee319e_bm156_20261105T142103Z"
UKBB = "ukbb_arivale_metabolites_1ee319e_bm156_20261106T014817Z"
OUT = Path(__file__).parent / "primary_curie_v121_rows.json"


def _records():
    for panel in ("necs", "arivale", "xuetal"):
        for line in open(RUNS / CROSS / "_cache" / panel / "entity_cache.jsonl"):
            obj = json.loads(line)
            if obj.get("type") == "batch":
                for entry in obj["entries"]:
                    yield panel, entry["result"]
    for line in open(RUNS / UKBB / "_cache" / "mapping_cache.jsonl"):
        yield "ukbb_arivale", json.loads(line)["v"]


def main() -> None:
    logging.disable(logging.WARNING)
    normalizer = Normalizer()
    rows: dict[str, dict] = {}
    for panel, rec in _records():
        curies = sorted(normalizer.get_curies(rec.get("identifiers") or {}, log_warnings=False)[0])
        assert (curies[0] if curies else None) == rec.get("primary_curie"), rec["query_name"]
        prefixes = {c.split(":", 1)[0] for c in curies}
        equiv = {p: ids for p, ids in (rec.get("kg_equivalent_ids") or {}).items() if p in prefixes}
        row = {
            "name": rec["query_name"],
            "chosen_kg_id": rec.get("chosen_kg_id"),
            "chosen_kg_id_review": rec.get("chosen_kg_id_review") or None,
            "recorded_primary_curie": rec.get("primary_curie"),
            "curies": curies,
            "kg_equivalent_ids": equiv,
        }
        # One fixture row per distinct recorded result; ``panels`` lists every panel it occurred in (with
        # repeats), so per-panel impact counts are still recoverable.
        key = json.dumps(row, sort_keys=True)
        rows.setdefault(key, {**row, "panels": []})["panels"].append(panel)
    payload = {
        "_provenance": {
            "engine": "prod engine 1ee319e / e9601a3 (v1.2.1 release commit, version-only diff)",
            "client": "biomapper 1.5.6",
            "runs": [CROSS + " (necs, arivale, xuetal)", UKBB + " (ukbb_arivale mapping cache)"],
            "kg": "KRAKEN kg_version 2.3.0, git 3dd08a5b7b17e320ceb0c0091e09b781d4d90242",
            "curies": "rebuilt by Normalizer.get_curies over the recorded identifiers; sorted(curies)[0] "
            "== recorded_primary_curie asserted on every row",
        },
        "rows": list(rows.values()),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"wrote {len(rows)} distinct rows ({sum(len(r['panels']) for r in rows.values())} recorded) to {OUT}")


if __name__ == "__main__":
    main()
