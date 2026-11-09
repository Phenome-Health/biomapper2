"""ICD-10-CM twin control gate: may a missed ``ICD10CM:X`` be committed through ``ICD10:X`` as EXACT?

Why this exists
---------------
Since v1.3.0 a provided ``ICD10CM:<code>`` keeps its prefix, and KRAKEN 2.3.0 has no ``ICD10CM`` nodes,
so codes that used to resolve through the accidental ``ICD10:`` rewrite now come back unresolved. The
Phase 1a fix retries the same code as ``ICD10:<code>`` (the "twin"). Whether that hit may be labelled
``mapping_relation = exact`` or only ``broad`` is a factual question about KRAKEN's ``ICD10:`` cliques,
and this study answers it with a committed artifact instead of an assumption.

What it measures (decided at Checkpoint 2, 2026-11-08)
------------------------------------------------------
* **Gate: title agreement.** For every CDC FY code whose twin canonicalizes, compare the CDC long title
  with the KRAKEN node's name + synonyms (deterministic token Dice score, fixed cutoff recorded in the
  artifact). The gate passes at >= ``GATE_THRESHOLD`` agreement on rows with a label, with n >=
  ``GATE_MIN_N``; fewer rows reports ``insufficient_n``. CDC titles and KRAKEN labels are produced by
  different organisations, so this is independent evidence about the twin node.
* **Consistency (reported, NOT gating).** For Mondo SSSOM ``skos:exactMatch`` ICD10CM rows (range targets
  excluded), does ``canonicalize(ICD10:X)`` land in ``canonicalize(MONDO:m)``'s clique? Babel builds its
  disease cliques partly FROM Mondo xrefs, so agreement here is Babel-Mondo consistency, not evidence.
* **Provenance probe.** For a seeded sample of agreeing twin codes, record which ids sit in the twin's
  clique and which sources provided the node, so a reader can see how circular the Mondo check is.
* **Regression list.** Every CDC code's twin hit (code -> node) is the Phase 1a fixture: "twin reproduces
  current ``ICD10:`` resolution on the pinned KG build".
* **Denylist.** Codes that disagree on EITHER check are written to the runtime denylist; the mapper
  emits their twin hits as ``broad`` whatever the gate says.
* **Hand-review sheet.** A seeded random 50 of the title mismatches, for Trent to adjudicate. This
  script never adjudicates: ``gate.reviewed`` is False until a human records the review, and the
  shipped twin relation stays ``broad`` until then.

Run
---
    uv run python -m studies.analysis.icd10cm_twin_control \
        --cdc-zip /path/icd10cm-code-descriptions-<FY>.zip --sssom /path/mondo.sssom.tsv \
        [--v121-project /path/to/v1.2.1/checkout]

Results are ALWAYS written (SOP: expensive runs persist by default) to a timestamped
``results/icd10cm_twin_control_<UTC>.json`` plus the stable ``results/icd10cm_twin_control.json`` copy.
A Kestrel failure aborts the run before anything is written, so a partial artifact can never read as
complete.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).parent / "results"
ARTIFACT_NAME = "icd10cm_twin_control.json"
REVIEW_SHEET_NAME = "icd10cm_twin_title_review.tsv"
DENYLIST_PATH = REPO_ROOT / "src" / "biomapper2" / "data" / "icd10cm" / "twin_denylist.json"
REGRESSION_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "icd10cm_twin_regression.json"

CDC_ORDER_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/icd10cm-code-descriptions-2027.zip"
)
CDC_ORDER_MEMBER = "icd10cm-order-2027.txt"
MONDO_SSSOM_URL = "http://purl.obolibrary.org/obo/mondo/mappings/mondo.sssom.tsv"

# Gate parameters (Trent, Checkpoint 2, 2026-11-08).
GATE_THRESHOLD = 0.95
GATE_MIN_N = 200

# Title comparison. Token-set Sorensen-Dice, max over the node's name and synonyms.
TITLE_MATCH_CUTOFF = 0.6
REVIEW_SAMPLE_K = 50
REVIEW_SEED = 20261108
PROVENANCE_SAMPLE_K = 100

# Success-criteria codes from the requirements doc; each gets a route label.
SUCCESS_CRITERIA_CODES = ("E75.21", "E83.01", "F40.0", "I10", "E11", "E11.65", "F32.9")
V121_SAMPLE_K = 200

TITLE_AGREE = "agree"
TITLE_DISAGREE = "disagree"
TITLE_NO_LABEL = "no_label"

GATE_PASS = "pass"
GATE_FAIL = "fail"
GATE_INSUFFICIENT_N = "insufficient_n"

RELATION_EXACT = "exact"
RELATION_BROAD = "broad"

# Boilerplate that CDC titles carry and graph labels do not ("..., unspecified", "NOS", "NEC").
_BOILERPLATE = frozenset({"unspecified", "nos", "nec", "other", "specified", "elsewhere", "classified"})
# Function words. ``with`` / ``without`` are deliberately KEPT: they carry meaning in ICD titles.
_STOPWORDS = frozenset(
    {"of", "the", "and", "or", "in", "a", "an", "to", "due", "by", "for", "on", "at", "as", "disorder"}
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ------------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderRow:
    """One row of the CDC order file: a dotted code, whether it is billable (vs a header), its title."""

    code: str
    billable: bool
    title: str


@dataclass(frozen=True)
class SssomRow:
    mondo: str
    code: str
    is_range: bool


def redot(code: str) -> str:
    """CDC files carry undotted codes (``E1165``); KRAKEN and users write them dotted (``E11.65``)."""
    code = code.strip()
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def parse_order_file(text: str) -> list[OrderRow]:
    """Parse the fixed-width CDC order file (see icd10OrderFiles.pdf).

    Columns: order number (1-5), code (7-13), header/billable flag (15), short title (17-76), long
    title (78-). The long title is used: the short one is truncated by CDC.
    """
    rows: list[OrderRow] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        code = line[6:13].strip()
        flag = line[14:15]
        if not code or flag not in {"0", "1"}:
            raise ValueError(f"malformed CDC order row: {line!r}")
        long_title = line[77:].strip() or line[16:76].strip()
        rows.append(OrderRow(code=redot(code), billable=flag == "1", title=long_title))
    return rows


def parse_sssom_icd10cm_exact(text: str) -> list[SssomRow]:
    """The Mondo SSSOM rows that assert ``MONDO skos:exactMatch ICD10CM:<code>``, in file order."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    rows: list[SssomRow] = []
    for rec in reader:
        obj = rec.get("object_id") or ""
        if rec.get("predicate_id") != "skos:exactMatch" or not obj.startswith("ICD10CM:"):
            continue
        code = obj.split(":", 1)[1]
        rows.append(SssomRow(mondo=rec["subject_id"], code=code, is_range="-" in code))
    return rows


# ------------------------------------------------------------------------------------------------
# Title agreement (the gate)
# ------------------------------------------------------------------------------------------------
def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def title_tokens(text: str) -> frozenset[str]:
    """Lower-case content tokens with CDC boilerplate and function words removed, lightly stemmed."""
    tokens = _NON_ALNUM.split(text.lower())
    return frozenset(_stem(t) for t in tokens if t and t not in _BOILERPLATE and t not in _STOPWORDS)


def _dice(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def title_verdict(cdc_title: str, labels: Iterable[str | None]) -> tuple[str, float | None]:
    """``(agree|disagree|no_label, best score)`` for a CDC title against a node's name + synonyms."""
    usable = [lbl for lbl in labels if lbl and str(lbl).strip()]
    if not usable:
        return TITLE_NO_LABEL, None
    cdc = title_tokens(cdc_title)
    best = max(_dice(cdc, title_tokens(str(lbl))) for lbl in usable)
    return (TITLE_AGREE if best >= TITLE_MATCH_CUTOFF else TITLE_DISAGREE), best


def gate_decision(agree: int, disagree: int, threshold: float = GATE_THRESHOLD, min_n: int = GATE_MIN_N) -> dict:
    """The gate verdict on rows with a label. ``insufficient_n`` below ``min_n``, never a pass or fail."""
    n = agree + disagree
    rate = agree / n if n else None
    if n < min_n:
        status = GATE_INSUFFICIENT_N
    elif rate is not None and rate >= threshold:
        status = GATE_PASS
    else:
        status = GATE_FAIL
    return {
        "status": status,
        "n": n,
        "agree": agree,
        "disagree": disagree,
        "rate": rate,
        "threshold": threshold,
        "min_n": min_n,
    }


def twin_relation_for(gate: dict, reviewed: bool) -> str:
    """The relation the twin may ship with: ``exact`` only on a human-REVIEWED pass, ``broad`` otherwise."""
    return RELATION_EXACT if reviewed and gate.get("status") == GATE_PASS else RELATION_BROAD


# ------------------------------------------------------------------------------------------------
# Consistency (reported, not gating)
# ------------------------------------------------------------------------------------------------
def consistency_check(rows: list[SssomRow], canonical: dict[str, str | None]) -> dict:
    """Cluster-level comparison: ``canonicalize(ICD10:X) == canonicalize(MONDO:m)`` per SSSOM row."""
    counts = {"consistent": 0, "inconsistent": 0, "twin_null": 0, "range_excluded": 0}
    out_rows: list[dict] = []
    inconsistent: set[str] = set()
    for row in rows:
        if row.is_range:
            counts["range_excluded"] += 1
            continue
        twin_node = canonical.get(f"ICD10:{row.code}")
        mondo_node = canonical.get(row.mondo) or row.mondo
        if twin_node is None:
            verdict = "twin_null"
        elif twin_node == mondo_node:
            verdict = "consistent"
        else:
            verdict = "inconsistent"
            inconsistent.add(row.code)
        counts[verdict] += 1
        out_rows.append(
            {"mondo": row.mondo, "code": row.code, "twin_node": twin_node, "mondo_node": mondo_node, "verdict": verdict}
        )
    non_null = counts["consistent"] + counts["inconsistent"]
    return {
        "counts": counts,
        "rate_on_non_null": counts["consistent"] / non_null if non_null else None,
        "inconsistent_codes": sorted(inconsistent),
        "rows": out_rows,
    }


# ------------------------------------------------------------------------------------------------
# Denylist / review sheet
# ------------------------------------------------------------------------------------------------
def build_denylist(title_disagree: Iterable[str], consistency_inconsistent: Iterable[str]) -> list[str]:
    return sorted(set(title_disagree) | set(consistency_inconsistent))


def sample_for_review(mismatches: list[dict], k: int = REVIEW_SAMPLE_K, seed: int = REVIEW_SEED) -> list[dict]:
    """A seeded random ``k`` of the mismatches, in code order (all of them when fewer than ``k``)."""
    ordered = sorted(mismatches, key=lambda m: m["code"])
    if len(ordered) <= k:
        return ordered
    return sorted(random.Random(seed).sample(ordered, k), key=lambda m: m["code"])


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600; artifacts are ordinary tracked files
    os.replace(tmp, path)


def write_denylist(path: Path, codes: Iterable[str], *, source_artifact: str, kg_build: str) -> None:
    payload = {
        "description": "ICD-10-CM codes whose ICD10: twin hit disagrees with the CDC title or Mondo SSSOM; "
        "their twin hits are emitted as mapping_relation=broad. Generated by "
        "studies/analysis/icd10cm_twin_control.py; do not edit by hand.",
        "source_artifact": source_artifact,
        "kg_build": kg_build,
        "codes": sorted(set(codes)),
    }
    _atomic_write(path, json.dumps(payload, indent=2) + "\n")


def write_review_sheet(path: Path, sample: list[dict]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(["code", "cdc_title", "twin_node", "node_name", "best_score", "reviewer_verdict", "reviewer_note"])
    for m in sample:
        writer.writerow(
            [m["code"], m["cdc_title"], m["twin_node"], m.get("node_name") or "", f"{m['score']:.3f}", "", ""]
        )
    _atomic_write(path, buf.getvalue())


# ------------------------------------------------------------------------------------------------
# Orchestration (injected I/O so the logic is testable offline)
# ------------------------------------------------------------------------------------------------
def _node_labels(node: dict | None) -> list[str]:
    if not node:
        return []
    return [node.get("name") or ""] + list(node.get("synonyms") or [])


def run_study(
    *,
    order_rows: list[OrderRow],
    sssom_rows: list[SssomRow],
    canonicalize: Callable[[list[str]], dict[str, str | None]],
    get_nodes: Callable[[list[str]], dict[str, dict]],
    threshold: float = GATE_THRESHOLD,
    min_n: int = GATE_MIN_N,
) -> dict:
    """Compute every result. Raises on any Kestrel failure (nothing is written by this function)."""
    twin_curies = [f"ICD10:{r.code}" for r in order_rows]
    sssom_curies = sorted({f"ICD10:{r.code}" for r in sssom_rows if not r.is_range} | {r.mondo for r in sssom_rows})
    t0 = time.monotonic()
    canonical = canonicalize(sorted(set(twin_curies) | set(sssom_curies)))
    canonicalize_s = time.monotonic() - t0

    hits = {r.code: canonical.get(f"ICD10:{r.code}") for r in order_rows}
    hit_nodes = sorted({n for n in hits.values() if n})
    nodes = get_nodes(hit_nodes) if hit_nodes else {}

    counts = {TITLE_AGREE: 0, TITLE_DISAGREE: 0, TITLE_NO_LABEL: 0}
    mismatches: list[dict] = []
    agreeing: list[str] = []
    for row in order_rows:
        node_id = hits[row.code]
        if not node_id:
            continue
        node = nodes.get(node_id)
        verdict, score = title_verdict(row.title, _node_labels(node))
        counts[verdict] += 1
        if verdict == TITLE_DISAGREE:
            mismatches.append(
                {
                    "code": row.code,
                    "cdc_title": row.title,
                    "twin_node": node_id,
                    "node_name": (node or {}).get("name"),
                    "score": score,
                }
            )
        elif verdict == TITLE_AGREE:
            agreeing.append(row.code)
    gate = gate_decision(counts[TITLE_AGREE], counts[TITLE_DISAGREE], threshold=threshold, min_n=min_n)

    consistency = consistency_check(sssom_rows, canonical)

    # Provenance probe on a seeded sample of agreeing twin codes.
    probe_codes = sorted(random.Random(REVIEW_SEED).sample(sorted(agreeing), min(PROVENANCE_SAMPLE_K, len(agreeing))))
    sssom_by_code = {r.code: r.mondo for r in sssom_rows if not r.is_range}
    probe = []
    for code in probe_codes:
        node = nodes.get(hits[code] or "") or {}
        equiv = list(node.get("equivalent_ids") or [])
        probe.append(
            {
                "code": code,
                "twin_node": hits[code],
                "icd10_in_clique": f"ICD10:{code}" in equiv,
                "icd10cm_in_clique": f"ICD10CM:{code}" in equiv,
                "mondo_ids_in_clique": sorted(e for e in equiv if e.startswith("MONDO:")),
                "sssom_mondo": sssom_by_code.get(code),
                "provided_by": sorted(node.get("provided_by") or []),
            }
        )
    probe_summary = {
        "n": len(probe),
        "icd10_in_clique": sum(p["icd10_in_clique"] for p in probe),
        "icd10cm_in_clique": sum(p["icd10cm_in_clique"] for p in probe),
        "node_is_mondo": sum(1 for p in probe if (p["twin_node"] or "").startswith("MONDO:")),
        "provided_by_babel_normalizer": sum(1 for p in probe if "infores:sri-node-normalizer" in p["provided_by"]),
    }

    regression_codes = {code: {"kg_id": node, "route": "twin"} for code, node in sorted(hits.items()) if node}
    success = {
        code: {"twin_node": hits.get(code, canonical.get(f"ICD10:{code}")), "in_cdc_order_file": code in hits}
        for code in SUCCESS_CRITERIA_CODES
    }
    for code, rec in success.items():
        rec["route"] = "twin" if rec["twin_node"] else "twin_miss"

    denylist = build_denylist([m["code"] for m in mismatches], consistency["inconsistent_codes"])
    review_sample = sample_for_review(mismatches)
    return {
        "title_gate": {
            "counts": counts,
            "cutoff": TITLE_MATCH_CUTOFF,
            "method": "token-set Sorensen-Dice, max over node name+synonyms; boilerplate/stopwords removed",
            "boilerplate": sorted(_BOILERPLATE),
            "stopwords": sorted(_STOPWORDS),
            "mismatches": mismatches,
        },
        "gate": {
            **gate,
            "reviewed": False,
            "review_adjudication": None,
            "twin_relation": twin_relation_for(gate, reviewed=False),
        },
        "mondo_consistency": {**consistency, "role": "Babel-Mondo consistency only; NOT gate evidence"},
        "provenance_probe": {"summary": probe_summary, "rows": probe},
        "regression": {
            "cdc_code_count": len(order_rows),
            "twin_hit_count": len(regression_codes),
            "twin_hit_billable": sum(1 for r in order_rows if r.billable and hits[r.code]),
            "codes": regression_codes,
        },
        "success_criteria_routes": success,
        "denylist": {"count": len(denylist), "codes": denylist},
        "review_sample": {"k": REVIEW_SAMPLE_K, "seed": REVIEW_SEED, "rows": review_sample},
        "timing": {
            "canonicalize_s": round(canonicalize_s, 2),
            "canonicalize_curies": len(set(twin_curies) | set(sssom_curies)),
            "get_nodes_ids": len(hit_nodes),
        },
    }


def run_and_write(
    *,
    order_rows: list[OrderRow],
    sssom_rows: list[SssomRow],
    canonicalize: Callable[[list[str]], dict[str, str | None]],
    get_nodes: Callable[[list[str]], dict[str, dict]],
    inputs: dict,
    kg_build: dict,
    out_dir: Path,
    stable_dir: Path,
    denylist_path: Path,
    review_path: Path,
    regression_fixture_path: Path | None = None,
    v121_crosscheck: dict | None = None,
    min_n: int = GATE_MIN_N,
) -> dict:
    """Run the study, then write the artifact(s). Nothing is written unless the whole run succeeded."""
    result = run_study(
        order_rows=order_rows, sssom_rows=sssom_rows, canonicalize=canonicalize, get_nodes=get_nodes, min_n=min_n
    )
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "study": "icd10cm_twin_control",
        "generated_at": stamp,
        "inputs": inputs,
        "kg_build": kg_build,
        "gate_parameters": {
            "threshold": GATE_THRESHOLD,
            "min_n": min_n,
            "title_cutoff": TITLE_MATCH_CUTOFF,
            "decided": "Trent, Checkpoint 2, 2026-11-08",
        },
        "v121_crosscheck": v121_crosscheck,
        **result,
    }
    text = json.dumps(artifact, indent=2, sort_keys=False) + "\n"
    timestamped = out_dir / f"icd10cm_twin_control_{stamp}.json"
    _atomic_write(timestamped, text)
    _atomic_write(stable_dir / ARTIFACT_NAME, text)
    write_denylist(
        denylist_path,
        result["denylist"]["codes"],
        source_artifact=f"studies/analysis/results/{ARTIFACT_NAME}",
        kg_build=str(kg_build.get("kg_version", "unknown")),
    )
    write_review_sheet(review_path, result["review_sample"]["rows"])
    if regression_fixture_path is not None:
        fixture = {
            "description": "Phase 1a regression fixture: provided ICD10CM:<code> -> the node its ICD10: twin "
            "canonicalizes to on the pinned KG build. Derived from studies/analysis/results/icd10cm_twin_control.json.",
            "kg_build": kg_build,
            "codes": result["regression"]["codes"],
            "success_criteria_routes": result["success_criteria_routes"],
        }
        _atomic_write(regression_fixture_path, json.dumps(fixture, indent=1) + "\n")
    print(f"Artifact written: {timestamped}")
    print(f"Stable copy:      {stable_dir / ARTIFACT_NAME}")
    return artifact


# ------------------------------------------------------------------------------------------------
# Live I/O
# ------------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_canonicalize(curies: list[str]) -> dict[str, str | None]:
    from biomapper2.utils import BisectBudget, kestrel_request

    budget = BisectBudget()
    result = kestrel_request(
        method="POST", endpoint="canonicalize", batch_field="curies", batch_items=curies, batch_size=2000, budget=budget
    )
    missing = [c for c in curies if c not in result]
    if missing:
        # Kestrel answers a miss with an explicit null; an ABSENT key is not a definite null.
        raise RuntimeError(
            f"/canonicalize omitted {len(missing)} curies (e.g. {missing[:5]}); refusing to treat as null"
        )
    return result


def _live_get_nodes(ids: list[str]) -> dict[str, dict]:
    from biomapper2.utils import kestrel_request

    raw = kestrel_request(
        method="POST",
        endpoint="get-nodes",
        batch_field="curies",
        batch_items=ids,
        batch_size=2000,
        json={"slim": False, "truncate_long_fields": False},
    )
    missing = [i for i in ids if not isinstance(raw.get(i), dict)]
    if missing:
        raise RuntimeError(f"/get-nodes returned no record for {len(missing)} nodes (e.g. {missing[:5]})")
    return {i: {k: raw[i].get(k) for k in ("name", "synonyms", "equivalent_ids", "provided_by")} for i in ids}


_V121_SNIPPET = r"""
import json, sys, logging
logging.disable(logging.CRITICAL)
import pandas as pd
from biomapper2.biolink_client import BiolinkClient
from biomapper2.core.normalizer import Normalizer
n = Normalizer(biolink_client=BiolinkClient())
out = {}
for code in json.load(sys.stdin):
    item = pd.Series({"name": "x", "icd10cm": code})
    r = n.normalize(item=item, provided_id_fields=["icd10cm"], array_delimiters=[",", ";"])
    out[code] = list(r["curies"])
print("@@JSON@@" + json.dumps(out))
"""


def v121_crosscheck(project_dir: Path, codes: list[str]) -> dict:
    """Run the v1.2.1 normalizer in its own env; confirm each ICD10CM:X was rewritten to ICD10:<same X>."""
    proc = subprocess.run(
        ["uv", "run", "--project", str(project_dir), "python", "-c", _V121_SNIPPET],
        input=json.dumps(codes),
        capture_output=True,
        text=True,
        check=True,
        cwd=project_dir,
    )
    payload = proc.stdout.split("@@JSON@@", 1)[1]
    curies = json.loads(payload)
    differences = {c: v for c, v in curies.items() if v != [f"ICD10:{c}"]}
    return {
        "n": len(codes),
        "same_code_icd10_rewrite": len(codes) - len(differences),
        "differences": differences,
        "codes": sorted(codes),
    }


def _fetch_health() -> dict:
    from biomapper2.config import get_kestrel_api_url
    from biomapper2.provenance import fetch_kg_build_info

    url = get_kestrel_api_url()
    kestrel_version, info = fetch_kg_build_info(url)
    return {"kestrel_url": url, "kestrel_version": kestrel_version, **info.model_dump()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cdc-zip", type=Path, required=True, help=f"CDC code-descriptions zip ({CDC_ORDER_URL})")
    parser.add_argument("--sssom", type=Path, required=True, help=f"Mondo SSSOM TSV ({MONDO_SSSOM_URL})")
    parser.add_argument("--v121-project", type=Path, default=None, help="checkout of v1.2.1 with a synced env")
    parser.add_argument("--sample", type=int, default=None, help="timing sample: only the first N CDC codes")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="override the results dir")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    with zipfile.ZipFile(args.cdc_zip) as zf:
        member = next(n for n in zf.namelist() if n.endswith(CDC_ORDER_MEMBER))
        order_text = zf.read(member).decode("utf-8")
    order_rows = parse_order_file(order_text)
    sssom_rows = parse_sssom_icd10cm_exact(args.sssom.read_text(encoding="utf-8"))
    if args.sample:
        order_rows = order_rows[: args.sample]
        sssom_rows = []

    kg_build = _fetch_health()
    inputs = {
        "cdc_order_zip": {
            "url": CDC_ORDER_URL,
            "member": member,
            "sha256": _sha256(args.cdc_zip),
            "rows": len(order_rows),
        },
        "mondo_sssom": {"url": MONDO_SSSOM_URL, "sha256": _sha256(args.sssom), "icd10cm_exact_rows": len(sssom_rows)},
    }
    crosscheck = None
    if args.v121_project is not None:
        codes = [r.code for r in order_rows]
        sample = sorted(
            set(random.Random(REVIEW_SEED).sample(codes, min(V121_SAMPLE_K, len(codes)))) | set(SUCCESS_CRITERIA_CODES)
        )
        crosscheck = v121_crosscheck(args.v121_project, sample)

    sample_mode = bool(args.sample)
    out_dir = args.out_dir if not sample_mode else Path(tempfile.mkdtemp(prefix="icd10cm_twin_sample_"))
    run_and_write(
        order_rows=order_rows,
        sssom_rows=sssom_rows,
        canonicalize=_live_canonicalize,
        get_nodes=_live_get_nodes,
        inputs=inputs,
        kg_build=kg_build,
        out_dir=out_dir,
        stable_dir=out_dir,
        denylist_path=DENYLIST_PATH if not sample_mode else out_dir / "deny.json",
        review_path=out_dir / REVIEW_SHEET_NAME,
        regression_fixture_path=REGRESSION_FIXTURE_PATH if not sample_mode else None,
        v121_crosscheck=crosscheck,
        min_n=GATE_MIN_N,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
