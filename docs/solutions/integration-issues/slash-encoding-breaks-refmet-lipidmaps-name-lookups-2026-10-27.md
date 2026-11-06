---
title: "Encoding '/' as %2F 404s every slash-bearing name at Metabolomics Workbench and LIPID MAPS (and frozen caches kept the negatives)"
date: 2026-10-27
category: integration-issues
module: annotators/name-path-quoting
problem_type: integration_issue
component: service_object
symptoms:
  - "RefMet benchmark arm strict accuracy fell 87.9% -> 72.7%; slash-bearing names 349/438 -> 67/448 while non-slash stayed ~97%"
  - "All 448/448 slash names reported certificate_refmet_availability=no_match: silent, no error, circuit breaker stayed closed"
  - "Client log shows HTTP 404 on /rest/refmet/name/MG%2018%3A1%2F0%3A0... (Apache rejects an encoded %2F before any lookup)"
  - "After a literal-slash fix, mixture names ('mannitol/sorbitol', 'vitamin D2/D3') resolve to the text before the slash, and an HTTP 200 'This output item does not exist' body trips the Tier B circuit breaker"
  - "Code fix alone changed nothing on frozen/live_backup boxes: the RefMet freeze TSV, the Tier B freeze and the prod SQLite store held stale slash no_match rows"
root_cause: wrong_api
resolution_type: code_fix
severity: high
related_components:
  - testing_framework
  - database
tags:
  - url-encoding
  - path-segment
  - metabolomics-workbench
  - refmet
  - lipid-maps
  - lipid-shorthand
  - frozen-cache
  - silent-no-match
---

# Encoding '/' as %2F 404s every slash-bearing name at Metabolomics Workbench and LIPID MAPS

## Problem

Commit `a4aeef7` (2026-08-24, org PR #84, Goslin lipid-shorthand route) switched the metabolite-name
path segment from `quote(name)` to `quote(name, safe="")` at four Metabolomics Workbench (MW) and
LIPID MAPS call sites. Both servers return a 404 for an encoded `%2F` before doing any lookup, so every
slash-bearing name (sn-position lipid shorthand, ceramides, GalCer, FAHFA) silently lost its RefMet
vote, its MW InChIKey and its LIPID MAPS lookup. Because the 404 was classified as a clean `no_match`,
nothing failed loudly. Prod RefMet coverage dropped about 15 points for two months.

## Symptoms

- RefMet benchmark arm strict accuracy: **87.9%** (suite_20260805, engine `d059564`) to **72.7%**
  (suite_20260927, `e324cac`). Slash-bearing names went 349/438 to **67/448**; non-slash held near 97%.
- **448/448** slash names had `certificate_refmet_availability=no_match`.
- Client log: `GET /rest/refmet/name/MG%2018%3A1%2F0%3A0.../inchi_key -> 404` and
  `/rest/refmet/match/PC%2016%3A0%2F18%3A1 -> 404`.
- LMSD common-name coverage dropped too.
- It was found only when a swing-attribution pass diffed per-row outputs of two suite runs. The
  aggregate looked like ordinary "RefMet coverage drift".

## What Didn't Work

- **The `a4aeef7` assumption.** An unescaped `/` "injects a path segment", so encode it. Nobody probed
  the live servers, which reject `%2F` outright.
  ```python
  # a4aeef7 (same change in tier_b._fetch_mw, structure_resolver._fetch_mw_inchikey,
  # lipidmaps_rest._fetch_json)
  -        url = f"{self.BASE_URL}/{quote(metabolite_name)}"
  +        url = f"{self.BASE_URL}/{quote(metabolite_name, safe='')}"
  ```
- **Treating the 404 as "not found" instead of fixing the URL** (session history). The 2026-09-16
  investigation (`docs/plans/2026-09-16-002-feat-lipid-hierarchy-aware-mapping-plan.md`, finding F1)
  correctly saw that MW 404s on any encoded slash, and that three slash names in a row tripped the
  process-wide RefMet breaker (`@circuit(failure_threshold=3, recovery_timeout=300)`), suppressing
  RefMet for every name for 5 minutes. The follow-up fix (`test_mw_slash_breaker`) classified slash
  404s as `no_match` to keep the breaker closed. That stopped the collateral damage but turned a
  whole-class outage into ordinary-looking misses.
- **The docstring rationalization.** Commit `c605516` described the 404 as "RefMet has no entry for
  that raw string", documenting the symptom as expected behaviour.
- **Tests pinning `%2F`.** `test_metabolomics_workbench.py`, `test_certificate_tier_b.py`,
  `test_lipidmaps_rest_enricher.py` and `test_mw_slash_breaker.py` asserted `"%2F" in url`, which
  locked the bug in.
- **Round 1: `safe="/"` everywhere** (fork PR #92, org #123, Greptile 5/5). With a literal slash, MW
  answers for the text *before* the slash on non-lipid mixture names, a confident wrong vote:
  `mannitol/sorbitol` -> Mannitol, `butyrate/isobutyrate (4:0)` -> Butyric acid,
  `vitamin D2/D3` -> Vitamin D2, `glucose/zzqqxx` -> Glucose. For some paths MW also returns HTTP 200
  with the plain-text body `This output item does not exist<br> Choose from: ...`; Tier B `_get` read
  that as a JSON decode failure (`lookup_failed`) and tripped the Tier B breaker (43 failures on 98
  slash rows).
- **First "digit-adjacent" regex in round 2:** `(?<=\d)/(?=\s*(?:[A-Za-z]{1,2}-?)?\d)`. Greptile
  showed it kept `vitamin D2/D3` literal (digit on both sides) and encoded `PA 18:1(5Z)/12:0` (a `)`
  precedes the slash).
- **Swapping the prod freeze TSV alone.** Prod runs `REFMET_FREEZE_MODE=live_backup`: the TSV only
  seeds an *empty* SQLite write-through store, so the stale slash `no_match` rows in
  `cache/refmet_store.sqlite` stayed in force.

## Solution

**1. One shared encoder, `utils.quote_name_path_segment`.** A slash stays literal only between two
lipid chain specs; every other slash is encoded.

```python
# Text before the slash must END in carbons:double-bonds + optional suffix ((5Z), ;O3, n3, (OH));
# text after must START with one, after an optional short prefix (P-, O-, d, t).
_CHAIN_BEFORE_SLASH = re.compile(r"\d+:\d+[^\s/,]*$")
_CHAIN_AFTER_SLASH = re.compile(r"\s*(?:[A-Za-z]{1,2}-?)?\d+:\d+")

MW_UNPARSEABLE_PATH_PREFIX = "This output item does not exist"

def quote_name_path_segment(name: str) -> str:
    parts, start = [], 0
    for i, ch in enumerate(name):
        if ch == "/" and _CHAIN_BEFORE_SLASH.search(name, 0, i) and _CHAIN_AFTER_SLASH.match(name, i + 1):
            parts.append(name[start:i])
            start = i + 1
    parts.append(name[start:])
    return "/".join(quote(part, safe="") for part in parts)
```

| Input | Path segment | Server result |
|---|---|---|
| `PC 16:0/18:1` | `PC%2016%3A0/18%3A1` | resolves the full sn-position name |
| `PA 18:1(5Z)/12:0` | `PA%2018%3A1%285Z%29/12%3A0` | resolves |
| `mannitol/sorbitol` | `mannitol%2Fsorbitol` | 404, correct `no_match` |
| `vitamin D2/D3` | `vitamin%20D2%2FD3` | 404, correct `no_match` |

All four call sites use it (MW annotator, `structure_resolver._fetch_mw_inchikey`, `tier_b._fetch_mw`,
`lipidmaps_rest._fetch_json`).

**2. The MW "unparseable path" 200 is not-found, never a failure.** Names with three or more chain
slashes still produce it.

```python
# tier_b._get
response.raise_for_status()
if str(getattr(response, "text", "") or "").startswith(MW_UNPARSEABLE_PATH_PREFIX):
    return None, self._cache_state(response), False   # the name's problem, not an outage
```

**3. PubChem keeps `safe=""`.** PubChem decodes `%2F`, and a literal slash would be read as a PUG-REST
path separator. Encoding must be chosen per service.

**4. Follow-up: PubChem 400/404 returns `None`** (fork #94, org #128). `_fetch_pubchem_inchikey` used
to raise on `PUGREST.NotFound`; the fail-soft `except` in `_resolve_name_key` then returned before the
lipid-resolver hop, so that hop never ran for names PubChem doesn't know.

**5. Tests.** `tests/test_name_path_quoting.py` covers lipid vs mixture encoding (table above), a
mixture name sent encoded to `/refmet/match`, and the unparseable body as not-found. Tests that pinned
`%2F` now pin the exact literal-slash URLs.

**6. Rebuild everything the buggy engine wrote.** Both freezes were built after `a4aeef7`, so their
slash rows are frozen negatives. The remedy depends on the box's freeze mode:
- **Frozen / freeze-first (dev):** re-query only the slash rows with the fixed code and rewrite
  `refmet_freeze_mw_match_2026-09-08.tsv` (276 slash rows) and `tier_b_freeze_20260917T101627Z.tsv`
  (98 slash rows). Check that non-slash rows are byte-identical. Result: 228/276 now vote.
- **`live_backup` (prod):** back up `cache/refmet_store.sqlite`, delete its slash `no_match` rows
  (1,672 rows, 1,460 written live during the bug) so they refill from live, and point
  `REFMET_SNAPSHOT_PATH` at the rebuilt TSV.

**Verification.**
- Paired RefMet arm on the persisted 2026-09-27 1,500-name subsample (client 1.5.5, KRAKEN 2.3.0
  `3dd08a5b`): prod `e324cac` 1,091/1,500 (72.7%) vs dev `a654220` 1,377/1,500 (91.8%). Slash names
  67/448 -> 352/448; non-slash 1,024 -> 1,025/1,052; 287 rows gained, 1 lost (non-slash noise).
- Full prod suite after deploy (`cdd05e5`, suite_20261023T064254Z): RefMet 91.1%, other arms within
  about 1 point.
- Live probe of all 276 freeze slash names: 228 lipids vote on the whole name, 17 mixtures return
  `no_match`, 0 Tier B failures.

## Why This Works

MW and LIPID MAPS are path-addressed Apache services that (a) reject `%2F` at the HTTP layer with no
lookup, and (b) split on a literal `/`, where their lipid nomenclature parser rebuilds an sn-position
name like `PC 16:0/18:1` from the extra segments. So a chain-spec slash must be literal, the only form
the server resolves, and any other slash must be encoded: a literal one makes MW answer for the first
component (a confident wrong id), while the encoded one 404s, which is the correct `no_match` for a
mixture. The regexes match lipid chain grammar (`C:D` plus suffix on the left, optional
`P-`/`O-`/`d`/`t` plus `C:D` on the right) rather than "a digit nearby", which is why `vitamin D2/D3`
stays encoded and `18:1(5Z)/12:0` stays literal. Treating the plain-text 200 as not-found reserves the
breaker for real outages.

## Prevention

- **Live-probe path-addressed APIs with both encodings** (`%2F` and literal `/`) before choosing one,
  and record the probe date and observed responses in the docstring ("Verified live 2026-10-21").
- **Test the exact URL against recorded live behaviour, not a belief about it.** Include a lipid chain
  name (`PC 16:0/18:1`, `PA 18:1(5Z)/12:0`), a mixture name (`mannitol/sorbitol`, `vitamin D2/D3`) and
  a multi-slash isomer name. Never assert an encoded form the server has not been seen to accept.
- **A rule that downgrades errors to `no_match` to protect a breaker needs a per-class canary.** Keep
  a benchmark slice or monitor for slash-bearing rows (`slash_bearing_name_rate` is the exposure
  metric) and alert if its hit rate collapses while the aggregate looks fine. Keep per-row outputs so
  run-to-run swing diffs stay possible.
- **Rebuild freezes and caches after any annotator or URL fix.** Anything the buggy engine wrote holds
  its negatives. Re-query the affected class and confirm unaffected rows are byte-identical.
- **Know each box's `REFMET_FREEZE_MODE`.** `frozen` means rebuild the TSV; `live_backup` means purge
  stale rows from the SQLite store, because the TSV only seeds an empty store.
- **Gate fixes with a paired before/after benchmark** on a persisted, pinned subsample (same names,
  client and KG versions). Report the targeted class and the rest separately, with gained and lost
  rows.
- **A 5/5 review is not a live probe.** Round 1 scored 5/5 on Greptile and still shipped a
  confident-wrong regression on mixtures; a probe against real mixture names caught it.

## Related Issues

- `docs/plans/2026-09-16-002-feat-lipid-hierarchy-aware-mapping-plan.md` (not on `dev`; on the
  `feat/tier-b-graded-level` line): original discovery (F1, slash 404 and process-wide breaker); its Units 2-7 (level cascade, `mapping_relation`/ambiguity
  tags) remain unshipped.
- `docs/solutions/best-practices/trustworthy-gates-invoke-test-real-shape-faithful-fallbacks-2026-08-04.md`:
  tests must assert the real upstream shape, the rule the `%2F` tests broke.
- `docs/solutions/runtime-errors/benchmark-runner-empty-dataset-fail-fast-2026-08-06.md`: a 200 is not
  success; validate the body (same class as the MW "does not exist" page).
- `docs/solutions/integration-issues/github-actions-schedule-trigger-drops-workflow-dispatch-inputs-2026-08-05.md`:
  a 404 is not proof of absence.
- Fork PRs trentleslie/biomapper2 #92 (superseded literal-slash fix), #93 (chain-spec fix), #94
  (PubChem 404); org Phenome-Health/biomapper2 #123, #124, #128; regression origin org #84.
- Background: org issues #20, #28 (MW annotator origin and endpoint choice).
