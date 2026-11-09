---
date: 2026-11-08
topic: icd10cm-fallback
---

# ICD-10-CM: provided-code fallback now, disease-name annotator later

## Problem Frame

#141 (released in v1.3.0) made `icd10cm` its own vocabulary, so `ICD10CM:` ids keep their prefix instead of being fuzzy-rewritten to `ICD10:`. That was the right call, but KRAKEN 2.3.0 has no `ICD10CM` nodes yet (Phenome-Health/kraken#30, owned by Adil Bazarkulov). The old rewrite was quietly doing useful work. KRAKEN's 2,586 `ICD10:` ids include 2,164 dotted subcodes, and the 585 with two or more digits after the dot are really ICD-10-CM codes stored under `ICD10` (NCATSTranslator/Babel#1033). Checked against public Kestrel `/canonicalize` on 2026-11-07:

| Input | Result |
|---|---|
| `ICD10:E75.21` | `MONDO:0010526` |
| `ICD10CM:E75.21` | null |
| `ICD10:E11` | `MONDO:0005148` |
| `ICD10CM:E11` | null |
| `ICD10:E11.65` / `ICD10CM:E11.65` | null / null |

So since v1.3.0, coded clinical data that resolved before now comes back unresolved. The #141/#142 PR text understated this ("subcodes don't resolve either way").

The RefMet pattern (a category-gated annotator in `_select_annotators`) does not fix this: annotators work from names, and in the default `missing` mode they skip any row that already has a provided id. The rows that regressed all have a provided id. The closer precedent is the gene-symbol fallback: a deterministic bridge with a kill switch, applied on a miss.

Who is affected: anyone mapping claims or EHR diagnosis codes through biomapper2, starting with Adil's coded-diagnosis work.

## Requirements

**Phase 1a: same-code twin retry (the regression fix, ships first)**
- R1. When a provided `ICD10CM:<code>` id does not link to a KG node, retry the same code as `ICD10:<code>`. A hit counts as **resolved** with `mapping_relation = exact`, **provided the control gate in R1b passes**.
- R1b. Control gate before shipping: compare twin-hit targets with Mondo's SSSOM `skos:exactMatch` rows to ICD10CM (2,099 rows, ranges excluded) on the codes the two sets share. If agreement clears the threshold set at planning sign-off, twin hits ship as exact. If it doesn't, twin hits are reported as `broad`/unverified, like walk-up hits. Report the agreement number together with its artifact.
- R4. The fallback is triggered by the provided id's **prefix** (`ICD10CM`), whatever the row's entity type.
- R5. Direct linking always wins. The fallback runs only for rows where no provided or annotator curie linked, so a fallback curie never outvotes a direct one in the resolver. Once KRAKEN has `ICD10CM` nodes, the fallback runs only on misses.
- R6. A config kill switch turns the fallback off without a code revert, mirroring `GENE_SYMBOL_FALLBACK_ENABLED`. Phase 1b gets its own switch, so the riskier walk-up can be disabled on its own.
- R7a. A failed fallback lookup (Kestrel timeout/5xx) makes the row `error`, never `unresolved`. A broader candidate is never committed unless every more specific candidate returned a definite null.
- R7. Output records which route produced the match (direct / same-code twin / parent walk-up, plus the parent code used) and the KG build it was resolved against (Kestrel `/metagraph` fingerprint), so a result can be traced and reproduced.

**Phase 1b: parent walk-up (new capability, its own PR after 1a)**
- R2. If the same-code retry also misses, walk up the ICD-10-CM hierarchy (e.g. `E11.65` → `E11.6` → `E11`), trying each parent as `ICD10CM:` and then `ICD10:`. The most specific hit wins. A walk-up hit is `resolved` with **`mapping_relation = broad`** (`skos:broadMatch`), reusing the lipid `mapping_relation` precedent made entity-general. It is never presented as exact.
- R3. Candidate parents are successive truncations that **exist in the pinned CDC snapshot**. Drop a 7th-character extension first. Skip intermediates that don't exist, such as placeholder-X forms (`T36.0X`). Stop at the 3-character category, and never cross into a range (`I10-I16`) or another category. A provided code missing from the snapshot (newer FY, deleted, typo) gets the twin retry only, no walk-up, and the route records that.
- R3b. Results record the pinned hierarchy (CDC FY) version, but only on rows the walk-up served.
- R3c. Benchmark and analysis scorers count `broad` separately from exact `resolved`.

**Phase 2: disease-name annotator (after KRAKEN ships CM nodes AND a concrete user need)**
- R8. A name→ICD-10-CM annotator, selected in `_select_annotators` for `biolink:DiseaseOrPhenotypicFeature` and its descendants. This is the RefMet/SmallMolecule pattern, using a broader category than `Disease` because CM includes symptom (R), status (Z) and injury (S/T) codes.
- R9. Its vote is one candidate among the existing annotators and never overrides a MONDO hit from Kestrel search. `CATEGORY_PREFERRED_NAMESPACES` stays `{"MONDO"}` for Disease.
- R10. Phase 2 does not start until (a) Adil's KRAKEN build with `ICD10CM` nodes is live, and (b) someone needs CM codes from names, e.g. a pipeline that wants CM codes as output. Before (a), the annotator would emit codes that link to nothing. Phase 2 gets its own brainstorm when it starts.

## Success Criteria
- **1a:** On a fixed list of `ICD10CM:` codes, everything that resolved on v1.2.x through the `ICD10:` rewrite resolves again, to the same node. Include codes like `E75.21`, `E83.01`, `F40.0`, `I10` and `E11`.
- **1a:** The R1b control agreement is measured and reported against Mondo SSSOM, not against v1.2.x output. Reproducing old behavior is not proof that it is correct.
- **1b:** `ICD10CM:E11.65` returns `MONDO:0005148` with `mapping_relation = broad`, and existing scorers do not count it as exact.
- No previously resolved non-ICD row changes (existing benchmark suite, no new regressions).
- The test set includes real billable patient codes, not just header codes like `E11`, and each code is labeled with its expected route as checked against `/canonicalize`. Checked 2026-11-08: `I10` is a twin hit (`ICD10:I10` → `MONDO:0005044`). `F32.9` is a walk-up to `F32` (`MONDO:0002009`). `E11.65` is a walk-up to `E11`. (Adil's "no Mondo row" for `I10`/`F32.9` refers to Mondo's SSSOM file, not to Kestrel.) Report coverage by route on a realistic billable sample (Adil's code list if available).
- After kraken#30 ships, measure how often the fallback is still used, instead of assuming it disappears. Then decide whether the walk-up stays as a deliberate feature (and who owns refreshing the CDC snapshot) or gets switched off.

## Scope Boundaries
- No ICD-10-CM nodes, clustering or equivalence logic in biomapper2. That is KRAKEN's job (kraken#30).
- No UMLS-sourced ICD-10-CM content. Its license category bars public systems (Adil's reading of UMLS §12.3/12.4).
- No change to Disease preferred-namespace ranking.
- No WHO ICD-10 → CM translation beyond the same-code twin.

## Key Decisions
- Same-code twin = exact if it passes the Mondo control gate (R1b), because restoring old behavior alone doesn't prove it right. Parent walk-up = `mapping_relation = broad`, because the coverage is real but less specific.
- Split Phase 1: 1a (twin) is a small regression patch; 1b (walk-up, CDC snapshot, broad plumbing) is new capability that shouldn't hold 1a up. Three reviewers raised this independently.
- Broad-match label: reuse the existing `mapping_relation = exact|broad` (lipid precedent) instead of adding a new status, so the API has one broad-match vocabulary.
- Trigger the fallback by prefix rather than category: a provided `ICD10CM:` id states its own code system, so the entity type adds nothing and could wrongly block it.
- Do the regression fix before the annotator: the fallback pays off immediately, while the annotator is blocked on KRAKEN.
- Hierarchy source = the CDC `icd10cm_order_<FY>.txt` file (all codes including headers, with a header flag; ~98k entries), not the billable-only `icd10cm_codes_<FY>.txt`. It is free, published by the US government, and pinned and shipped as a local snapshot. Codes in it are undotted (`E1165`), so they are converted to dotted form (dot after the 3rd character) before comparing with curies. NLM Clinical Tables covers billable codes only, so it can't supply parents (checked 2026-11-07: an `E11.6` query returns its 12 children, not `E11.6`).

## Dependencies / Assumptions
- Assumption: Adil's KRAKEN plan uses the same CDC file. Pin the same FY release in both repos.
- Assumption: a parent walk-up needs no Babel lookup of its own, because KRAKEN's `ICD10:` ids resolve through `/canonicalize` like any other prefix.
- Risk: if Babel#1033 is fixed upstream, the CM-shaped `ICD10:` ids may disappear and the twin's coverage with them. The kill switch would not detect this; the post-kraken#30 fallback-usage check would.
- Out of scope but noted: undotted CM input (`E1165`, common in claims extracts) fails `is_icd10_id` today and never reaches the fallback.

## Outstanding Questions

### Resolve Before Planning
(none)

### Decide at plan sign-off
- [Affects R1b][User decision] The twin-agreement threshold that counts as passing, e.g. ≥95% on the codes shared with Mondo SSSOM.

### Deferred to Planning
- [Affects R2, R7][Technical] How `mapping_relation` becomes entity-general. Precedent: `lipid_resolution.mapping_relation` (`exact`/`broad`, `skos:exactMatch`/`skos:broadMatch`), built in `core/resolver.py` `lipid_mapping_relation`, exposed in `api/models/responses.py`, and mirrored into certificate provenance by `Mapper._issue_certificate`. It must reach both the API response and the PyPI client.
- [Affects R1, R2][Technical] Resolve all twin and ancestor candidates in one batched `/canonicalize` call, then pick the most specific hit (prod is ~4.4 s/entity and rate-limited).
- [Affects R1–R3][Technical] Which stage owns the retry: normalizer, linker or resolver. Read `core/linker.py` and `core/resolver.py`; the gene-symbol bridge sits in an annotator, which is the wrong stage for provided ids.
- [Affects R7][Technical] How the CDC snapshot is shipped and versioned (package data vs. a pinned path with a version sidecar, as `REFMET_SNAPSHOT_PATH` does).
- [Affects R8][Needs research] For Phase 2 name lookup, the CDC file's own names (local index, no live calls, matching the RefMet freeze approach) vs. live NLM Clinical Tables. Leaning local.
- [Affects success criteria][Needs research] Build the regression list by running v1.2.x vs. v1.3.0 over the CDC code set, or over KRAKEN's 2,586 `ICD10:` ids, against prod.

## Next Steps
→ `/ce:plan` for Phase 1a (then 1b). Phase 2 waits on kraken#30 plus a concrete request.
