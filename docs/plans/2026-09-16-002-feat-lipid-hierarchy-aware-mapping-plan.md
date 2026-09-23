---
title: "feat: lipid hierarchy-aware mapping (level cascade, generalization + ambiguity tags, slash-safe lookups)"
type: feat
status: draft
date: 2026-09-16
origin: interactive investigation session 2026-09-16 (Dagflo diagrams 876/877/878), live runs on branch feat/tier-b-graded-level @ 4e8a873
---

# feat: lipid hierarchy-aware mapping

**Target repo:** biomapper2
**Base for PRs:** `dev` on the personal fork `trentleslie/biomapper2`, Greptile first, then org `dev`.
**Paths:** relative to `src/biomapper2/` unless they start with `tests/` or `docs/`.
**Status:** planning only. No code has been changed. Units 0 and 1 are bug fixes that stand alone and
should ship first. Units 2–7 depend on the open decisions in "Decisions Needed".

## Overview

Lipid shorthand input (e.g. `PC 16:0/18:1`) is currently mapped by collapsing it to species level
(`PC 34:1`), and it can land on a *broader* KG node than the input. Nothing in the output records that
detail was lost, or that the result covers many structures. Along the way, several bugs turned up. The
most severe: slash-bearing names trip the RefMet circuit breaker for the whole process, which
degrades RefMet for **every** entity, lipid or not.

This plan:
1. fixes the slash / 404 / breaker defects;
2. fixes the LIPID MAPS multi-row parser;
3. makes Goslin output available at every lipid level and cascades lookups from most specific to least;
4. adds explicit `mapping_relation` (exact / broad / narrow) and `ambiguity` outputs;
5. makes the resolver's tie-break lipid-level-aware;
6. gives Tier B a structure-free lipid check, since lipid species nodes usually have no InChIKey.

## Problem Frame — observed behavior (reproduce before changing anything)

All observations below were made with live services on 2026-09-16. Commands are in "Repro".

### Worked example: `PC 16:0/18:1`, `entity_type="metabolite"`, `candidate_limit=1`, Tier B on

| Step | What happened |
|---|---|
| `metabolomics-workbench` (RefMet `/match`, raw name) | URL-encodes `/` as `%2F`; MW's web server returns an Apache 404 for any encoded slash (`foo%2Fbar` too). `raise_for_status` → retries → `refmet_availability = unavailable`, no vote |
| `goslin-lipid` | pygoslin parses it (dialect Goslin, level `SN_POSITION`), `canonical_name = "PC 34:1"` (species), binder RefMet `/match` → `RM0010728` |
| `kestrel-hybrid-search` (limit 1) | top row `RM:0015001` "PC 16:0/18:1" (exact, canonical, on-category) |
| Link | `RM:0015001` ← 1 CURIE; `RM:0010728` ← 1 CURIE (separate KG nodes; equiv ChEBI `134594` vs `64517`) |
| Resolve | 1–1 tie → `_stable_majority` → both RM (preferred) → `min(_curie_sort_key)` → **`RM:0010728` ("PC 34:1")** |
| Review flag | none: `_choose_best_kg_id` only reads `kg_ids_assigned["metabolomics-workbench"]`, which is empty (404). Goslin's RefMet id is not treated as a RefMet vote |
| Tier B | not looked up: `RM:0010728` has no InChIKey → `in_population` false. Certificate: `state=unavailable`, `structure_status=structure_absent`, `tier_b_outcome=off`, `provenance.tier_b_enabled=false` (despite Tier B being enabled) |

The same pattern occurred for `PE 18:0/20:4` (species node chosen). `TG 16:0_18:1_18:2` got
`selection_conflict=conflict_no_structure`. `LPC 16:0` resolved via PubChem Tier B
(`connectivity`, worst `contradicted`).

### Findings

**F1 (P0, affects all entities): slash names trip the RefMet circuit breaker process-wide.**
`MetabolomicsWorkbenchAnnotator._do_refmet_request` is decorated `@circuit(failure_threshold=3,
recovery_timeout=300)`. A slash-bearing name → MW 404 → `requests.HTTPError` (a `RequestException`)
→ retried → raised → counted as a breaker failure. After three such names, RefMet `/match` is skipped
for 300 s for **all** names and **all instances**: the breaker is attached to the function, so the
Goslin binder's separate instance shares it. Verified:

```
before  carnitine                voted
slash   PC 16:0/18:1             unavailable
slash   PE 18:0/20:4             unavailable
slash   SM d18:1/16:0            unavailable
after   glucose (same instance)  unavailable
after   taurine (OTHER instance) unavailable
```

This matters most for the long-lived FastAPI process and for dataset runs with lipid panels. It
applies when the RefMet freeze mode is `off` (no `REFMET_SNAPSHOT_PATH`), which was the local
default in these runs. Confirm the deployed freeze mode.

The docstring on `_request_once` claims `quote(..., safe="")` makes slash names safe. That was true
of the *old* failure (an extra path segment silently returning no candidate), but the server rejects
`%2F` outright, and double-encoding (`%252F`) returns the no-match sentinel. **MW REST cannot be
queried with a slash in the name at all.**

**F2: a 404 is classified `unavailable`, not `no_match`.** A definitive "this path doesn't exist"
answer is folded into the outage axis, which also feeds F1.

**F3: Tier B MW and PubChem hops 404 on slash names** (`tier_b.py::_fetch_mw`, `_fetch_pubchem`,
both `quote(safe="")`). They return `None` → `unresolvable`: honest about the result, but the name
was never actually looked up. `structure_resolver._resolve_name_key` has the same exposure on the
candidate side.

**F4: the LIPID MAPS enricher ignores multi-row responses.** `lipidmaps_rest.py::enrich_checked`
reads top-level `lm_id` / `inchi_key`. For species and molecular-species queries, LIPID MAPS returns
`{"Row1": {...}, "Row2": {...}, ...}`, so the result is `{}` with `ok=True` → `unresolvable`. Also:
- `abbrev/PC 34:1` returned 22 rows, and **row order changed between two calls**. Never take `Row1`.
- `abbrev` with any slash → 404 (same web-server behavior as MW).
- `abbrev/PC 16:0_18:1` → `[]`, but `abbrev_chains/PC 16:0_18:1` → 8 rows: 6 with `16:0` at sn-1
  (double-bond variants 6Z/6E/9Z/9E/11Z/11E, which gives **3 distinct InChIKey first blocks**) and 2 reversed.

**F5: Goslin always canonicalizes to species level.** `goslin_grammar.py::_canonical_name` calls
`get_lipid_string(LipidLevel.SPECIES)`. pygoslin can emit every level, verified for the same parse:
`SPECIES → "PC 34:1"`, `MOLECULAR_SPECIES → "PC 16:0_18:1"`, `SN_POSITION → "PC 16:0/18:1"`.
The canonical name only goes to the RefMet binder. Kestrel hybrid search and Tier B never see Goslin
output.

**F6: the resolver's tie-break is level-blind.** On a genuine tie, the lower numeric local id wins
(`_curie_sort_key`). For lipids this is arbitrary with respect to specificity, and in the example it
chose the broader node over the exact one without any flag.

**F7: Tier B is structurally blind to most lipid mappings.** Tier B needs the committed node to carry
an InChIKey (`node_blocks_from_equivalent_ids`). Species-level lipid nodes generally have none.
Separately, a row that is out of scope reports `tier_b_outcome=off` and `provenance.tier_b_enabled=false`
even when Tier B is enabled. That's misleading, because it reads as "Tier B disabled".

**F8 (minor, verify): `annotator_availability` shows `kestrel-hybrid-search` and `goslin-lipid` as
`not_queried` even when they contributed votes.** It may be intended to track only RefMet-backed
sources. If so, document it. If not, fix it.

## Best-practice grounding

- **"/" is a claim of proof.** Shorthand notation (Liebisch et al. 2013; LIPID MAPS update 2020): `_` =
  sn-position *not known*, `/` = sn-position *proven*. Levels: species (`PC 34:1`) → molecular species
  (`PC 16:0_18:1`) → sn-position (`PC 16:0/18:1`) → DB-position (`PC 16:0/18:1(9Z)`) → structure defined
  → full structure → complete structure. The 2020 update warns that annotations from m/z alone are
  "frequently incorrect due to over-interpretation".
- **Lipidomics Standards Initiative:** "report only what is experimentally proven". Database search on
  MS data is typically only defensible at species / molecular-species level.
- **RefMet** intentionally has species-level ("sum composition") entries for precursor-only data.
- **SwissLipids** is the only source with an explicit parent/child hierarchy (category → class →
  species → molecular → structural → isomeric subspecies), available via its SPARQL endpoint and mapper.
- **Goslin 2.0** can emit a name at any level, but defines no strategy for one name matching many
  database entries. That is our responsibility.
- **SSSOM / SKOS** provide the standard vocabulary for mapping relations: `skos:exactMatch`,
  `skos:broadMatch` (object is broader than subject), `skos:narrowMatch`, plus
  `mapping_justification` (e.g. lexical match).

Sources: see "Sources & References".

## Requirements

- **R1.** A slash-bearing name must never degrade RefMet for other entities (fix F1). A 404 from a
  path-addressed registry is `no_match` or `not_queryable`, never an outage (F2).
- **R2.** Never send a raw `/` to MW or LIPID MAPS REST path segments. Use the `_` form (plus
  client-side sn filtering) or a local source (F1, F3, F4).
- **R3.** The LIPID MAPS enricher returns a **set** of candidate structures for multi-row responses,
  order-independent. It never picks `Row1` (F4).
- **R4.** Goslin output exposes every level name, the input's level, formula, mass and dialect (F5).
- **R5.** Lipid lookups cascade from the input's level down to species, across RefMet, LIPID MAPS
  and Kestrel, and record the level at which each hit was found (F5).
- **R6.** Every lipid row reports `query_lipid_level`, `matched_lipid_level` and `mapping_relation`
  ∈ {`exact`, `broad`, `narrow`, `unknown`}, SSSOM-aligned (the core ask: "query was changed to a more
  ambiguous level").
- **R7.** Every lipid row reports ambiguity separately from generalization: `ambiguous` (bool),
  `candidate_structure_count`, `ambiguity_basis` (e.g. `lipidmaps_abbrev_chains`), judged at the
  matched KG node's level.
- **R8.** On a tie between lipid nodes, the resolver prefers the node whose level equals the query
  level, and flags a generalization it could not avoid (F6).
- **R9.** Tier B gains a structure-free lipid check and set-based structure comparison, and
  distinguishes "out of scope" from "disabled" (F7).
- **R10.** All new outputs are **additive** in the API response and dataset flat columns. Existing
  field meanings don't change.

## Scope Boundaries

- `biolink:SmallMolecule` lipids only, meaning names pygoslin parses. Non-lipids follow today's path
  unchanged, except for the R1 breaker fix, which benefits everyone.
- Keep `certificate.issue()` pure and IO-free (existing constraint). Pre-compute lipid evidence and
  pass it in.
- Do not rebuild existing components: `LipidGrammar`, `GoslinLipidAnnotator`, `LipidStructureResolver`,
  `LipidMapsRestEnricher`, `StructureResolver`, `IndependentStructureLookup`, the RefMet freeze.
  Extend them.
- **Not in scope:** SwissLipids integration (see Decisions), changes to the Dagflo diagrams.

## Coordination with in-flight work

`docs/plans/2026-09-16-001-feat-tier-b-resolution-level-plan.md` (R6: default Tier B on for
metabolites) interacts with F1 and F3. Turning Tier B on by default multiplies slash-name calls
against MW. Unit 0 should land **before** Tier B is defaulted on. Unit 6 touches the same
`issue()` / `ResolutionCertificate` surface as that plan's R1–R4, so coordinate field names to avoid
merge conflicts.

## Decisions (RATIFIED 2026-09-16 by Trent — all recommendations below accepted as written)

1. **Trust "/"?** Treat input `/` as proven sn-position, or downgrade to `_` by default? Vendors often
   use `/` loosely. *Recommendation:* config flag `LIPID_TRUST_SN_POSITION`, default `false`; record
   `query_lipid_level_asserted` and `query_lipid_level_effective`.
2. **Result policy:** choose the most specific KG node that matches, or deliberately stop at the level
   the data supports? *Recommendation:* most specific node at or above the *effective* query level,
   never below it (never `narrow`); always emit `mapping_relation`.
3. **Kestrel level queries:** send level-specific names to Kestrel hybrid search (one extra call per
   level)? *Recommendation:* yes, but only for levels not already hit, within the same
   `candidate_limit`.
4. **SwissLipids:** add as a hierarchy source (parent/child links) now or later? *Recommendation:*
   later, as a separate plan. The cascade in Unit 3 gets most of the value without it.
5. **Where the new fields live:** a separate `lipid_resolution` object on `EntityMappingResult`
   (recommended), or new fields on `ResolutionCertificate`? *Recommendation:* separate object; mirror
   `mapping_relation` and `ambiguous` into the certificate's provenance so a certificate read alone
   still shows them.

## Implementation Units

Keep each test file to ≤ 8 tests (project rule). Mark live-service tests `@pytest.mark.integration`.
Include at least one end-to-end test through `Mapper.map_entity_to_kg()` per feature. Do not restate
measured figures in code comments (`tests/test_no_measured_figures_in_prose.py`).

### Unit 0 — Slash-safe RefMet and breaker isolation (P0, ship first)
**Files:** `core/annotators/metabolomics_workbench.py`, `core/tier_b.py`, `core/structure_resolver.py`, `config.py`
- Before any MW REST call: if the name contains `/`, do not call `/match` with it. Either (a) query the
  `_`-substituted form and record `query_transformed="slash_to_underscore"`, or (b) return
  `not_queryable` without a network call. Pick one, applied the same way at all MW call sites (the
  annotator, Tier B `_fetch_mw`, and the structure resolver's MW fetch).
- Classify HTTP 404 (and 400) from `/match` as `no_match`, not a raised exception, so it never counts
  toward the breaker.
- Consider per-name 4xx exclusion from the breaker in general: only 5xx, timeout and transport errors
  should count.
- Update the `_request_once` docstring. Its `safe=""` rationale is incomplete (the server rejects `%2F`).
- **Tests:** (1) three slash names then `carnitine` → `carnitine` is not `unavailable` (stubbed
  session returning 404); (2) 404 → `no_match`; (3) second annotator instance unaffected;
  (4) integration: the repro script below.

### Unit 1 — LIPID MAPS multi-row parsing (ship second)
**Files:** `core/annotators/lipidmaps_rest.py`, `core/lipid_structure_resolver.py`, `core/certificate.py` (TierBOutcome)
- Parse `RowN`-wrapped responses into a list of `{lm_id, name, abbrev, abbrev_chains, inchi_key}`.
  Sort deterministically (e.g. by `lm_id`). Never depend on response order.
- `enrich_checked` returns a single mapping only when exactly one row exists. Otherwise expose the set
  (new method, e.g. `candidates_checked`).
- `LipidStructureResolver`: on more than one candidate, return a new `TierBOutcome.AMBIGUOUS` carrying
  the candidate key set, instead of `unresolvable`.
- Route slash names through `abbrev_chains` with `_`, then filter client-side by sn order (see Unit 2
  for the chain list).
- **Tests:** multi-row fixture → set; row-order permutation → identical output; single row → as today;
  empty list → `unresolvable`; 5xx → `lookup_failed`.

### Unit 2 — Goslin exposes every level
**Files:** `core/annotators/goslin_grammar.py`, `core/annotators/goslin_lipid.py`
- Extend `LipidParse` with `level_names: dict[str, str]` (species / molecular_species / sn_position /
  … as available), `input_level`, and the ordered chain list (for sn filtering).
- Keep `canonical_name` as-is for backward compatibility (species level).
- Add all of it to the Goslin annotator metadata (`goslin_level_names`, `goslin_input_level`).
- **Tests:** `PC 16:0/18:1` → three level names; `PC 34:1` → species only; non-lipid → `None`.

### Unit 3 — Level cascade lookups (after Decisions 1–3)
**Files:** `core/annotators/goslin_lipid.py` (binder loop), `core/annotation_engine.py`, possibly `core/annotators/kestrel_hybrid.py`
- For levels from the effective query level down to species: RefMet `/match` (slash-safe per Unit 0),
  LIPID MAPS (Unit 1), Kestrel hybrid search with the level name (Decision 3). Stop at the first level
  with a hit per source. Record `matched_level` on each vote's metadata.
- The votes enter the existing assigned-ID flow unchanged; only metadata is added.
- **Tests:** stubbed sources where only species hits → `matched_level=species`; sn-level Kestrel hit
  → `matched_level=sn_position`; end-to-end `Mapper.map_entity_to_kg("PC 16:0/18:1")`.

### Unit 4 — Level-aware lipid tie-break
**Files:** `core/resolver.py`
- In `_stable_majority`, when tied candidates are lipids with known levels, prefer the candidate whose
  level equals the effective query level. Then fall back to today's namespace and numeric order.
- If the chosen node is broader than the query level, set a review hint (e.g. `lipid_generalized`).
  Coordinate the name with `selection_conflict`'s closed whitelist; this may need to be a new field
  rather than a new whitelist value.
- **Tests:** the worked example → `RM:0015001` chosen; only a species node available → species chosen +
  hint; non-lipid ties unchanged.

### Unit 5 — Output fields (API + dataset)
**Files:** `api/models/responses.py`, `api/routes/mapping.py` (`extract_mapping_result`), `mapper.py`, dataset flat-column emission
- New `lipid_resolution` object (null for non-lipids): `query_lipid_level_asserted`,
  `query_lipid_level_effective`, `matched_lipid_level`, `mapping_relation` (`exact|broad|narrow|unknown`),
  `mapping_predicate` (`skos:exactMatch|skos:broadMatch|skos:narrowMatch`), `query_transformed`
  (e.g. `slash_to_underscore`, `goslin_species_canonical`), `ambiguous`, `candidate_structure_count`,
  `ambiguity_basis`, `goslin_dialect`, `goslin_formula`, `goslin_mass`.
- Mirror `mapping_relation` and `ambiguous` into certificate provenance (Decision 5).
- Flat columns for the dataset path, prefixed `lipid_`.
- **Tests:** `tests/test_certificate_api_surface.py`-style schema test; a non-lipid row has
  `lipid_resolution: null`; an end-to-end lipid row has all fields.

### Unit 6 — Tier B for lipids
**Files:** `mapper.py` (`_issue_certificate` population predicate), `core/certificate.py`, `core/tier_b.py`
- **Structure-free check:** parse the committed node's *name* with Goslin and compare it to the query
  parse. Same class and same sum composition → corroborated at `lipid_species` level. Different →
  contradicted. The node at a coarser level than the query → corroborated plus `mapping_relation=broad`.
  Add `ResolutionLevel` values (or a parallel `lipid_resolution_level`) and a new `comparison_rule`,
  e.g. `goslin_level_composition/v1`.
- **Set-based structure check:** when LIPID MAPS returns a candidate set (Unit 1) and the node has
  InChIKeys, corroborate if they intersect. Record `candidate_structure_count`.
- **Population:** lipid rows with no node InChIKey become in scope for the structure-free check.
- **Honest outcome:** add a Tier B outcome `out_of_scope` (distinct from `off`), and set
  `provenance.tier_b_enabled` from config, not from whether a lookup ran.
- **Tests:** composition match / mismatch; node broader; intersecting set; out of scope vs off.

### Unit 7 — Availability map accuracy (minor)
**Files:** `core/annotation_engine.py`
- Verify F8. Either populate `annotator_availability` for non-RefMet annotators or document in the
  API field description that it covers RefMet-backed sources only.

## Repro

```bash
# F1 breaker (live MW; freeze mode off)
uv run python -c "
import logging; logging.disable(logging.CRITICAL)
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator as M
a=M(); b=M()
s=lambda x,n: x.get_availability({'name':n},'name')['metabolomics-workbench']
print('carnitine', s(a,'carnitine'))
for n in ['PC 16:0/18:1','PE 18:0/20:4','SM d18:1/16:0']: print(n, s(a,n))
print('glucose', s(a,'glucose')); print('taurine (other instance)', s(b,'taurine'))
"

# Encoded slash rejected by both registries
curl -s -o /dev/null -w '%{http_code}\n' 'https://www.metabolomicsworkbench.org/rest/refmet/match/foo%2Fbar'   # 404
curl -s 'https://www.metabolomicsworkbench.org/rest/refmet/match/PC%2016%3A0_18%3A1'                          # RM0090134
curl -s 'https://www.lipidmaps.org/rest/compound/abbrev_chains/PC%2016%3A0_18%3A1/all/json'                   # Row1..Row8

# Worked example end to end
BIOMAPPER2_TIER_B_ENABLED=1 uv run python -c "
import json, logging; logging.disable(logging.CRITICAL)
from biomapper2.mapper import Mapper
r = Mapper().map_entity_to_kg(item={'name':'PC 16:0/18:1'}, name_field='name', provided_id_fields=[],
    entity_type='metabolite', annotation_mode='all', candidate_limit=1)
print(json.dumps({k: r[k] for k in ['assigned_ids','kg_ids','chosen_kg_id','resolution_certificate']}, indent=1, default=str))
"
```

Run the breaker repro in a fresh process: the breaker state persists within a process for 300 s.

## Risks

- **Live-service drift:** LIPID MAPS row order and contents, RefMet names and Kestrel scores all
  change. Tests must use fixtures, with integration tests only as smoke checks.
- **Call volume:** the cascade (Unit 3) and lipid Tier B (Unit 6) add external calls. Respect the
  existing throttles, memoize per name and level, and keep failures out of the breaker (Unit 0).
- **Contract:** all new fields are additive (R10). `selection_conflict` is a closed whitelist, so don't
  widen it without the contract discussion noted in `resolver.py`.
- **Over-claiming:** without Decision 1, an input `/` would be trusted and `mapping_relation=exact`
  could be reported for an sn-level match the data never proved.

## Sources & References

- Liebisch et al. 2020, *Update on LIPID MAPS classification, nomenclature, and shorthand notation for MS-derived lipid structures*, J Lipid Res — https://pmc.ncbi.nlm.nih.gov/articles/PMC7707175/
- Liebisch et al. 2013, *Shorthand notation for lipid structures derived from mass spectrometry*, J Lipid Res — https://www.jlr.org/article/S0022-2275(20)35708-4/fulltext
- Lipidomics Standards Initiative, *Recommendations for good practice in MS-based lipidomics* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8585648/
- Kopczynski et al., *Goslin 2.0* — https://pmc.ncbi.nlm.nih.gov/articles/PMC9047418/ ; *Goslin* — https://pubs.acs.org/doi/10.1021/acs.analchem.0c01690
- Aimo et al., *The SwissLipids knowledgebase for lipid biology* — https://academic.oup.com/bioinformatics/article/31/17/2860/183669
- RefMet help — https://www.metabolomicsworkbench.org/databases/refmet/refmet_help.php ; *RefMet: a reference nomenclature for metabolomics* — https://www.nature.com/articles/s41592-020-01009-y
- SSSOM `predicate_id` — https://mapping-commons.github.io/sssom/predicate_id/ ; SKOS reference — https://www.w3.org/TR/skos-reference/
- Code references (branch feat/tier-b-graded-level @ 4e8a873): `core/annotators/metabolomics_workbench.py` (`_do_refmet_request`, `_request_once`), `core/annotators/goslin_grammar.py` (`_canonical_name`), `core/annotators/goslin_lipid.py`, `core/annotators/lipidmaps_rest.py` (`enrich_checked`), `core/lipid_structure_resolver.py`, `core/tier_b.py` (`_resolve`, `_fetch_mw`, `_fetch_pubchem`), `core/structure_resolver.py` (`_resolve_name_key`), `core/resolver.py` (`_stable_majority`, `_choose_best_kg_id`), `mapper.py` (`_issue_certificate`), `core/certificate.py` (`issue`, `TierBOutcome`, `ResolutionCertificate`).
- Dagflo walkthroughs from the investigation: https://www.dagflo.com/diagrams/diagram/876 (pipeline), /877 (SmallMolecule gate, Tier B, candidate_limit), /878 (PC 16:0/18:1).
