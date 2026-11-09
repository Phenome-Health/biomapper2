---
title: "feat: ICD-10-CM provided-id fallback (same-code twin, then parent walk-up)"
type: feat
status: active
date: 2026-11-08
origin: docs/brainstorms/2026-11-08-icd10cm-fallback-requirements.md
---

# feat: ICD-10-CM provided-id fallback (same-code twin, then parent walk-up)

## Overview

Since v1.3.0 (#141), a provided `ICD10CM:<code>` keeps its prefix, but KRAKEN 2.3.0 has no `ICD10CM` nodes. Codes that used to resolve through the accidental `ICD10:` rewrite now come back unresolved. This plan adds a provided-id fallback in two PRs:

- **Phase 1a (twin).** On a miss, retry the same code as `ICD10:<code>`. This restores pre-#141 behavior, gated on a Mondo control check.
- **Phase 1b (walk-up).** If the twin also misses, climb the ICD-10-CM hierarchy using a pinned CDC snapshot, and label the hit `mapping_relation = broad`.

Phase 2 (a disease-name annotator) is out of scope (see origin: docs/brainstorms/2026-11-08-icd10cm-fallback-requirements.md).

## Problem Frame

See origin. In short:

- On public Kestrel (2026-11-07/08), `ICD10:E75.21`, `ICD10:F40.0`, `ICD10:E11`, `ICD10:I10` and `ICD10:F32` resolve to MONDO nodes, but their `ICD10CM:` forms are null.
- KRAKEN's `ICD10:` namespace already holds about 585 CM-shaped ids (Babel#1033).
- `E11.65` and `F32.9` resolve under neither prefix.
- Adil Bazarkulov's coded-diagnosis work (kraken#30) is the first affected consumer.

## Requirements Trace

**Phase 1a**
- R1. A provided `ICD10CM:X` that misses is retried as `ICD10:X`. A hit is resolved, with `mapping_relation` = the gate-decided value.
- R1b. The twin relation is gated on agreement with Mondo SSSOM exactMatch rows. The agreement number is reported with its artifact. The threshold is decided by Trent at plan sign-off.
- R4. The fallback is triggered by the `ICD10CM` prefix, whatever the entity type.
- R5. Fallback curies only act on rows where no direct curie linked, so they never outvote a direct curie.
- R6. Kill switches: one for the fallback as a whole, and a separate one for the walk-up.
- R7a. A failed fallback lookup is an error, never a silent unresolved. A broad candidate is only committed when every more specific candidate came back as a definite null.
- R7. Each row records its route (direct / twin / walk-up, plus the parent code). For the KG build, see Key Technical Decisions.

**Phase 1b**
- R2. Walk-up hits are resolved with `mapping_relation = broad` (`skos:broadMatch`). The most specific hit wins.
- R3. Candidates are truncations that exist in the snapshot. The 7th character is dropped first, placeholder-X intermediates are skipped, and the walk-up stops at 3 characters. A code that isn't in the snapshot gets the twin retry only.
- R3b. The snapshot version is stamped only on rows the walk-up served.
- R3c. Scorers report broad separately from exact. This lives in the client repo (Deferred to Separate Tasks).

**Success criteria (origin), by phase:**
- 1a: the regression list is restored, and the control agreement is reported (Unit 1, Unit 3).
- 1b: `E11.65` → `MONDO:0005148` broad (Unit 5).
- Both: no non-ICD regressions; each test code is labeled with its route.
- Later: fallback usage measured after kraken#30 (Deferred to Separate Tasks).

## Scope Boundaries

- No ICD-10-CM nodes or clustering. That is KRAKEN's job (kraken#30).
- No UMLS-sourced content, because of the license.
- No change to Disease `CATEGORY_PREFERRED_NAMESPACES` (stays `{"MONDO"}`).
- No support for undotted CM input (`E1165`). It fails `is_icd10_id` before reaching the fallback. That gap existed before this work and is noted in origin.
- No new engine-level `status` field. The engine has none today (see Context).
- Phase 2 (disease-name annotator): gets its own brainstorm after kraken#30 ships and a concrete request exists.

### Deferred to Separate Tasks

- **Client `biomapper` repo:** surface the new `mapping_relation` / `icd10cm_route` fields in `src/biomapper/models.py`, and split broad vs exact in `benchmarks/scorers/` (`curie_scorer.py`, `arm_scoring.py`) (R3c). This is a separate PR in that repo after 1b merges.
- **Per-row KG build fingerprint for non-ICD rows.** That gap applies to every entity type, so it is a separate task if wanted. (ICD fallback rows do get a stamp in this plan; see Key Technical Decisions.)
- **Post-kraken#30 fallback-usage measurement**, and the keep-or-retire decision on the walk-up. Rerun the Unit 1 study once KRAKEN ships CM nodes.
- **KRAKEN `nextversion` pin bump to ≥ v1.3.0.** Adil's call, not required.
- **Phase 2 name annotator.** Gets its own brainstorm.

## Context & Research

### Relevant Code and Patterns

- **Pipeline:** `src/biomapper2/mapper.py`.
  - `map_entity_to_kg`: normalize → link → resolve → certify.
  - `map_dataset_to_kg`: the same steps over a DataFrame.
  - The row update writes `chosen_kg_id`, `lipid_resolution` and `refmet_snapshot_version`.
- **Linker:** `src/biomapper2/core/linker.py`.
  - `_link_dataframe` unions every row's curies into one `get_kg_ids` call. `_link_entity` handles the single-entity path.
  - `get_kg_ids` batches `/canonicalize`; misses are simply absent from the result.
  - `_reverse_curie_map` keys `kg_ids` by the **original** curie.
  - **Seam:** write twin/ancestor hits back under the original `ICD10CM:` key, so voting needs no change.
- **Resolver:** `src/biomapper2/core/resolver.py`.
  - `_choose_best_kg_id` / `_stable_majority`: vote by count of supporting curies. Disease rows get the plain majority vote.
  - `lipid_mapping_relation` and the relation constants (`exact`/`broad`/`narrow`/`unknown` plus SKOS predicates).
  - `build_lipid_resolution` and `LIPID_RESOLUTION_FIELDS`.
- **API:** `src/biomapper2/api/models/responses.py` (`LipidResolution.mapping_relation`, result model) and `src/biomapper2/api/routes/mapping.py`.
  - `extract_mapping_result`.
  - Per-row `error=str(e)` in the single, batch and NDJSON stream routes. A failure fails the whole `/map/dataset` call.
- **Certificate mirror:** `Mapper._issue_certificate` adds `mapping_relation` to `extra_provenance` only when non-None, so non-lipid certificates are byte-identical today.
- **Snapshot precedent:**
  - `src/biomapper2/core/annotators/refmet_snapshot.py`: `reset` / `is_present` / `version` / `lookup`; a malformed freeze is rejected whole.
  - `config.py` `derive_refmet_snapshot_version` (`.version` sidecar).
  - `mapper.py` `_refmet_snapshot_version_for`: version stamped only on rows the freeze served.
- **Kill-switch precedent:**
  - `config.GENE_SYMBOL_FALLBACK_ENABLED`: module constant.
  - `config.RERESOLUTION_ENABLED`: env-driven; tests flip it via `monkeypatch.setattr(config, ...)`.
- **Validators:** `src/biomapper2/core/normalizer/validators.py` `is_icd10_id` (dotted only, beyond 3 characters); `vocab_config.py` `icd10cm` entry. Tests: `tests/test_normalizer.py::TestIcd10cm`.
- **Tests:**
  - Mock Kestrel with `monkeypatch.setattr("biomapper2.core.linker.kestrel_request", fake)` (`tests/test_equivalent_ids_lookup_status.py`).
  - Live tests use `pytestmark = [integration, requires_api]`.
  - `scripts/check.sh` runs ruff, black, pyright, then pytest.
- **Study precedent:** `studies/analysis/request_timeout.py` writes a committed JSON to `studies/analysis/results/`.
- **Deploy and packaging:** `.github/workflows/deploy-api.yml` runs from a git checkout (`uv sync`, editable install). The wheel packages only `src/biomapper2` (`pyproject.toml`), so the snapshot must live under `src/biomapper2/`. `.gitignore` ignores `*.tsv` and `data/**/*.json`, so pick an extension and negation that keep it tracked.

### Institutional Learnings

- `docs/solutions/integration-issues/slash-encoding-breaks-refmet-lipidmaps-name-lookups-2026-10-27.md`: frozen stores keep stale negatives after a code fix. The snapshot must carry a version and a rebuild script, and a fetch error must never be recorded as "no parent".
- `docs/solutions/best-practices/nginx-edge-rate-limit-for-sync-blocking-fastapi-2026-11-05.md`: prod is about 4.4 s/entity and rate-limited, so don't add round-trips (see the batching decision).
- Vault `docs/solutions/best-practices/fail-closed-guards-must-not-no-op-on-absent-input-2026-07-13.md`: a configured-but-missing snapshot must fail loudly, not quietly skip the walk-up.
- Vault `docs/solutions/integration-issues/hand-built-fixture-hid-a-live-production-bug-2026-09-23.md`: build `/canonicalize` fixtures from recorded real responses.
- Vault `docs/solutions/best-practices/benchmark-scorer-defects-under-credited-resolver-2026-07-26.md`: score with equivalence sets (cluster-level equality), not literal id equality. This applies to the control gate.
- `docs/solutions/integration-issues/human-gene-symbols-resolve-to-wrong-species-orthologs-2026-06-15.md`: report the fallback fraction by route.
- `docs/solutions/build-errors/equivalent-ids-prefix-mismatch-and-ci-type-error-2026-04-29.md`: verify prefix spellings against live Kestrel, and run pyright locally.

### External References

- CDC FY2027 ICD-10-CM `icd10cm-code-descriptions-2027.zip` (the order file), ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/
- Mondo SSSOM: purl.obolibrary.org/obo/mondo/mappings/mondo.sssom.tsv (2,099 `skos:exactMatch` ICD10CM rows, 78 broad; checked by Adil 2026-11-07)
- kraken#30, and Adil's comment of 2026-11-07

## Key Technical Decisions

- **Every candidate rides the existing `/canonicalize` batch, with no added sequential round-trip per row.** Before the first `get_kg_ids` call, expand each provided `ICD10CM:X` into its twin `ICD10:X` (1a), plus its existing-ancestor `ICD10CM:`/`ICD10:` forms (1b). That is at most about 11 extra curies per code. They all go in the batched call that already runs, and selection happens afterwards, per row. Total calls grow only by `ceil(added_candidates / KESTREL_BATCH_SIZE_CANONICALIZE)` (batch size 2000). Rationale: prod latency and rate limits. Every candidate is answered in the same response, which is what R7a needs, with one exception: see the bisect guard.
- **Bisect guard (R7a):** when `KESTREL_BISECT_ON_5XX_ENABLED` is on, `kestrel_request` drops isolated items instead of raising, so an absent candidate would no longer mean a definite null. The fallback selector must treat any isolated candidate of a row as unknown and raise for that row. It must never fall through to a broader ancestor. The flag is off by default, but this needs a test with it on.
- **Gate per row in the Linker, write back through a per-row overlay, and tier the evidence.**
  - Tiers: **direct > twin > walk-up**, evaluated per row against the *unmodified* batch result. A "direct" hit is any of the row's own curies (provided or annotator-assigned) linking under its own prefix; this covers `annotation_mode='all'`.
  - Only the highest tier present votes. If any curie in the row linked directly, no fallback is used (R5). If any `ICD10CM` code in the row has a twin hit, that row's walk-up hits are discarded. This settles rows with several `ICD10CM` codes: broad evidence never outvotes exact evidence.
  - Fallback nodes are written under the original `ICD10CM:X` key through a **per-row overlay**, a copy-on-write view of the cache built inside `_link_entity`, or an explicit `overrides` argument to `_format_kg_id_fields`. Never write into the shared `curie_to_kg_id_cache` from `_link_dataframe`: that would leak a fallback hit into later rows and make results depend on row order.
  - The route side-map is **per curie**. The row's `mapping_relation` / `icd10cm_route` are derived from the curies that back `chosen_kg_id`, and are set only when `chosen_kg_id` is a fallback node (taking the weakest relation among them).
  - Rationale: the smallest seam. It leaves `_reverse_curie_map`, `kg_ids_provided` and the resolver unchanged, and doesn't add `ICD10:` curies the user never supplied.
- **Errors propagate the same way primary-link failures do.** Because fallback candidates share the primary batch, a Kestrel failure raises exactly as it does today: per-row `error` in the entity/batch/stream routes, and a failed run on the dataset path. No new per-row error field. Rationale: parity with existing behavior, with no route-specific failure semantics to invent. This meets R7a's "never silently unresolved".
- **`mapping_relation` becomes an entity-general top-level field.** Add a top-level `mapping_relation` (+ `mapping_predicate`) to the row, the API result model and the TSV flat columns, plus `icd10cm_route` (`twin` | `walkup:<parent>`).
  - For lipid rows, the new field is filled from `lipid_resolution.mapping_relation`. `lipid_resolution` itself is unchanged.
  - The new fields are additive and null by default, following the `lipid_resolution` / `refmet_snapshot_version` precedent. The API does not use `exclude_none`, and must not start, because that would strip existing null fields. Invariant: **existing keys, columns and values are unchanged; the new keys and columns are null or empty on non-ICD, non-lipid rows.**
  - The certificate `extra_provenance` mirror gets a generalized parameter, set only when the field is set. It does not reuse the lipid one, which would also emit `ambiguous`. Certificates for non-ICD, non-lipid rows stay byte-identical.
  - Rationale: one broad/exact vocabulary across entity types (origin decision), and no break for lipid consumers.
- **The twin relation is a config value set from the gate result, bound to its artifact.** `ICD10CM_TWIN_MAPPING_RELATION` is `exact` or `broad`, set from the Unit 1 artifact against Trent's threshold.
  - A unit test reads `studies/analysis/results/icd10cm_twin_control.json` and the recorded threshold, and asserts the constant matches.
  - The KG build the gate ran on is stored next to the constant. If the live `/health` build differs, a warning is logged once per process, and the stamped `kg_build` makes the mismatch visible on every fallback row.
  - **Disagreement denylist:** codes the study lists as disagreeing are committed as a runtime list. Their twin hits are emitted as `broad`, not exact.
  - Rationale: R1b. It keeps the gate honest beyond merge day.
- **Switches:**
  - `ICD10CM_FALLBACK_ENABLED`: env-driven, default on, disables both twin and walk-up.
  - `ICD10CM_WALKUP_ENABLED`: default on, disables the walk-up only.
  - Both are env-driven like `RERESOLUTION_ENABLED`, so ops can flip them without a code change and tests can monkeypatch them. This meets the intent of origin R6 ("mirroring `GENE_SYMBOL_FALLBACK_ENABLED`", i.e. a no-revert off switch) in a form that is easier to operate.
- **The snapshot is a committed, derived file, plus a build script.**
  - The script fetches the CDC FY order zip, verifies a pinned SHA-256, and writes a trimmed code list. Each entry is a dotted code plus a header flag, a few hundred KB.
  - The file has a `.version` sidecar (FY + source SHA-256 + build date) and lives **inside the package** at `src/biomapper2/data/icd10cm/`. It is loaded via `importlib.resources`, with an env override `ICD10CM_HIERARCHY_PATH`.
  - Rationale: the wheel only packages `src/biomapper2`, and `PROJECT_ROOT` is wrong in wheel and git-dependency installs (e.g. KRAKEN). A repo-root file would make fail-loud init break every Mapper there. Packaging the file fixes this for every install type.
  - Fail loud: if the walk-up is enabled and the file is missing or malformed, the Mapper raises at init. A configured-but-broken state never degrades silently.
- **KG build provenance (R7):** every row the fallback served gets a `kg_build` stamp. It is fetched once per process from the same Kestrel `/health` call that `RunProvenance` uses (`src/biomapper2/provenance.py`), and cached. This is ICD-only scope, mirroring the snapshot-version stamp. Twin hits exist only because of a specific KG build (Babel#1033), so they are the results most in need of build provenance. Non-ICD per-row stamping stays deferred.
- **The control gate compares at the cluster level, not by literal ids.**
  - For each Mondo SSSOM exactMatch row `(MONDO:m, ICD10CM:X)` with no range target: agreement holds when `canonicalize(ICD10:X) == canonicalize(MONDO:m)`.
  - Rows whose twin is null are counted separately (no coverage). They are not counted as disagreement.

## Open Questions

### Resolved During Planning

- Which stage owns the retry? → The Linker, before the batch call (expansion) and after it (per-row selection).
- Where does the broad flag live? → An entity-general top-level `mapping_relation`, following the lipid precedent.
- How is the snapshot shipped? → A committed derived file plus a build script, with a pinned source SHA and a `.version` sidecar.
- How many extra round-trips? → None. Candidates ride the existing batch.
- How is the regression list built? → The Unit 1 study canonicalizes `ICD10:<code>` for **every** CDC code (no length filter, because Babel#1033 puts CM-only subcodes under `ICD10:`). Hits become the 1a fixture. This is stated honestly as "twin reproduces current `ICD10:` resolution on the pinned KG build". A v1.2.1 cross-check on a sample confirms the old rewrite was prefix-only, i.e. it produced `ICD10:<same code>`.

### Decided at Checkpoint 2 (Trent, 2026-11-08)

- **Gate evidence:** a provenance probe plus an independent title check. The title check compares the CDC CM title with the KRAKEN node label of the twin's clique, for every twin hit, and a random 50 mismatches are hand-reviewed. **The gate is title agreement.** Mondo SSSOM agreement is reported as Babel–Mondo consistency only, not as evidence.
- **Threshold:** ≥ 95% agreement on non-null rows, with n ≥ 200. Below n = 200 the gate reports `insufficient_n`, and twin stays `broad`.
- **Dispatch scope:** PR 1 only (Units 1–3). PR 2 (Units 4–5) is dispatched after Trent reviews the Unit 1 gate result.

### Deferred to Implementation

- Exact helper and field names.
- `.gitignore` negation vs a non-ignored extension for the packaged snapshot.
- How bisect reports isolated items to the caller (a return flag vs an exception), for the bisect guard.
- Re-resolution: `_certify_and_reresolve` only swaps on CONTRADICTED small-molecule certificates, but R4 lets ICD codes appear on any row. If a swap replaces a fallback node, clear `mapping_relation` / `icd10cm_route`, or recompute them against the committed node.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```mermaid
flowchart TD
    A[row curies after normalize] --> B{any ICD10CM:X provided?}
    B -- no --> L[link as today]
    B -- yes --> C[expand candidates per X:<br/>twin ICD10:X  (1a)<br/>+ existing ancestors as ICD10CM:/ICD10: (1b, snapshot)]
    C --> D[ONE batched /canonicalize<br/>original curies + all candidates]
    D -->|HTTP failure| E[raise: same as primary link failure<br/>row error / dataset run fails]
    D --> F{row has any direct hit?}
    F -- yes --> G[direct result; fallback ignored<br/>mapping_relation unset]
    F -- no --> H{twin hit?}
    H -- yes --> I[map under ICD10CM:X<br/>mapping_relation = TWIN_RELATION<br/>route = twin]
    H -- no --> J{most specific ancestor hit?}
    J -- yes --> K[map under ICD10CM:X<br/>mapping_relation = broad<br/>route = walkup:parent<br/>stamp snapshot version]
    J -- no --> M[unresolved]
```

Walk-up order for `ICD10CM:X`, most specific first:
1. `X` with its 7th character dropped (if it has 7 characters).
2. Successive truncations that exist in the snapshot, down to 3 characters. Placeholder-X forms that don't exist are skipped.
3. Each candidate is tried as `ICD10CM:` before `ICD10:`.

If `X` itself is not in the snapshot, it gets the twin retry only, and `icd10cm_route` records `twin_only:not_in_snapshot` on a miss.

## Implementation Units

### Phase 1a: twin retry (PR 1)

- [x] **Unit 1: Twin control gate and regression list (study)**

**Goal:** Produce the R1b agreement number and the 1a regression list as committed artifacts, before the twin relation is set.

**Requirements:** R1b, success criteria (regression list, control agreement reported, route labels)

**Dependencies:** None (read-only against public Kestrel). Can run first.

**Files:**
- Create: `studies/analysis/icd10cm_twin_control.py`
- Create: `studies/analysis/results/icd10cm_twin_control.json`
- Create: `tests/fixtures/icd10cm_twin_regression.json` (code → expected KG id/route, derived from the study)
- Test: `tests/test_icd10cm_twin_control.py` (unit tests of the pure comparison logic on tiny fixtures)

**Approach:**
- Inputs, pinned: the Mondo SSSOM TSV (URL plus SHA-256 recorded in the artifact), the CDC FY2027 order file (SHA-256), and the Kestrel base URL plus `/health` provenance captured at run time.
- **Gate (title agreement):** for every twin hit, compare the CDC CM title with the KRAKEN node label/synonyms of `canonicalize(ICD10:X)`. Normalize case, punctuation and "unspecified"/"NOS"-style boilerplate, and use a deterministic token-overlap score with a fixed cutoff recorded in the artifact. Report agree / disagree / no-label counts, and the rate on non-null rows. Write a random 50 of the mismatches (seeded) to a hand-review sheet, and record the adjudication in the artifact before the relation is set. Gate passes at ≥ 95% with n ≥ 200.
- **Consistency (reported, not gating):** for Mondo SSSOM exactMatch rows with non-range targets, `canonicalize(ICD10:X) == canonicalize(MONDO:m)`. Agree / disagree / twin-null counts.
- **Denylist:** codes that disagree on either check go into the runtime denylist, so their twin hits are emitted `broad`.
- Regression list: `ICD10:<code>` hits for every CDC code (~98k curies, ~50 batched calls). The headline is the count of codes the twin restores. Assert that every success-criteria code is in the canonicalized set.
- v1.2.1 cross-check: run the v1.2.1 normalizer, in an isolated env, over the success-criteria codes plus a random sample of 200. Confirm each `ICD10CM:X` was rewritten to `ICD10:X` with the same code, and record any difference.
- Provenance probe (circularity): for a sample of agreeing twin codes, record where the `ICD10:X` id's clique membership comes from (Kestrel node `equivalent_ids` / source). This tells us whether the Mondo comparison is independent evidence or a Babel–Mondo consistency check.
- Also emit route labels for the success-criteria codes (`E75.21`, `E83.01`, `F40.0`, `I10`, `E11`, `E11.65`, `F32.9`).
- The artifact is written by default to a timestamped path, plus a stable `results/` copy (SOP: expensive runs save by default).

**Patterns to follow:** `studies/analysis/request_timeout.py` (docstring rationale, `RESULTS_DIR`); `get_kg_ids` batching.

**Test scenarios:**
- Happy path (title): twin hit whose KRAKEN label matches the CDC title after normalization → agree.
- Edge case (title): label differs only by "unspecified"/NOS boilerplate → agree; a genuinely different concept → disagree, and the code lands in the denylist.
- Happy path (consistency): SSSOM row `(MONDO:1, ICD10CM:A00)` whose twin canonicalizes into MONDO:1's cluster → counted as consistent.
- Edge case: a range target `I10-I16` → excluded from the denominator.
- Edge case: twin canonicalizes to null → counted as twin-null, not as disagree.
- Edge case: a disease with two exact targets, parent and child (`K28`, `K28.0`) → both rows evaluated independently and listed.
- Edge case: fewer non-null rows than the minimum n → the gate reports `insufficient_n` rather than pass or fail.
- Happy path: the disagreement list is written as the runtime denylist file the Unit 3 config reads.
- Error path: a Kestrel HTTP failure mid-run → the run aborts with a non-zero exit and does not write a partial artifact as complete.

**Verification:** The artifact exists with pinned input hashes, the KG build, the title-match cutoff, the hand-review adjudication and the provenance probe. **Stop for Trent's review of the gate result before Unit 3 sets `ICD10CM_TWIN_MAPPING_RELATION`.** Until then, Unit 3 ships with `broad` as the safe default.

**Compute budget:** About 98k codes at batch size 2000 is ~50 calls, plus 2.1k SSSOM rows and their MONDO targets. Run a 1k-code timing sample first, and record the projection in the PR.

- [x] **Unit 2: Entity-general `mapping_relation` and route fields**

**Goal:** Add the output contract (`mapping_relation`, `mapping_predicate`, `icd10cm_route`) without changing any existing output.

**Requirements:** R1, R2 (contract), R7 (route)

**Dependencies:** None

**Files:**
- Modify: `src/biomapper2/core/resolver.py` (shared relation constants and predicate lookup, reused rather than duplicated)
- Modify: `src/biomapper2/mapper.py` (row fields; lipid rows fill `mapping_relation` from `lipid_resolution`; certificate mirror condition)
- Modify: `src/biomapper2/api/models/responses.py` (optional fields on the result model)
- Modify: `src/biomapper2/api/routes/mapping.py` (`extract_mapping_result`, and the NDJSON stream handler that builds its `result` dict by hand; plain str/None values)
- Modify: `src/biomapper2/core/certificate.py` (generalized `mapping_relation` provenance parameter)
- Test: `tests/test_mapping_relation_field.py`

**Approach:** All new fields default to None and are omitted when None, so non-lipid, non-ICD responses and certificates stay byte-identical. Lipid `lipid_resolution` is unchanged.

**Patterns to follow:** `LIPID_RESOLUTION_FIELDS` and the `lipid_flat_columns` flattening; the `_issue_certificate` `extra_provenance` gate.

**Test scenarios:**
- Happy path: a lipid row with `lipid_resolution.mapping_relation = broad` → the top-level `mapping_relation = broad`, the predicate is `skos:broadMatch`, and `lipid_resolution` is unchanged.
- Edge case: a plain metabolite row → `mapping_relation` / `mapping_predicate` / `icd10cm_route` / `kg_build` are null in the API JSON, every pre-existing key and value is unchanged, and the certificate dict equals the pre-change snapshot.
- Integration: dataset path → the TSV gains the new columns (empty on non-ICD rows), and existing columns keep their order and values.
- Integration: the NDJSON stream route emits the new keys and serializes without error.

**Verification:** The existing test suite passes. Any snapshot-comparison test that breaks only because of the additive null keys is updated deliberately, with the diff reviewed.

- [x] **Unit 3: Twin retry in the Linker**

**Goal:** For rows whose provided `ICD10CM:X` misses, resolve through `ICD10:X` in the same batch, gated per row.

**Requirements:** R1, R4, R5, R6 (`ICD10CM_FALLBACK_ENABLED`), R7, R7a

**Dependencies:** Unit 2; Unit 1 (for the `ICD10CM_TWIN_MAPPING_RELATION` value)

**Files:**
- Create: `src/biomapper2/core/icd10cm_fallback.py` (candidate expansion and per-row selection; pure functions)
- Modify: `src/biomapper2/core/linker.py` (expand before `get_kg_ids`; select and write back after; return the route side-map)
- Modify: `src/biomapper2/config.py` (`ICD10CM_FALLBACK_ENABLED`, `ICD10CM_TWIN_MAPPING_RELATION`)
- Modify: `src/biomapper2/mapper.py` (carry the route into the row and certificate)
- Test: `tests/test_icd10cm_fallback.py` (unit, mocked `kestrel_request` using recorded fixtures)
- Test: `tests/test_icd10cm_fallback_live.py` (`integration`, `requires_api`)

**Execution note:** Write the regression-fixture test first (from Unit 1), failing on current dev.

**Patterns to follow:** `tests/test_equivalent_ids_lookup_status.py` mocking; `RERESOLUTION_ENABLED` monkeypatching.

**Test scenarios:**
- Happy path: provided `ICD10CM:E75.21` only → `chosen_kg_id = MONDO:0010526`, `mapping_relation` = configured twin relation, `icd10cm_route = twin`, and `kg_ids_provided` is keyed by `ICD10CM:E75.21`.
- Happy path: `ICD10CM:I10` → `MONDO:0005044` via twin.
- Edge case (R5): a row with `ICD10CM:E75.21` plus a provided `MONDO:0000001` that links → the direct result wins, the fallback is ignored, `mapping_relation` is unset, and the vote is unchanged from today.
- Edge case (R4): an `ICD10CM:` id on a row with entity type `NamedThing` → the fallback still applies.
- Edge case: `ICD10CM:E11.65` (twin null) with the walk-up not yet built → unresolved, and the route records a twin miss.
- Edge case: a direct `ICD10CM:` hit (simulating a future KRAKEN with CM nodes) → used directly, no fallback, `mapping_relation` unset.
- Error path: `kestrel_request` raises `HTTPError` → entity path raises (the API reports a per-row `error`) and the dataset path fails the run. The row is never quietly unresolved.
- Switch: `ICD10CM_FALLBACK_ENABLED = False` → output identical to v1.3.0 for every case above.
- Edge case (shared cache): two rows sharing `ICD10CM:E75.21`, one of which also has a linking `MONDO:` curie, run in both row orders → the MONDO row uses the direct result with no fallback vote, the other row uses the twin, and output is independent of order.
- Edge case (denylist): a denylisted code with a twin hit → `mapping_relation = broad`.
- Edge case (bisect): bisect enabled and the row's twin candidate isolated as a poison item → the row raises; it is not unresolved and does not fall through.
- Happy path: every fallback row carries a `kg_build` stamp.
- Integration: call count = `ceil((direct + candidates) / KESTREL_BATCH_SIZE_CANONICALIZE)`, including one fixture that crosses a 2000-item chunk boundary.
- Unit: the config constant matches the committed gate artifact and threshold.
- Integration (live): the success-criteria codes against public Kestrel match the Unit 1 route labels.

**Verification:** Every code in the regression fixture resolves to its recorded node, non-ICD tests are unchanged, and `scripts/check.sh` is green.

### Phase 1b: walk-up (PR 2, after PR 1 merges)

- [ ] **Unit 4: CDC hierarchy snapshot, build script and loader**

**Goal:** A pinned, versioned ICD-10-CM code set with a parent lookup, which fails loudly when it is misconfigured.

**Requirements:** R3, R3b, R6 (`ICD10CM_WALKUP_ENABLED`)

**Dependencies:** None (parallel with Phase 1a work, but ships in PR 2)

**Files:**
- Create: `scripts/build_icd10cm_hierarchy.py` (download, SHA-256 verify, re-dot after the 3rd character, write trimmed file + `.version`)
- Create: `src/biomapper2/data/icd10cm/icd10cm_order_FY2027.<ext>` and `.version` (packaged)
- Create: `src/biomapper2/core/icd10cm_hierarchy.py` (loader: `is_present`, `version`, `contains`, `ancestors(code)`; cached on the resolved path, with `reset()`)
- Modify: `src/biomapper2/config.py` (`ICD10CM_HIERARCHY_PATH` env override, defaulting to the packaged resource; `ICD10CM_WALKUP_ENABLED`)
- Modify: `pyproject.toml` (only if hatchling needs an explicit include for the data file)
- Modify: `.gitignore` if needed
- Test: `tests/test_icd10cm_hierarchy.py` (with a small fixture snapshot in `tests/fixtures/`)

**Approach:** Ancestors are generated most-specific-first:
- Drop the 7th character.
- Then truncate one character at a time, keeping only codes present in the snapshot.
- Stop at 3 characters.

A code absent from the snapshot returns no ancestors. If the walk-up is enabled and the file is missing or malformed, raise at Mapper init.

**Patterns to follow:** `core/annotators/refmet_snapshot.py` (loader shape, reject-whole on malformed input); `config.derive_refmet_snapshot_version`.

**Test scenarios:**
- Happy path: `E11.65` → `[E11.6, E11]`.
- Happy path: `F32.9` → `[F32]`.
- Edge case: `T36.0X1A` → drop `A` → `T36.0X1`, then skip non-existent `T36.0X` → `T36.0` → `T36`. Only codes present in the fixture are returned.
- Edge case: `S72.001A` → `[S72.001, S72.00, S72.0, S72]`, filtered to those present.
- Edge case: a 3-character code `I10` → `[]`.
- Edge case: a code not in the snapshot → `[]`, and `contains` is False.
- Edge case: the build script re-dots the undotted `E1165` → `E11.65`, and leaves `I10` as `I10`.
- Error path: walk-up enabled and the path is missing → Mapper init raises a clear error.
- Error path: a malformed file (bad header or row) → rejected whole, and init raises.
- Error path: the build script sees a SHA-256 mismatch → it aborts without writing.
- Integration (packaging): build the wheel and install it into a temp env outside the repo → the snapshot loads via `importlib.resources`, and Mapper init succeeds.

**Verification:** The committed snapshot loads, the version string includes FY and SHA, and all tests pass.

- [ ] **Unit 5: Walk-up in the fallback**

**Goal:** When the twin misses, commit the most specific ancestor hit as `mapping_relation = broad`.

**Requirements:** R2, R3, R3b, R5, R7, R7a

**Dependencies:** Units 3 and 4

**Files:**
- Modify: `src/biomapper2/core/icd10cm_fallback.py` (ancestor candidates in the expansion; selection order)
- Modify: `src/biomapper2/mapper.py` (stamp the snapshot version only on walk-up rows, following `_refmet_snapshot_version_for`)
- Test: `tests/test_icd10cm_fallback.py` (extend), `tests/test_icd10cm_fallback_live.py` (extend)

**Test scenarios:**
- Happy path: `ICD10CM:E11.65` → `MONDO:0005148`, `mapping_relation = broad`, predicate `skos:broadMatch`, `icd10cm_route = walkup:E11`, snapshot version stamped.
- Happy path: `ICD10CM:F32.9` → `MONDO:0002009` broad via `F32`.
- Edge case: ancestor priority. Mocked hits on both `E11.6` and `E11` → `E11.6` is chosen.
- Edge case: within one level, an `ICD10CM:E11.6` hit is preferred over `ICD10:E11.6`.
- Edge case: a twin hit present → the walk-up is never used (relation = twin relation, no snapshot stamp).
- Edge case (R5): a direct hit on another curie in the row → the walk-up is ignored.
- Edge case: a code not in the snapshot → twin only, and the route records `twin_only:not_in_snapshot`.
- Edge case (multi-code row): `ICD10CM:E75.21` (twin hit) plus `ICD10CM:E11.65` (walk-up only) → the twin tier wins, the walk-up is discarded, and the relation and route are the twin's.
- Edge case (multi-code, walk-up only): two codes whose walk-ups reach different nodes → normal vote between them; relation `broad`; the route reflects the curie(s) backing the chosen node.
- Switch: `ICD10CM_WALKUP_ENABLED = False` with the fallback on → behavior identical to PR 1.
- Integration: call count follows the `ceil` formula with the ancestor candidates included.
- Integration (live): E11.65 and F32.9 against public Kestrel.

**Verification:** The success-criteria codes match their labeled routes, the PR 1 regression fixture is still green, and non-ICD tests are unchanged.

## System-Wide Impact

- **Interaction graph:** Linker input is expanded only for rows carrying `ICD10CM:`. Resolver voting is untouched, because fallback hits are keyed under the original curie. Certificates gain fields only on ICD and lipid rows.
- **Error propagation:** Unchanged. A Kestrel failure raises the same way it does today. The bisect guard keeps silently isolated candidates from becoming false nulls.
- **State lifecycle risks:** requests-cache stores HTTP responses, not interpretations, so there are no new cached negatives. The RefMet/Tier B stores are not touched.
- **API surface parity:** New optional result fields reach the API JSON, the NDJSON stream and the TSV. The `biomapper` client and the MCP need follow-up work to surface them (separate task). Until then, consumers see `resolved` with no broad flag, which is the risk the scorer task addresses.
- **Unchanged invariants:**
  - Non-ICD rows: same `chosen_kg_id`, same certificate bytes, same call count. API/TSV output gains null new keys and columns only.
  - `lipid_resolution` shape.
  - `CATEGORY_PREFERRED_NAMESPACES`.
  - Normalizer behavior for `ICD10CM` (#141).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| WHO/CM meaning drift on twin hits | Unit 1 gate (evidence source is a Checkpoint 2 decision because of possible circularity); runtime denylist for disagreeing codes; relation bound to the artifact by a test |
| Gate circularity: Babel may have built CM-shaped `ICD10:` cliques from Mondo xrefs | Provenance probe in Unit 1; independent evidence per the Checkpoint 2 decision |
| Walk-up via `ICD10:` ancestors carries a WHO meaning, not only a broader one | Within each level, `ICD10CM:` forms are tried first; `broad` is documented as "less specific, meaning not verified" |
| Consumers read `resolved` and ignore `mapping_relation` | Field present on every ICD fallback row; client and scorer split as a tracked follow-up; note in release notes |
| Babel#1033 fixed upstream removes the CM-shaped `ICD10:` ids | Unit 1 study is rerunnable; the post-kraken#30 usage check catches coverage loss |
| Snapshot goes stale (new FY) | Build script plus pinned SHA; FY in the version stamp; a refresh is one script run and a PR |
| Walk-up adds KG coverage that later conflicts with KRAKEN CM nodes | Direct hits always win (R5); a walk-up hit on a code KRAKEN later covers becomes a direct hit automatically |
| Prod rate limits | No added sequential round-trips per row; extra chunks only past 2000 curies, by the asserted `ceil` formula |

## Documentation / Operational Notes

- Add one paragraph in the README or `docs/` on ICD-10-CM handling: routes, `mapping_relation`, the two switches, and snapshot refresh.
- Release notes for PR 1: "restores ICD10CM resolution lost in 1.3.0".
- PR routing per repo convention:
  - Open on fork `trentleslie/biomapper2`, base `dev`, for Greptile first, then promote to Phenome-Health `dev` → `main`.
  - The release-please PR needs the bot CI run approved.
- After deploy, run the Unit 1 success-criteria codes against the prod API.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-11-08-icd10cm-fallback-requirements.md](docs/brainstorms/2026-11-08-icd10cm-fallback-requirements.md)
- Related code: `src/biomapper2/core/linker.py`, `src/biomapper2/core/resolver.py`, `src/biomapper2/mapper.py`, `src/biomapper2/core/annotators/refmet_snapshot.py`
- Related PRs/issues: Phenome-Health/biomapper2#141, #142, #143; Phenome-Health/kraken#30; NCATSTranslator/Babel#1033
