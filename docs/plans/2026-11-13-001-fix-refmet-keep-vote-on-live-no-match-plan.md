---
title: "fix: keep the last-known-good RefMet vote when live /match turns into a no-match"
type: fix
status: active
date: 2026-11-13
origin: docs/brainstorms/2026-11-13-refmet-keep-vote-on-live-no-match-requirements.md
---

# fix: keep the last-known-good RefMet vote when live /match turns into a no-match

## Overview

In `live_backup` mode the RefMet annotator will stop letting a live no-match replace a stored vote. It returns the stored vote under a new source label, tracks what live last said separately from the vote, records each change of the live answer once, and ships a dry-run-first repair for store rows that were already overwritten. Fixes Phenome-Health/biomapper2#159.

## Problem Frame

Metabolomics Workbench changed `/match` without notice around 2026-11-12: Metabolon fatty-acid names with a shorthand suffix now return a no-match. Prod treats that as authoritative, drops the RefMet vote, overwrites the store's last-known-good id, and records nothing. On NECS this put n3 DPA on the n6 DPA node and DHA on a node with no structure. The old RefMet ids still exist upstream, so this is name-matching drift, not retirement (see origin: `docs/brainstorms/2026-11-13-refmet-keep-vote-on-live-no-match-requirements.md`).

## Requirements Trace

All from the origin document.

- R1-R3. Keep the stored vote against a live no-match only; live match with a different id wins; unreachable and never-voted cases unchanged.
- R4. A live no-match never overwrites a stored vote; the store is the only runtime source of last-known-good.
- R5-R7. Distinct source label with availability `voted`; no review flag or certificate-state change; kept votes appear in the per-batch source counts.
- R8-R11. Last observed live answer stored separately (as columns on the same row); each change recorded once, across workers; one documented listing command; history failures never fail a mapping.
- R12-R14. Repair command: exact-query comparison with a freeze file, dry run by default, apply restores rows and writes history; run by an operator on prod after deploy.
- R15. Editing or deleting a store row clears a kept vote.
- R16. `frozen` and `off` modes unchanged.

## Scope Boundaries

- No review flag, no new certificate state, no runtime freeze fallback in `live_backup`, no expiry of kept votes, no change to the 7-day HTTP cache, no change to `frozen` or `off`, no client change (see origin).
- The canonical-first then raw-spelling query order in `_fetch_refmet_data` is unchanged. A kept vote on the canonical query returns `voted`, so the raw spelling is not tried, exactly as before the drift.

### Deferred to Separate Tasks

- Scheduled canary over the frozen corpus, and a check that a kept id still exists in RefMet: follow-up issues.
- Blocking mapping handlers that take prod dark when Kestrel is slow: separate issue.
- Extending the freeze file to cover every benchmark input: separate reproducibility task.
- Running the repair on prod and the NECS verification: operator steps after release, listed under Operational Notes. They are not part of the PR.

## Context & Research

### Relevant Code and Patterns

- `src/biomapper2/core/annotators/metabolomics_workbench.py`: `_fetch_live_backup` (the write-back and outage path), `_result_from_store_hit`, `_fetch_live`, `_dispatch_refmet`, `_fetch_refmet_data` (canonical-first dispatch).
- `src/biomapper2/core/annotators/refmet_store.py`: `refmet_by_query` keyed on the exact query, short-lived connection per call, WAL plus `busy_timeout`, `get`, `upsert`, `seed_from_tsv` (reads the freeze TSV by exact `query_name`), `StoreHit`. The module docstring explains why keys are never case-folded and why the legacy table is left in place for rollback.
- `src/biomapper2/core/annotators/base.py`: `REFMET_SOURCE_*` constants and the comment block that documents each value.
- `src/biomapper2/mapper.py`: `_SNAPSHOT_ATTRIBUTABLE_SOURCES` and the `refmet_source_counts` tally, which counts whatever source string appears.
- `src/biomapper2/api/models/responses.py`: `refmet_source` is a plain string with a description listing the values.
- Tests to mirror: `tests/test_metabolomics_workbench_live_backup.py` (the `store_db` fixture, stubbed `_session`, one behavior per test), `tests/test_refmet_store.py`, `tests/test_refmet_availability_plumbing.py`.
- `scripts/` holds standalone Python entry points run with `uv run`; `scripts/check.sh` runs ruff, pyright (including `tests/`) and the fast tests.

### Institutional Learnings

- Unit tests for this annotator must stub `_session`. Patching `requests.get` silently hits the live API (found during the v1.4.0 `/match` fix).
- pyright runs on `tests/`: a monkeypatched lambda must use the same parameter names as the method it replaces.
- `refmet_snapshot.lookup` falls back to a case-folded key. It must not be used for anything that decides a store row.
- The store was re-keyed once already; rollback safety (old release still works against the new file) was a stated requirement then and applies here.

## Key Technical Decisions

- **Source label `kept_live_no_match`** (`REFMET_SOURCE_KEPT_LIVE_NO_MATCH` in `base.py`). It says what happened and cannot be confused with `freeze_backup`, which means live was down.
- **Last observed live answer lives on the same row**, as added columns (`live_status`, `live_refmet_id`, `live_changed_at`) on `refmet_by_query`. `live_changed_at` is the time the live answer last changed, not the time it was last checked. The columns are added on connect if absent: read the table's column list, add each missing column, and treat a "duplicate column name" error as success, because the other worker may have added it first (SQLite has no add-if-absent). The previous release names its columns explicitly when it reads and writes, so it keeps working against a migrated file, which preserves rollback.
- **A row with no recorded live answer is treated as if live last said what the row says.** Before this change the store was overwritten with every live answer, so `(status, refmet_id)` was the last live answer. This gives the right "old" value the first time a pre-existing vote meets a live no-match. For rows seeded from the freeze or restored by repair it is an assumption, not an observation, so each log row records whether its old value was observed (from the live columns) or assumed (from the row baseline).
- **History is an append-only table `refmet_change_log`** in the same SQLite file. One store function applies a live answer in a single write transaction: it updates the live-answer columns only if they differ (the comparison is in the update's own condition, not a read done earlier), inserts the log row only when that update changed a row, and in the same transaction writes the vote or no-match to the row where the matrix says to. Doing the log and the write-back together is required: with empty live columns the baseline is the row itself, so a write-back done first would make the change invisible and `id_changed` and `match_appeared` would never be logged. Two workers racing on the same query produce one log row. The same function refuses to replace a stored vote that has an id with a no-match, so R4 is enforced in SQL and not only by the annotator's read.
- **Change kinds** written to the log: `vote_kept` (live went to no-match while a vote is stored), `live_recovered` (a kept vote's query matches again), `id_changed`, `match_appeared` (no-match to match), `repair`. A stored no-match staying a no-match, or a vote confirmed with the same id, writes nothing.
- **Store read failure with a live no-match returns the live no-match and writes nothing.** Without the row the annotator cannot know whether a vote exists, and it must not overwrite one blind.
- **A seeded row is a last-known-good vote like any other** (the code makes no distinction today).
- **Snapshot version is unchanged in shape.** The kept result carries the stored row's snapshot version on the annotator-level result the way `_result_from_store_hit` does, but the mapper stamps the certificate's `refmet_snapshot_version` by source label alone, and `_SNAPSHOT_ATTRIBUTABLE_SOURCES` is not extended. A kept row's certificate value is therefore empty, the same as a `freeze_backup` row today.
- **The annotator reads the stored row through the existing `refmet_store.get`** (with `StoreHit` gaining defaulted live-answer fields), so existing tests that monkeypatch `get` keep intercepting the read. When the row's live-answer fields already equal the incoming live answer and no write-back is due, the annotator skips the store write entirely, preserving today's no-write-on-identical-hit rule.
- **A stored `unavailable` row, or a `voted` row with no id, is treated as no stored row**: the live answer is written as today and nothing is logged.
- **Admin entry point is `scripts/refmet_store_admin.py`** with `history` and `repair` subcommands. The logic lives in `refmet_store.py` functions so it is unit-tested without a subprocess.
- **Repair reads the freeze TSV directly by exact `query_name`**, the way `seed_from_tsv` does, never through `refmet_snapshot.lookup`.
- **After a repair the row's live answer stays `no_match`**, so the next request keeps the vote without writing a second `vote_kept` record. A repaired row carries `source=local_snapshot` and the freeze's version, as a seeded row would; the `repair` log row is what distinguishes it.
- **Read-only admin paths never create or migrate.** `history` and `repair` without apply open the store read-only and fail if the file does not exist, so a mis-resolved store path cannot produce a fresh empty store and a falsely clean dry run.
- **Store errors at annotator construction are best-effort too.** The seeding call in the annotator's constructor is wrapped so a store or migration error is logged and ignored instead of failing mapper start-up for that worker.

## Open Questions

### Resolved During Planning

- Source label string: `kept_live_no_match`.
- Where history lives and how "once" is enforced: a table in the store, written in the same transaction as the conditional live-answer update.
- Command form: `scripts/refmet_store_admin.py history|repair`.
- Seeded rows count as votes: yes.
- Snapshot-version attribution set: unchanged.

### Deferred to Implementation

- Exact helper names and the SQL text.
- Whether any existing test asserts the full set of `refmet_source` keys or the exact store schema; adjust only those that assert the old overwrite or an exact column list.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

Decision matrix for `live_backup`, per exact query:

| Live answer | Stored row | Result returned | Store vote | Log |
|---|---|---|---|---|
| no-match | vote with id | stored vote, source `kept_live_no_match` | untouched | `vote_kept`, first time only |
| no-match | no-match, or none | no-match, source `live_api` | no-match written (as today) | none |
| match, same id | vote | live vote, source `live_api` | untouched | `live_recovered` only if live last said no-match |
| match, different id | vote | live vote, source `live_api` | replaced with live | `id_changed` |
| match | no-match, or none | live vote, source `live_api` | vote written | `match_appeared` if a no-match row existed |
| match with no id | any | live result, store untouched (existing guard) | untouched | none |
| unreachable | vote or no-match | stored value, source `freeze_backup` (as today) | untouched | none |
| unreachable | none | unavailable (as today) | untouched | none |
| no-match | store read fails | no-match, source `live_api` | not written | none |
| match | store read fails | live vote, source `live_api` | not written (as today) | none |
| any | `unavailable` row, or `voted` row with no id | as for no stored row | live answer written | none |

## Implementation Units

- [x] **Unit 1: Store tracks the last live answer and logs changes**

**Goal:** Give the store a per-query record of what live last said, an append-only change log, and a way to list it, without changing what `get` and `upsert` do for existing callers.

**Requirements:** R4, R8, R9, R10 (listing function), R11

**Dependencies:** None

**Files:**
- Modify: `src/biomapper2/core/annotators/refmet_store.py`
- Test: `tests/test_refmet_store.py`

**Approach:**
- Add the three live-answer columns when a connection finds them missing (column-list check, add, tolerate the duplicate-column error from a concurrent worker) and create `refmet_change_log` if absent.
- Each `refmet_change_log` row stores the query, the old live status and id, the new live status and id, the change kind, whether the old value was observed or assumed, and a UTC timestamp.
- Add the single function described in Key Technical Decisions that applies a live answer in one write transaction: conditional live-column update, log row only when that update changed a row, and the row write-back where due, never replacing a vote that has an id with a no-match. It returns whether a change was recorded.
- `StoreHit` gains defaulted live-answer fields, filled by `get`.
- Add a function that lists log rows, newest first, with an optional limit.
- Keep the module's best-effort posture: callers catch and log; these functions do not swallow errors themselves beyond what `get` and `upsert` already do.
- Update the module docstring: the new columns, the log table, and that a stored vote is no longer overwritten by a live no-match.

**Execution note:** Implement test-first.

**Patterns to follow:**
- Short-lived connection per call, WAL and `busy_timeout`, as in `get` and `upsert`.
- The docstring style of the existing re-key explanation.

**Test scenarios:**
- Happy path: a fresh store gets the new columns and the log table; `get` and `upsert` behave as before.
- Happy path: recording a no-match observation against a voted row with empty live columns writes one log row whose old value is the row's vote and whose new value is no-match, and leaves the row's status and id untouched.
- Edge case: recording the same observation again writes no second log row and returns "no change".
- Edge case: recording a match with the same id after a no-match observation writes one log row and updates the live columns.
- Edge case: a store file created by the current release (no live columns, no log table) is migrated on first connect and its rows are intact.
- Edge case: recording an observation for a query with no row writes no log row and does not create a vote.
- Integration: two connections record the same new observation one after the other; exactly one log row exists.
- Integration: two threads open the same pre-migration store file at the same time; neither raises and each new column exists exactly once.
- Happy path: a log row holds the query, old and new status and id, kind, the observed-or-assumed marker and a UTC timestamp; the marker is "assumed" when the live columns were empty and "observed" otherwise.
- Edge case: applying a no-match to a row holding a vote with an id leaves the vote in place even when called without the annotator's prior read.
- Edge case: applying a match with a different id to a row with empty live columns writes exactly one `id_changed` row and replaces the vote; applying a match to a stored no-match with empty live columns writes exactly one `match_appeared` row.
- Error path: with no store configured, recording and listing are no-ops that return empty results.
- Happy path: listing returns rows newest first and honors the limit.

**Verification:**
- The store tests pass, and a store file written by the new code is still readable by the unchanged `get` of the current release (column names it selects are all present).

- [x] **Unit 2: Annotator keeps the vote against a live no-match**

**Goal:** Implement the decision matrix in `_fetch_live_backup` and add the source label.

**Requirements:** R1, R2, R3, R4, R5, R6, R11, R15, R16

**Dependencies:** Unit 1

**Files:**
- Modify: `src/biomapper2/core/annotators/metabolomics_workbench.py`
- Modify: `src/biomapper2/core/annotators/base.py`
- Modify: `src/biomapper2/api/models/responses.py` (the `refmet_source` field descriptions only)
- Modify: `src/biomapper2/core/certificate.py` (comments and docstrings that list the `refmet_source` values only)
- Test: `tests/test_metabolomics_workbench_live_backup.py`

**Approach:**
- Add `REFMET_SOURCE_KEPT_LIVE_NO_MATCH` and document it in the constants comment block.
- In `_fetch_live_backup`, read the stored row before deciding. On a live no-match with a stored vote that has an id: record the observation with kind `vote_kept`, do not upsert, and return a `voted` result built from the stored row with the new source label. On a live no-match with no stored vote: behave as today.
- On a live match: call the single store function from Unit 1 so the log row and the write-back happen together, with the kind from the matrix. Do not upsert first. The existing guard for a match without an id stays first.
- When the stored row's live-answer fields already equal the live answer and no write-back is due, make no store write.
- Wrap the constructor's seeding call so a store error there is logged and ignored.
- A store read failure on a live no-match returns the live no-match and skips the write. Any failure recording an observation is logged and ignored.
- Log one warning line when a vote is kept for the first time for a query (when the observation recorded a change), not on every request.
- Do not touch `_fetch_from_snapshot`, `_fetch_live`, `_dispatch_refmet` or `_fetch_refmet_data`.

**Execution note:** Implement test-first. The existing test `test_live_no_match_is_written_back_as_negative_cache` stays valid for a query with no stored vote; add the new cases beside it.

**Patterns to follow:**
- `_result_from_store_hit` for building a result from a store row.
- The `store_db` fixture and stubbed `_session` in the existing live_backup tests.

**Test scenarios:**
- Happy path: stored vote `RM0153631`, live no-match -> result is `voted` with that id and source `kept_live_no_match`; the stored row still holds the vote; one `vote_kept` log row.
- Edge case: the same request again -> same result, still one log row.
- Happy path: after a kept vote, live matches with the same id -> source `live_api`, one `live_recovered` log row.
- Happy path: stored vote, live match with a different id -> live id returned with source `live_api`, store row replaced, one `id_changed` log row.
- Happy path: no stored row, live no-match -> no-match with source `live_api`, a no-match row written, no log row.
- Happy path: stored no-match, live match -> vote returned and written, one `match_appeared` log row.
- Edge case: stored vote, live match with no id -> existing behavior, store untouched, no log row.
- Error path: live unreachable with a stored vote -> source `freeze_backup`, unchanged, no log row.
- Error path: live no-match and the store read raises -> no-match with source `live_api`, nothing written, mapping does not raise.
- Error path: stored vote, live no-match, recording the observation raises -> the kept vote is still returned.
- Edge case: a stored row that differs from the query only by case is not used (the store key is exact).
- Edge case: after the stored vote's row is deleted, a live no-match returns no-match with source `live_api`, writes a no-match row and no `vote_kept` record (R15).
- Edge case: a stored `unavailable` row and a stored `voted` row with no id are each treated as no stored row.
- Error path: live match and the store read raises -> live vote with source `live_api`, nothing written (the existing best-effort test keeps passing).
- Error path: the store raises while the annotator is constructed -> construction succeeds and a warning is logged.
- Edge case: a second identical live match for a row whose live fields already agree makes no store write.
- Regression: mode `frozen` and mode `off` results for the same inputs are identical to before (existing tests keep passing unchanged).

**Verification:**
- The live_backup tests pass, including every pre-existing test without edits to its expectations, and `scripts/check.sh` is clean.

- [x] **Unit 3: History listing and one-time repair command**

**Goal:** Let an operator list recorded changes and restore store rows that contradict a freeze file, dry run by default.

**Requirements:** R10, R12, R13, R14

**Dependencies:** Unit 1, Unit 2

**Files:**
- Modify: `src/biomapper2/core/annotators/refmet_store.py` (find contradictions, restore rows, consistent backup)
- Create: `scripts/refmet_store_admin.py`
- Test: `tests/test_refmet_store_admin.py`

**Approach:**
- A function reads a freeze TSV by exact `query_name` and returns every query where the freeze row is a vote with an id and the store row is a no-match, with the store row's `updated_at`. No case or whitespace folding.
- A function restores a given list of queries to the freeze's vote, sets each row's live answer to no-match, and writes one `repair` log row per query, in one transaction.
- The script exposes `history [--limit N]`, `repair --freeze PATH [--since DATE] [--apply --expect-count N]` and `backup DEST`. Without `--apply`, `repair` prints the contradiction list and exits without writing. `--since` limits the list to rows last updated on or after a date. `--apply` recomputes the list, refuses to proceed unless its size equals `--expect-count` (the size the operator reviewed), and restores only rows that are still a no-match at write time.
- `history` and the dry run open the store read-only and never create or migrate it; all subcommands exit non-zero when the store file does not exist. Each prints the resolved store path, its row count, and the freeze used.
- `backup` writes a consistent copy with SQLite's online backup, which is safe on a live WAL database where a plain file copy is not.
- The script prints plain tab-separated lines so the output can be saved next to a run.

**Execution note:** Implement test-first against temporary store and freeze files.

**Patterns to follow:**
- `seed_from_tsv` for reading the freeze TSV and deriving its version.
- Existing `scripts/*.py` argument handling.

**Test scenarios:**
- Happy path: store no-match plus freeze vote for the same exact query -> listed by the dry run, with the row's update time; the store is unchanged afterwards.
- Happy path: apply restores the row to the freeze's id, sets its live answer to no-match, and writes one `repair` log row.
- Edge case: a freeze vote whose query differs from the store row only by case is not listed.
- Edge case: a freeze no-match, a freeze vote with no id, and a store row that already votes are not listed.
- Edge case: apply with nothing to repair writes nothing and reports zero.
- Error path: a freeze path that cannot be read, or a TSV without the `query_name` and `status` columns, exits non-zero with a message and writes nothing.
- Error path: no store configured, or a configured store file that does not exist, exits non-zero with a message and creates no file.
- Edge case: the dry run and `history` leave the store file byte-for-byte unchanged, including on a pre-migration file.
- Edge case: `--since` excludes a contradicting row last updated before the date.
- Error path: `--apply` with an `--expect-count` that differs from the recomputed list writes nothing and exits non-zero.
- Edge case: a row that stopped being a no-match between the dry run and the apply is not overwritten.
- Happy path: `backup` produces a file that opens as a store with the same rows.
- Integration: after an apply, a live no-match for the repaired query returns the kept vote (Unit 2) and adds no second log row.
- Happy path: `history` prints recorded changes newest first.

**Verification:**
- The admin tests pass; a dry run against a copy of a store containing the five known contradictions lists exactly those five.

- [x] **Unit 4: Plumbing checks and documentation**

**Goal:** Prove the new label flows through to the certificate and batch counts unchanged in shape, and document the behavior.

**Requirements:** R5, R6, R7

**Dependencies:** Unit 2

**Files:**
- Test: `tests/test_refmet_availability_plumbing.py`
- Create: `docs/brainstorms/2026-11-13-refmet-keep-vote-on-live-no-match-requirements.md` (the approved requirements; not on `origin/dev` yet, so add it to the branch by copying it from the planning worktree, the directory that holds this plan file)
- Create: `docs/plans/2026-11-13-001-fix-refmet-keep-vote-on-live-no-match-plan.md` (this plan, added to the branch the same way)
- Modify: `docs/brainstorms/2026-09-09-write-through-refmet-freeze-spec.md` (one dated note under D6 pointing to this change)
- Modify: `README.md` (add a short section on RefMet freeze modes in `live_backup`: the kept-vote behavior, the label, and the admin script; the README does not mention `REFMET_FREEZE_MODE` today)

**Approach:**
- Add plumbing tests only; no production change is expected in `mapper.py` or `certificate.py`. If a test shows the label is dropped or rejected somewhere, fix that spot minimally and note it in the PR.
- The note under D6 states that a live no-match is authoritative only for queries with no stored vote, with the date and the issue number.

**Test scenarios:**
- Integration: a mapped row whose RefMet result is a kept vote has `refmet_availability` `voted` and `refmet_source` `kept_live_no_match` on the result and in the certificate, raises no `chosen_kg_id_review` flag on its own, and resolves to the same node as the same row with an ordinary live vote.
- Integration: a batch containing one kept vote reports it under `kept_live_no_match` in `refmet_source_counts`.
- Edge case: the kept row's certificate `refmet_snapshot_version` is empty, the same as a `freeze_backup` row, because the mapper stamps it by source label and the attributable set is unchanged.

**Verification:**
- The plumbing tests pass and the full fast suite is green.

## System-Wide Impact

- **Interaction graph:** only the `live_backup` branch of the RefMet annotator and the store module change. The resolver sees a normal RefMet vote. The certificate and API carry a new string value in an existing string field.
- **Error propagation:** every store and history operation on the mapping path stays best-effort: logged and ignored, never raised into a mapping.
- **State lifecycle risks:** schema migration on a live prod file with two workers starting together (idempotent column adds and `CREATE TABLE IF NOT EXISTS`, under `busy_timeout`); rollback to the previous release against a migrated file (extra columns and table are ignored by it). The previous release, if rolled back, would resume overwriting votes; the change log would then be stale but harmless.
- **API surface parity:** `refmet_source` gains one value. The Python client passes it through as a string. No request or response field is added.
- **Integration coverage:** the kept vote reaching the resolver, certificate and batch counts (Unit 4); repair followed by a live no-match (Unit 3).
- **Unchanged invariants:** `frozen` and `off` modes, `REFMET_LIVE_API_FALLBACK`, the canonical-first query order, the HTTP cache, the breaker, the outage path's `freeze_backup` label, exact-query keying, and the rule that a match without an id never touches the store.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| A kept vote is stale because RefMet really retired the entry | Label, source counts and change log make it visible; deleting the store row clears it (R15). Canary and by-id check are named follow-ups. |
| Detection becomes pull-only once numbers stop moving | Kept votes appear in every batch's source counts and a warning is logged the first time each vote is kept. The canary follow-up closes the gap. |
| Migration misbehaves on the prod store | Additive only, tolerant of two workers migrating at once, tested against a file created by the current release, and a store error at start-up no longer fails the worker. Operator takes a consistent backup before deploy. |
| A cached live no-match delays or masks changes for up to 7 days | Accepted and documented; out of scope. Kept-vote counts can keep rising for a week after deploy as cached matches expire, with answers unchanged. |
| Prod keeps overwriting stored votes until this ships, so the repair list can grow past the five known rows, and votes for queries outside the freeze are lost for good | Take a consistent backup of the prod store now, before more rows are overwritten. The repair uses a per-row rule, not a fixed count (see Operational Notes). |
| A kept vote on the canonical spelling means the raw spelling is never queried, even if it now matches live | Accepted limitation, same as before the drift. None of the five known queries has a canonical form. |
| A live answer that flaps writes a log row per flip | Accepted. The 7-day cache bounds the rate per query. |
| The admin script run as the wrong user, or against the wrong store path, breaks or misses the service's store | Read-only paths never create files; the script prints the resolved path and row count; the operator runs it as the service user and checks the service log for store warnings afterwards. |
| The 20 no-match rows written outside the freeze since 2026-11-10 may hide lost votes | Not repairable from the store. Listed in the origin document; can be checked against earlier benchmark artifacts separately. |
| Kept votes change prod answers relative to the 2026-11-12 run | Intended. The preprint reruns must happen after this is deployed and the repair applied, and should report the kept-vote count. |

## Documentation / Operational Notes

- Release path: PR to the fork's `dev` for Greptile, then the org `dev` and `main` promotion PRs, then the release PR. No direct deploys.
- Before deploy: take a consistent backup of the prod store (now, and again just before the release). Until the admin script is on prod, use SQLite's online backup from Python, not a plain file copy.
- After the release is on prod, as the service user: run the repair dry run with the prod freeze file and `--since 2026-11-12`. Review every listed row: apply it only if its freeze id still resolves in RefMet by id. The five known queries are the expected minimum; more rows are possible. Rows updated before 2026-11-12, or whose id no longer resolves, are left alone and reported. Then run with `--apply --expect-count N`, save both outputs with the run artifacts, and confirm the next mapped request logs no store warning.
- Verification on prod: `docosahexaenoate (dha; 22:6n3)` and `docosapentaenoate (n3 dpa; 22:5n3)` return RefMet `voted` and commit RM:0153631 and RM:0153164. The source is `kept_live_no_match` while live still says no-match; `live_api` with the same id is also a pass, since it means Metabolomics Workbench reverted the drift.
- Conventional commit type `fix(refmet)` so the release tooling cuts a patch release.

## Premises

| kind | subject | expected | check |
|---|---|---|---|
| ref | `origin/dev` of the repo | exists | `git -C <repo> rev-parse --verify origin/dev` |
| path | `src/biomapper2/core/annotators/refmet_store.py` | exists | stat |
| path | `src/biomapper2/core/annotators/metabolomics_workbench.py` | exists | stat |
| path | `tests/test_metabolomics_workbench_live_backup.py` | exists | stat |
| path | `tests/test_refmet_availability_plumbing.py` | exists | stat |
| path | `docs/brainstorms/2026-09-09-write-through-refmet-freeze-spec.md` | exists | stat |
| check | `_fetch_live_backup` is defined in the annotator on `origin/dev` | 1 | `git -C <repo> grep -c "def _fetch_live_backup" origin/dev -- src/biomapper2/core/annotators/metabolomics_workbench.py \| cut -d: -f3` |
| check | store table is still keyed on the exact query on `origin/dev` | 1 | `git -C <repo> grep -c "CREATE TABLE IF NOT EXISTS refmet_by_query" origin/dev -- src/biomapper2/core/annotators/refmet_store.py \| cut -d: -f3` |
| blocker | Issue Phenome-Health/biomapper2#159 still open | OPEN | `gh issue view 159 --repo Phenome-Health/biomapper2 --json state --jq .state` |

## Sources & References

- **Origin document:** `docs/brainstorms/2026-11-13-refmet-keep-vote-on-live-no-match-requirements.md`
- Earlier design narrowed by this plan: `docs/brainstorms/2026-09-09-write-through-refmet-freeze-spec.md` (D6)
- Issue: Phenome-Health/biomapper2#159
- Related fixes: v1.4.0 `/match` `refmet_id` backfill; #155 bare RefMet twin
