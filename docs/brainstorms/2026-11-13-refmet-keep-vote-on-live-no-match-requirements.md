---
date: 2026-11-13
topic: refmet-keep-vote-on-live-no-match
issue: Phenome-Health/biomapper2#159
---

# Keep the last-known-good RefMet vote when live `/match` turns into a no-match

## Problem Frame

In `live_backup` mode (prod), the RefMet annotator treats every live `/match` answer as authoritative. The write-through freeze spec (`docs/brainstorms/2026-09-09-write-through-refmet-freeze-spec.md`, D6) made a live no-match a real negative that is written back to the store. That holds when RefMet has never known a query. It fails when RefMet used to match a query and stops, which Metabolomics Workbench did without notice around 2026-11-12 for Metabolon fatty-acid names with a shorthand suffix (for example `docosahexaenoate (dha; 22:6n3)`).

When that happens today:

- the RefMet vote disappears and the resolver falls back to whatever Kestrel returned. On NECS, n3 DPA then commits the n6 DPA node (wrong) and DHA commits a node with no structure;
- the store row holding the last-known-good id is overwritten with the no-match (`refmet_by_query` is one row per query, last-writer-wins);
- nothing records that the answer changed. The drift was found only by diffing two benchmark runs.

Affected: anyone mapping metabolite names on prod, and every benchmark or harmonization number that depends on a RefMet vote, including the preprint's.

Evidence checked on 2026-11-13:

- The raw live response for `docosahexaenoate (dha; 22:6n3)` is HTTP 200 with `refmet_name: "-"`: a real no-match answer, not a rejected request.
- The ids the drifted names used to resolve to still exist in RefMet with structures (`RM0153631` "DHA", `RM0153164` "DPA"). This is name-matching drift, not retirement of the entries.
- Prod's store has exactly 5 rows where the freeze says voted and the store says no-match, all written 2026-11-12 between 04:44 and 04:58 UTC: `linoleamide (18:2n6)`, `eicosapentaenoate (epa; 20:5n3)`, `docosahexaenoate (dha; 22:6n3)`, `docosapentaenoate (n3 dpa; 22:5n3)`, `docosapentaenoate (n6 dpa; 22:5n6)`. No older contradictions exist.

## Definitions

- **Query**: the exact string sent to `/match`, which is the store's key. One input name can produce two queries (canonical spelling first, raw spelling on a no-match); that order is unchanged. Every requirement below operates per query.
- **Live no-match**: an answer the annotator already classifies as NO_MATCH (a 200 with `refmet_name` or `refmet_id` of `-`, or a per-name 404). Timeouts, transport errors, 5xx and an open breaker are "live unreachable", not a no-match.
- **Last-known-good vote**: a store row for the query with status `voted` and a RefMet id.

## Requirements

**Behavior when live disagrees with a known-good answer**

- R1. In `live_backup` mode, when live returns a no-match for a query that has a last-known-good vote, the annotator returns that vote instead of the no-match.
- R2. R1 applies only to a live no-match. When live returns a match with a different RefMet id, live wins, as today.
- R3. When live is unreachable, behavior is unchanged (serve the store as backup). When live returns a no-match and the query has no last-known-good vote, the result is a no-match and is written to the store, as today.
- R4. A live no-match never overwrites a stored vote. The store is the only runtime source of last-known-good; the freeze file is not consulted at runtime in `live_backup` mode.

**Visibility**

- R5. A result served under R1 carries a distinct RefMet source label, different from `live_api`, `freeze_backup`, `local_snapshot` and `not_in_snapshot`. Its availability is `voted`.
- R6. A kept vote does not raise a review flag and does not change the certificate state on its own.
- R7. Kept votes are counted in the existing per-batch RefMet source counts, so a run summary shows how many votes were kept without anyone querying the history.

**History**

- R8. The store keeps, per query, the last observed live answer separately from the last-known-good vote, so a change in what live says can be detected even while the vote is kept.
- R9. Each change in the live answer for a query is recorded durably and once: the query, the old and new live status and id, and a UTC timestamp. That covers a match turning into a no-match (the first time R1 keeps a vote), a kept vote's query matching again, a no-match turning into a match, and a match changing id. Repeated requests with the same live answer add no record, including across the two API workers.
- R10. The history can be listed with one documented command, without reading application logs.
- R11. A failure to read or write the history or the last observed live answer never fails a mapping, in line with the store's existing best-effort rule.

**One-time repair of rows already overwritten**

- R12. A repair command compares the store with a freeze file by exact query (no case or whitespace folding) and lists every row where the freeze has a vote with an id and the store has a no-match, with each row's last update time.
- R13. The command changes nothing unless told to apply. Applying restores the listed rows to the freeze's vote and writes a history record for each.
- R14. The repair is run by an operator on prod after deploy, with the dry-run list reviewed first. Expected list on prod today: the 5 queries named above.

**Clearing a kept vote**

- R15. Deleting or editing a query's store row is the supported way to clear a kept vote, and it works for every query, because the store is the only runtime source (R4).

**Unchanged modes**

- R16. `frozen` and `off` modes behave exactly as before, including `REFMET_LIVE_API_FALLBACK`.

## Success Criteria

- Offline tests cover:
  - live no-match with a stored vote: vote kept, label set, store vote intact, one history record;
  - the same again: no second history record;
  - live matches again after a kept vote: normal `live_api` result, one history record;
  - live no-match with no stored vote: no-match, stored;
  - live match with a different id: live wins, store updated, history record;
  - live unreachable with a stored vote: served as `freeze_backup`, unchanged;
  - history or state write failure: mapping still succeeds;
  - repair dry run lists contradicting rows and writes nothing; repair apply restores them; a freeze row that differs only by case is not used.
- No existing RefMet annotator test changes its expected outcome, other than tests that assert the old overwrite.
- On prod after deploy and repair, `docosahexaenoate (dha; 22:6n3)` and `docosapentaenoate (n3 dpa; 22:5n3)` vote again with the new source label and commit RM:0153631 and RM:0153164, the nodes they committed on 2026-11-06.
- A batch that includes a kept vote shows it in the source counts.

## Scope Boundaries

- No review flag and no new certificate state for kept votes (decided: label only).
- No runtime fallback to the freeze file in `live_backup` mode (decided: one-time repair instead).
- No scheduled canary over the frozen corpus, and no check that a kept id still exists in RefMet. Both are reasonable follow-ups.
- No automatic expiry of kept votes.
- No change to the annotator's 7-day HTTP cache. A cached live answer can delay both the first kept vote and the history timestamps by up to 7 days.
- No change to how a live match with a different id is handled.
- Not the fix for prod going unresponsive when Kestrel is slow (blocking handlers); that is separate.
- No client (`biomapper`) change. The client passes `refmet_source` through as a string.
- Extending the freeze file to cover every benchmark input is a separate reproducibility task.

## Key Decisions

- Keep the stored vote only against a live no-match (Trent, 2026-11-13): a new no-match is far more likely upstream drift than a real retirement, and losing the vote produced a wrong node. A live match with a different id is positive evidence and is trusted. The 2026-11-13 check supports this for the known cases: the old ids still exist in RefMet.
- Label only, no review flag (Trent, 2026-11-13): the kept answer was good before, and extra flagged rows would inflate the review counts the preprint reports.
- One-time reviewed repair, not a standing freeze fallback (Trent, 2026-11-13): the store stays the single source at runtime, so a kept vote can always be cleared, outage and non-outage paths agree, and no older legitimate no-match is resurrected. Prod has only 5 rows to repair.
- This narrows D6 of the 2026-09-09 spec: a live no-match stays authoritative only for queries with no stored vote.

## Dependencies / Assumptions

- Prod runs `REFMET_FREEZE_MODE=live_backup` with `REFMET_STORE_PATH` and `REFMET_SNAPSHOT_PATH` set in the service's `.env` (verified on the host 2026-11-12; not visible in the repo). The freeze covers the NECS, Arivale and Xu panels (1,686 queries).
- 20 store rows outside the freeze were written as live no-matches since 2026-11-10. Whether any of them previously voted is unknown, and the store cannot say. They are not repaired by this change.
- A kept vote can be stale if RefMet deliberately retires or merges an entry. The label (R5), counts (R7), history (R9, R10) and R15 are the mitigation.
- Until this ships, any number from a `live_backup` run depends on which cached answers have expired. After it ships, a `live_backup` run should report its kept-vote count alongside its numbers. The preprint reruns still need the prod RefMet configuration decided separately.
- With `REFMET_STORE_PATH` unset in `live_backup` mode the store is a no-op, so R1 and the history do nothing. That is existing behavior and already warned about.

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R5][Technical] The exact source label string. No consumer enumerates `refmet_source` (it is a plain string in the API models and certificate); the constants and their comment block live in `src/biomapper2/core/annotators/base.py`. Check whether the snapshot-version attribution set in `src/biomapper2/mapper.py` should include the new label, and whether any test asserts the exact set of source keys.
- [Affects R8, R9][Technical] How the last observed live answer and the history are stored in the existing SQLite store, including migrating a store created before this change, and how "once" is enforced across the two workers.
- [Affects R10, R12][Technical] The form of the listing and repair commands (module entry point or script) and how the repair identifies the freeze file.
- [Affects R3, R8][Technical] Whether a store row seeded from the freeze counts as a last-known-good vote exactly like a live-written one (the code makes no distinction today, so presumably yes).

## Next Steps

-> /ce:plan for structured implementation planning
