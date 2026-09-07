"""Lipid independent-structure resolver: Goslin shorthand -> LIPID MAPS REST -> FULL InChIKey.

Why this exists
---------------
The dominant KG-conflation class is Metabolon lipid shorthand that MW and PubChem (the existing
independent-structure hops) cannot resolve by exact name -- the REFUSED-link population the
cross-cohort adjudication is dominated by. Goslin parses and
NORMALIZES the shorthand but emits no structure; LIPID MAPS binds a Goslin-canonical name to an
InChIKey. This resolver chains the two into the one "independent structure for a query name" service
the ``independent-structure-iface`` contract promises, so both the QUERY side (Tier B,
``IndependentStructureLookup``) and the CANDIDATE side (``StructureResolver``) can resolve a lipid.

The circularity firewall (KTD1 / ledger L3)
-------------------------------------------
LIPID MAPS is KG-independent -- valid for the cross-cohort resolution certificate -- but it IS the
gold for the LMSD benchmark arm, so a lipidmaps-sourced structure is circular against that arm. Every
RESOLVED result carries ``source="lipidmaps"`` so the benchmark axis can exclude those rows from its
LMSD lipid oracle. This mirrors the existing ``lipidmaps_rest_enrichment_fired`` coverage-not-accuracy
flag. (An offline pygoslin-SMILES -> RDKit InChIKey path would be a cleaner independence story and
would tag ``source="goslin"``; that is a noted follow-up, not this cycle.)

Outcome discipline
------------------
Same ``lookup_failed`` vs ``unresolvable`` discipline as the other hops: a Goslin parse-miss is a
clean ``unresolvable`` (the name is not a lipid), a parsed-but-unbindable lipid is ``unresolvable``
(LIPID MAPS answered "unknown"), and a 5xx/transport failure is ``lookup_failed`` and is NOT memoized
-- caching a transient outage would pin it onto every later occurrence of the name.
"""

from __future__ import annotations

import logging
from typing import Any

from .certificate import TIER_B_SOURCE_LIPIDMAPS, TierBOutcome, TierBResult

log = logging.getLogger(__name__)

_UNRESOLVED = TierBResult(source=None, inchikey_block=None, outcome=TierBOutcome.UNRESOLVABLE)
_LOOKUP_FAILED = TierBResult(source=None, inchikey_block=None, outcome=TierBOutcome.LOOKUP_FAILED)


class LipidStructureResolver:
    """Resolve a (possibly lipid) query name to an independent FULL InChIKey via Goslin + LIPID MAPS.

    Both collaborators are injectable so the class is fully unit-testable on literals; no test may
    exercise it against pygoslin construction or a live LIPID MAPS endpoint.
    """

    def __init__(self, grammar: Any | None = None, enricher: Any | None = None) -> None:
        # Construct the real collaborators lazily so importing this module never pulls in pygoslin or
        # opens a session; a caller that only ever passes fakes pays nothing.
        if grammar is None:
            from .annotators.goslin_grammar import LipidGrammar

            grammar = LipidGrammar()
        if enricher is None:
            from .annotators.lipidmaps_rest import LipidMapsRestEnricher

            enricher = LipidMapsRestEnricher()
        self._grammar = grammar
        self._enricher = enricher
        self._memo: dict[str, TierBResult] = {}

    def resolve(self, name: str | None) -> TierBResult:
        """Resolve one query name. Never raises; a service failure comes back as ``lookup_failed``."""
        key = (name or "").strip()
        if not key:
            return _UNRESOLVED
        if key in self._memo:
            memo = self._memo[key]
            return TierBResult(
                source=memo.source,
                inchikey_block=memo.inchikey_block,
                outcome=memo.outcome,
                cache_state="process_memo",
            )

        result = self._resolve(key)
        if result.outcome is TierBOutcome.LOOKUP_FAILED:
            # Deliberately NOT memoized -- a transient LIPID MAPS outage must not become a durable
            # property of the name (mirrors IndependentStructureLookup.lookup).
            return result
        self._memo[key] = result
        return result

    def _resolve(self, name: str) -> TierBResult:
        parsed = self._grammar.parse(name)
        if parsed is None:
            return _UNRESOLVED  # not a lipid; a clean "unknown", never a failure
        mapping, ok = self._enricher.enrich_checked(parsed.canonical_name)
        if not ok:
            log.warning("LIPID MAPS lookup failed for '%s'; recording lookup_failed", parsed.canonical_name)
            return _LOOKUP_FAILED
        inchikey = mapping.get("INCHIKEY")
        if not inchikey:
            return _UNRESOLVED  # parsed as a lipid but LIPID MAPS has no structure for it
        return TierBResult(
            source=TIER_B_SOURCE_LIPIDMAPS,
            # FULL key on purpose (block2 present): the lipid source is the one hop that can supply
            # stereo, which the certificate's structural-key comparison uses when both sides carry it.
            inchikey_block=str(inchikey).upper(),
            outcome=TierBOutcome.RESOLVED,
        )
