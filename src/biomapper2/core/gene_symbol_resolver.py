"""Curated, non-search fallback for human gene nodes that Kestrel search cannot surface.

A small set of human genes whose protein product is a marketed therapeutic are conflated by the
upstream Translator/Babel layer into a single node named for the drug (e.g. GH1 -> the SOMATROPIN
node). The node still carries the gene symbol as a synonym and an HGNC equivalent, but it never ranks
for the gene symbol in any Kestrel search modality, so `prefer_human` re-ranking can never reach it.

This resolver is a deterministic, non-search bridge: for the curated symbols only, it returns the
canonical human NCBIGene CURIE, accepting it only after `/get-nodes` confirms the node carries the
queried symbol as a synonym AND an HGNC equivalent (so it cannot inject a wrong-but-human gene). It is
a no-op for every other symbol. This is a temporary measure; the durable fix is upstream (Kestrel-side
exact-symbol retrieval / de-conflation). See docs/plans/2026-06-17-001-feat-gene-symbol-fallback-resolver-plan.md.
"""

import contextvars
import logging

from ..config import HUMAN_MARKER_PREFIXES, KESTREL_BATCH_SIZE_CANONICALIZE
from ..utils import kestrel_request

_LAST_FAULT_SYMBOLS: contextvars.ContextVar[dict[int, str] | None] = contextvars.ContextVar(
    "_GENE_RESOLVER_LAST_FAULTS", default=None
)

# Case-folded human-marker prefixes, matched against the prefix of each equivalent id. Reuses the
# single source of truth in config (the same set _select_result uses) so the two human-gate paths
# cannot drift, and is case-insensitive to tolerate lower-cased CURIE prefixes from the KG.
_HUMAN_MARKER_PREFIXES_CF = {p.casefold() for p in HUMAN_MARKER_PREFIXES}

# Verified 2026-06-16: each of these human gene nodes carries its gene symbol as a synonym and an
# HGNC equivalent, yet is unreachable by symbol across hybrid/text/vector search (drug-conflation).
CURATED_GENE_SYMBOL_TO_CURIE = {
    "GH1": "NCBIGene:2688",
    "CALCA": "NCBIGene:796",
    "POMC": "NCBIGene:5443",
    "CRH": "NCBIGene:1392",
    "CTLA4": "NCBIGene:1493",
    "GBA1": "NCBIGene:2629",
}

# Case-folded lookup so queries match regardless of case.
_CURATED = {symbol.casefold(): curie for symbol, curie in CURATED_GENE_SYMBOL_TO_CURIE.items()}


class GeneSymbolResolver:
    """Resolve a curated gene symbol to its HGNC-verified human NCBIGene CURIE (or None)."""

    @property
    def _last_fault_symbol(self) -> str | None:
        mapping = _LAST_FAULT_SYMBOLS.get()
        return mapping.get(id(self)) if mapping is not None else None

    @_last_fault_symbol.setter
    def _last_fault_symbol(self, value: str | None) -> None:
        current = _LAST_FAULT_SYMBOLS.get() or {}
        updated = dict(current)
        if value is None:
            updated.pop(id(self), None)
        else:
            updated[id(self)] = value
        _LAST_FAULT_SYMBOLS.set(updated or None)

    def last_call_faulted(self, symbol: str | None = None) -> bool:
        """True if the last resolve call in this context encountered a Kestrel transport fault."""
        last_fault = self._last_fault_symbol
        if symbol is None:
            return last_fault is not None
        return last_fault == str(symbol).strip().casefold()

    def resolve(self, symbol: str | None) -> str | None:
        """Return the canonical human NCBIGene CURIE for a curated symbol, else None.

        Returns None (a no-op) for any non-curated symbol, and rejects a curated symbol whose node
        does not carry the queried symbol (as its name or a synonym) or lacks an HGNC equivalent —
        never fabricates and never returns a different human gene.
        """
        curie, _ = self.resolve_checked(symbol)
        return curie

    def resolve_checked(self, symbol: str | None) -> tuple[str | None, bool]:
        """Return ``(curie, faulted)`` atomically for the current lookup."""
        self._last_fault_symbol = None
        if not symbol:
            return None, False
        key = str(symbol).strip().casefold()
        curie = _CURATED.get(key)
        if curie is None:
            return None, False  # no-op for everything outside the curated set

        try:
            nodes = kestrel_request(
                method="POST",
                endpoint="get-nodes",
                batch_field="curies",
                batch_items=[curie],
                batch_size=KESTREL_BATCH_SIZE_CANONICALIZE,
                json={"slim": False, "truncate_long_fields": False},
            )
        except Exception as exc:
            self._last_fault_symbol = key
            from .fault_tracking import record_kestrel_fault

            record_kestrel_fault(None, f"gene-symbol fallback /get-nodes failed for {curie} ({symbol!r}): {exc}")
            logging.warning(f"gene-symbol fallback: get-nodes failed for {curie} ({symbol!r}): {exc}")
            return None, True

        node = nodes.get(curie) if isinstance(nodes, dict) else None
        if not isinstance(node, dict):
            return None, False

        # Verify identity (queried symbol present on the node), not just humanity (HGNC present). Match
        # on the node `name` OR a synonym, mirroring _select_result._symbol_matches so the two identity
        # checks cannot diverge: today the curated nodes are drug-named with the symbol in `synonyms`,
        # but if upstream de-conflation moves the symbol back into `name`, accepting either keeps the
        # guard correct instead of silently rejecting a now-corrected node.
        identity = {str(s).strip().casefold() for s in (node.get("synonyms") or [])}
        identity.add(str(node.get("name", "")).strip().casefold())
        if key not in identity:
            return None, False
        # Match on the CURIE prefix, case-insensitively, against the shared human-marker set — a
        # case-sensitive `startswith("HGNC:")` would silently reject a node whose equivalent ids
        # arrive lower-cased (e.g. "hgnc:4261"), dropping a genuinely human curated gene.
        if not any(
            str(e).split(":", 1)[0].strip().casefold() in _HUMAN_MARKER_PREFIXES_CF
            for e in (node.get("equivalent_ids") or [])
        ):
            return None, False

        return curie, False
