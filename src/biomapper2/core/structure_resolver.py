"""Layered InChIKey-connectivity resolution for the resolver's source-weighting rule.

Resolves each node's InChIKey first block (the 2-D structural skeleton) in order: KG
``get-nodes`` -> Metabolomics Workbench by name -> PubChem by name. All external calls are
guarded by a timeout and return ``None`` on any error rather than raising (mirroring
:meth:`Linker.get_equivalent_ids`), so a lookup failure degrades to "unresolvable" and the
caller flags the feature for review instead of committing a wrong choice.
"""

import logging
from typing import Any
from urllib.parse import quote

import requests_cache

from ..config import (
    CACHE_DIR,
    CACHE_IGNORED_PARAMETERS,
    MW_INCHIKEY_URL,
    PUBCHEM_INCHIKEY_URL,
    STRUCTURE_LOOKUP_TIMEOUT_S,
)
from .linker import Linker


class StructureResolver:
    """Adjudicates whether two KG nodes share InChIKey connectivity (2-D structure)."""

    def __init__(self, linker: Linker, lipid_resolver: Any | None = None) -> None:
        self.linker = linker
        self._session = requests_cache.CachedSession(
            str(CACHE_DIR / "structure_http"),
            ignored_parameters=CACHE_IGNORED_PARAMETERS,
        )
        self._name_cache: dict[str, str | None] = {}  # inchikey block by node name (per process)
        self._name_full_cache: dict[str, str | None] = {}  # FULL inchikey by node name (per process)
        # Candidate-side lipid hop (KTD6). Lipid shorthand nodes never resolve via KG/MW/PubChem by
        # name, so a lipid candidate is un-matchable in re-resolution without this. Shared with the
        # query-side Tier B lookup so both sides read one lipid resolver + cache. None disables it.
        self._lipid_resolver = lipid_resolver

    def connectivity_match(self, node_a: str, node_b: str) -> bool | None:
        """True if the nodes share ANY InChIKey first block, False if both resolve and share none,
        None if either is unresolvable across all layers.

        Set intersection, not equality against ``keys[0]``: a KG node's INCHIKEY list is multi-valued
        (neutral parent, conjugate anion, salt, stereoisomers) and its ordering is arbitrary, so
        comparing one arbitrary representation to another reports a false structural conflict when the
        shared block simply sits at a different index. This is the same artifact PR #36 fixed in the
        Hajjar scorer (the gold key sat at position 2-4 in 12 of 19 apparent misses).

        The loosening is one-directional and its cost is a **false agreement**, not a false conflict:
        a False can only become True, never the reverse. Where the old form could report a spurious
        conflict, this form can report a spurious match — if a KG node's ``equivalent_ids.INCHIKEY``
        list is itself conflated (entries for two genuinely different molecules), any single shared
        entry now reads as "same molecule" and the divergent-RefMet flag is suppressed. That is the
        accepted trade: KG INCHIKEY lists are dominated by protonation/salt/stereo variants of one
        structure, and a suppressed flag is a missed review prompt whereas the old false conflict was
        an actively wrong adjudication. Nodes sharing no block at all still return False, and
        ``inchikey_blocks`` keeps the MW/PubChem name fallback, so ``conflict_no_structure`` cannot
        inflate either.
        """
        records = self.linker.get_node_records([node_a, node_b])
        blocks_a = self.inchikey_blocks(node_a, (records.get(node_a) or {}).get("name"), records)
        blocks_b = self.inchikey_blocks(node_b, (records.get(node_b) or {}).get("name"), records)
        if not blocks_a or not blocks_b:
            return None
        return bool(blocks_a & blocks_b)

    def inchikey_block(self, node_id: str, node_name: str | None, records: dict[str, Any] | None = None) -> str | None:
        """First InChIKey block for a node: KG record -> MW by name -> PubChem by name -> lipid."""
        records = records if records is not None else self.linker.get_node_records([node_id])
        keys = ((records.get(node_id) or {}).get("equivalent_ids") or {}).get("INCHIKEY") or []
        if keys:
            return self._first_block(keys[0])
        if not node_name:
            return None
        if node_name in self._name_cache:
            return self._name_cache[node_name]
        block = self._first_block(self._resolve_name_key(node_name))
        self._name_cache[node_name] = block
        return block

    def structural_inchikey(
        self, node_id: str, node_name: str | None, records: dict[str, Any] | None = None
    ) -> str | None:
        """FULL InChIKey for a node: KG record -> MW by name -> PubChem by name -> lipid.

        Companion to :meth:`inchikey_block` that keeps the second (stereo) block so re-resolution can
        compare at the structural key (block1 + block2[:8]) instead of connectivity alone. Used only
        on the CANDIDATE nodes; the committed node's key is never read as the re-resolution anchor.
        """
        records = records if records is not None else self.linker.get_node_records([node_id])
        keys = ((records.get(node_id) or {}).get("equivalent_ids") or {}).get("INCHIKEY") or []
        if keys:
            first = next((k for k in keys if k), None)
            return str(first).upper() if first else None
        if not node_name:
            return None
        if node_name in self._name_full_cache:
            return self._name_full_cache[node_name]
        key = self._resolve_name_key(node_name)
        full = str(key).upper() if key else None
        self._name_full_cache[node_name] = full
        return full

    def structural_inchikeys(
        self, node_id: str, node_name: str | None, records: dict[str, Any] | None = None
    ) -> list[str]:
        """ALL FULL InChIKeys graph-asserted for a node, else the single name-resolved key.

        A candidate node may carry SEVERAL graph-asserted InChIKeys, and the query's independent
        structure can match any one of them. :meth:`structural_inchikey` returns only the first, so
        re-resolution matching on it would reject a valid candidate whose match is a non-first key.
        This companion returns the full set (order-preserving, de-duplicated, upper-cased) so the
        caller can accept a match against ANY asserted structure. Falls back to the name-resolved key.
        """
        records = records if records is not None else self.linker.get_node_records([node_id])
        keys = ((records.get(node_id) or {}).get("equivalent_ids") or {}).get("INCHIKEY") or []
        out: list[str] = []
        seen: set[str] = set()
        for k in keys:
            if k:
                ku = str(k).upper()
                if ku not in seen:
                    seen.add(ku)
                    out.append(ku)
        if out:
            return out
        if not node_name:
            return []
        # Reuse the single-key name hop (and its cache); a name resolves to at most one structure.
        one = self.structural_inchikey(node_id, node_name, records)
        return [one] if one else []

    def _resolve_name_key(self, node_name: str) -> str | None:
        """Full InChIKey for a NAME via MW -> PubChem -> lipid hop. Fail-soft (``None`` on error)."""
        try:
            key = self._fetch_mw_inchikey(node_name) or self._fetch_pubchem_inchikey(node_name)
        except Exception:
            logging.warning("Structure lookup failed for '%s'; treating as unresolvable", node_name, exc_info=True)
            return None
        if key:
            return key
        if self._lipid_resolver is not None:
            from .certificate import TierBOutcome

            lipid = self._lipid_resolver.resolve(node_name)
            if lipid.outcome is TierBOutcome.RESOLVED and lipid.inchikey_block:
                return lipid.inchikey_block
        return None

    def inchikey_blocks(self, node_id: str, node_name: str | None, records: dict[str, Any] | None = None) -> set[str]:
        """ALL KG-asserted InChIKey first-blocks for a node (the full ``equivalent_ids`` list).

        ``inchikey_block`` returns only ``keys[0]``; a KG node's INCHIKEY list is multi-valued
        (neutral parent, conjugate anion, salt, stereoisomers) and ``keys[0]`` ordering is arbitrary,
        so a gold structure can match a *non-first* entry (the Hajjar keys[0] artifact). This returns
        the union of every entry's first-block, letting a scorer test set membership instead of
        equality against one arbitrary representation. When the KG lists no InChIKey, falls back to
        the singular name-resolution path (a singleton, or empty when unresolvable) — never a
        spurious block.
        """
        records = records if records is not None else self.linker.get_node_records([node_id])
        keys = ((records.get(node_id) or {}).get("equivalent_ids") or {}).get("INCHIKEY") or []
        blocks = {b for b in (self._first_block(k) for k in keys if k) if b}
        if blocks:
            return blocks
        single = self.inchikey_block(node_id, node_name, records)
        return {single} if single else set()

    @staticmethod
    def _first_block(inchikey: str | None) -> str | None:
        return inchikey.split("-")[0] if inchikey else None

    def _fetch_mw_inchikey(self, name: str) -> str | None:
        """Metabolomics Workbench: GET /rest/refmet/name/{name}/inchi_key.

        ``safe=""`` is load-bearing. ``quote`` defaults to ``safe="/"``, which leaves a slash in the
        name UNESCAPED -- so a lipid-shorthand name splits into extra URL path segments, the request
        404s, and the caller records the name as structurally unresolvable rather than as a failed
        lookup. Share of affected names, per arm: artifact field ``slash_bearing_name_rate``.
        """
        url = f"{MW_INCHIKEY_URL}/{quote(name, safe='')}/inchi_key"
        resp = self._session.get(url, timeout=STRUCTURE_LOOKUP_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            key = data.get("inchi_key")
            return key if key and key != "-" else None
        return None

    def _fetch_pubchem_inchikey(self, name: str) -> str | None:
        """PubChem: GET /rest/pug/compound/name/{name}/property/InChIKey/JSON.

        ``safe=""`` for the same reason as :meth:`_fetch_mw_inchikey` -- an unescaped slash injects
        URL path segments and turns a resolvable name into a silent 404.
        """
        url = f"{PUBCHEM_INCHIKEY_URL}/{quote(name, safe='')}/property/InChIKey/JSON"
        resp = self._session.get(url, timeout=STRUCTURE_LOOKUP_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props and props[0].get("InChIKey"):
            return props[0]["InChIKey"]
        return None
