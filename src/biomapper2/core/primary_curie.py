"""Pick the one CURIE a response reports as the entity's identifier, consistent with ``chosen_kg_id``.

``curies`` is the sorted union of every CURIE any source produced for the entity (provided IDs plus every
annotator's hits). Its first element is alphabetical, not "the answer": when the resolver commits a node
that the alphabetically-first CURIE does not link to (a RefMet override such as ``divergent_refmet``, or
a plain vote the first CURIE lost), ``curies[0]`` names a different KG node than ``chosen_kg_id``. The
client's ``primary_curie`` read ``curies[0]``, so a user reading it got the losing hit (e.g.
``dihomo-linolenate (20:3n3 or n6)`` -> chosen ``RM:0153916``, ``curies[0]`` ``CHEBI:88670``).

Rule (:func:`select_primary_curie`):

1. No ``chosen_kg_id`` -> ``curies[0]`` (or None). Unresolved rows keep the legacy value.
2. Otherwise the FIRST CURIE in ``curies`` order that is an identifier of the chosen node, where the
   chosen node's identifier set is ``{chosen_kg_id}`` + the CURIEs the linker resolved to it
   (``kg_ids[chosen_kg_id]``) + its ``kg_equivalent_ids`` rejoined as ``PREFIX:local_id``.
3. If no CURIE in ``curies`` is in that set -> ``chosen_kg_id`` itself.

Because step 2 walks ``curies`` in its existing order, whenever ``curies[0]`` is already an identifier of
the chosen node the result is exactly ``curies[0]``: the value only changes on rows where it was wrong.
"""

from collections.abc import Iterable, Mapping

__all__ = ["chosen_node_identifiers", "select_primary_curie"]


def chosen_node_identifiers(
    chosen_kg_id: str,
    kg_ids: Mapping[str, Iterable[str]] | None,
    kg_equivalent_ids: Mapping[str, Iterable[str]] | None,
) -> set[str]:
    """Every CURIE known to identify ``chosen_kg_id``'s node.

    ``kg_equivalent_ids`` is grouped ``{prefix: [local_id]}``, split on the FIRST colon by the linker
    (``Linker.get_equivalent_ids_checked``), so ``f"{prefix}:{local_id}"`` reproduces the graph's CURIE.
    """
    identifiers = {chosen_kg_id}
    identifiers.update((kg_ids or {}).get(chosen_kg_id) or [])
    for prefix, local_ids in (kg_equivalent_ids or {}).items():
        identifiers.update(f"{prefix}:{local_id}" for local_id in local_ids or [])
    return identifiers


def select_primary_curie(
    curies: list[str] | None,
    chosen_kg_id: str | None,
    kg_ids: Mapping[str, Iterable[str]] | None = None,
    kg_equivalent_ids: Mapping[str, Iterable[str]] | None = None,
) -> str | None:
    """Return the response's ``primary_curie`` (see module docstring for the rule)."""
    curies = curies or []
    if not chosen_kg_id:
        return curies[0] if curies else None
    node_ids = chosen_node_identifiers(chosen_kg_id, kg_ids, kg_equivalent_ids)
    for curie in curies:
        if curie in node_ids:
            return curie
    return chosen_kg_id
