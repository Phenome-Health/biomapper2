"""End-to-end test of source-weighting through Mapper.map_entity_to_kg (no live APIs).

Mapper.__init__ builds a BiolinkClient (downloads the bmt model over the network), so the
Mapper is assembled via __new__ with mocked stages while the *real* map_entity_to_kg
orchestration and the *real* Resolver run — exercising the full resolution wiring offline.
"""

from unittest.mock import MagicMock

import pandas as pd

from biomapper2.core.resolver import Resolver
from biomapper2.mapper import Mapper


def _offline_mapper(connectivity):
    m = Mapper.__new__(Mapper)  # bypass the network-bound __init__

    m.biolink_client = MagicMock()
    m.biolink_client.standardize_entity_type = lambda t: "biolink:SmallMolecule"
    m.biolink_client.get_descendants = lambda c: {"biolink:SmallMolecule"}

    m.annotation_engine = MagicMock()
    m.annotation_engine.annotate = lambda **k: pd.Series(dtype=object)

    m.normalizer = MagicMock()
    m.normalizer.get_standard_prefix = lambda v: None
    m.normalizer.normalize = lambda **k: pd.Series(dtype=object)

    m.linker = MagicMock()
    m.linker.link = lambda s: pd.Series(
        {
            "kg_ids": {"CHEBI:bmp": ["a", "b"], "CHEBI:refmet": ["RM:1"]},
            "kg_ids_provided": {},
            "kg_ids_assigned": {"metabolomics-workbench": {"CHEBI:refmet": ["RM:1"]}},
        }
    )
    # Returns (mapping, lookup_ok): the certificate has to tell a Kestrel outage apart from a node
    # the graph genuinely lists nothing for.
    m.linker.get_equivalent_ids_checked = lambda ids, prefixes=None: ({}, True)
    m.tier_b = None  # Tier B is opt-in and must stay off on the default path

    m.resolver = Resolver(linker=m.linker, biolink_client=m.biolink_client)
    m.resolver.structure_resolver = MagicMock()
    m.resolver.structure_resolver.connectivity_match.return_value = connectivity
    return m


def test_e2e_map_entity_flags_divergent_refmet():
    out = _offline_mapper(connectivity=False).map_entity_to_kg(
        {"name": "some acid"}, name_field="name", provided_id_fields=[], entity_type="metabolite"
    )
    assert out["chosen_kg_id"] == "CHEBI:refmet"  # source-weighted to RefMet
    assert out["chosen_kg_id_review"] == "divergent_refmet"  # different connectivity -> flagged


def test_e2e_same_connectivity_silent_flip():
    out = _offline_mapper(connectivity=True).map_entity_to_kg(
        {"name": "some acid"}, name_field="name", provided_id_fields=[], entity_type="metabolite"
    )
    assert out["chosen_kg_id"] == "CHEBI:refmet"
    assert out["chosen_kg_id_review"] is None  # same molecule -> silent
