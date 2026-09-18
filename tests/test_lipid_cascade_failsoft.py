"""Unit 3 robustness: the level cascade is fail-soft. A Kestrel failure must not discard the RefMet /
LIPID MAPS votes already gathered; an input asserted at sn-position OR FINER is capped to
molecular-species for querying with trust off (so a structurally-detailed slash name never escapes the
policy); and one bad row must not abort a bulk lipid panel. Real offline pygoslin, all sources faked.
"""

from __future__ import annotations

import pandas as pd

from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator

_SM = "biolink:SmallMolecule"


class _FakeSource:
    """A name-binding source (RefMet or Kestrel). Records queried names; votes only for known names."""

    def __init__(self, slug: str, known: dict | None = None) -> None:
        self.slug = slug
        self.seen_names: list[str] = []
        self._known = known or {}

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        name = entity.get(name_field)
        self.seen_names.append(name)
        vocab_map = self._known.get(name, {})
        return {self.slug: {vocab: {id_: {} for id_ in ids} for vocab, ids in vocab_map.items()}}


class _RaisingKestrel:
    slug = "kestrel-hybrid-search"

    def get_annotations(self, *args, **kwargs):
        raise RuntimeError("kestrel is down")


def test_kestrel_failure_is_swallowed_and_refmet_votes_survive():
    binder = _FakeSource("metabolomics-workbench", {"PC 34:1": {"refmet_id": ["RM0001"]}})
    ann = GoslinLipidAnnotator(binder=binder, kestrel=_RaisingKestrel())  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 16:0/18:1"}, name_field="name", category=_SM)
    # The RefMet species vote still comes back even though Kestrel raised.
    assert out["goslin-lipid"]["refmet_id"]["RM0001"]["matched_level"] == "species"


def test_structurally_detailed_slash_name_is_capped_at_molecular_species_with_trust_off():
    # FULL_STRUCTURE input still carries a "/" sn claim, so trust-off must cap it at molecular-species.
    binder = _FakeSource("metabolomics-workbench", {"PA 18:1_12:0": {"refmet_id": ["RM0002"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PA 18:1(5Z)/12:0"}, name_field="name", category=_SM)
    assert not any("/" in n for n in binder.seen_names)  # no slash-bearing name is ever queried
    meta = out["goslin-lipid"]["refmet_id"]["RM0002"]
    assert meta["query_lipid_level_asserted"] == "full_structure"
    assert meta["query_lipid_level_effective"] == "molecular_species"
    assert meta["matched_level"] == "molecular_species"


def test_bulk_one_bad_row_does_not_abort_the_batch():
    class _Binder:
        slug = "metabolomics-workbench"

        def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
            name = entity.get(name_field)
            if name == "PC 34:1":
                raise RuntimeError("boom on the bad row's species query")
            vocab_map = {"FA 16:0": {"refmet_id": ["RM9"]}}.get(name, {})
            return {self.slug: {v: {i: {} for i in ids} for v, ids in vocab_map.items()}}

    ann = GoslinLipidAnnotator(binder=_Binder())  # pyright: ignore[reportArgumentType]
    df = pd.DataFrame({"name": ["PC 16:0/18:1", "FA 16:0"]})
    col = ann.get_annotations_bulk(df, name_field="name", category=_SM)
    assert col.iloc[0] == {}  # the failing row degraded to an empty vote
    assert "goslin-lipid" in col.iloc[1]  # the other row still mapped
