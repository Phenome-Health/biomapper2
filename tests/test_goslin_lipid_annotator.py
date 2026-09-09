import pandas as pd

from biomapper2.core.annotators.goslin_lipid import GoslinLipidAnnotator


class _FakeBinder:
    """Stands in for the RefMet/MW name binder. Records the name it was queried with."""

    slug = "metabolomics-workbench"

    def __init__(self, result_ids=None):
        self.seen_names = []
        self._result_ids = result_ids or {}

    def get_annotations(self, entity, name_field, category, prefixes=None, **kwargs):
        name = entity.get(name_field)
        self.seen_names.append(name)
        vocab_map = self._result_ids.get(name, {})
        # Shape: {slug: {vocab: {id: {}}}}
        return {self.slug: {vocab: {id_: {} for id_ in ids} for vocab, ids in vocab_map.items()}}


def test_lipid_shorthand_binds_via_canonicalized_query():
    # Binder only "knows" the CANONICAL name; raw "PC 34:1" would have missed.
    binder = _FakeBinder(result_ids={"PC 34:1": {"refmet_id": ["REFMET:RM0001"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    entity = {"name": "PC 34:1"}
    out = ann.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")
    # bound under the goslin-lipid slug
    assert "goslin-lipid" in out
    assert out["goslin-lipid"]["refmet_id"]["REFMET:RM0001"]["goslin_canonical"].startswith("PC ")
    assert out["goslin-lipid"]["refmet_id"]["REFMET:RM0001"]["goslin_dialect"] == "Goslin"
    # the binder was queried with the canonical name, not the raw shorthand
    assert binder.seen_names[0].startswith("PC ")


def test_non_lipid_returns_empty_and_never_queries_binder():
    binder = _FakeBinder()
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "caffeine"}, name_field="name", category="biolink:SmallMolecule")
    assert out == {}
    assert binder.seen_names == []  # fail-soft: non-lipid short-circuits before binding


def test_blank_name_returns_empty():
    ann = GoslinLipidAnnotator(binder=_FakeBinder())  # pyright: ignore[reportArgumentType]
    assert ann.get_annotations({"name": None}, name_field="name", category="biolink:SmallMolecule") == {}


def test_enrichment_off_by_default_is_never_called():
    class _ExplodingEnricher:
        def enrich(self, canonical_name):
            raise AssertionError("enrichment must be OFF by default")

    fake = _FakeBinder(result_ids={"FA 16:0": {"refmet_id": ["REFMET:RM9"]}})
    ann = GoslinLipidAnnotator(binder=fake)  # pyright: ignore[reportArgumentType]
    assert ann._enrichment is None  # default off
    # a run with default construction never touches enrichment
    out = ann.get_annotations({"name": "FA 16:0"}, name_field="name", category="biolink:SmallMolecule")
    assert "goslin-lipid" in out


def test_enrichment_when_injected_adds_lipidmaps_ids_and_flags_fired():
    class _FakeEnricher:
        def enrich(self, canonical_name):
            return {"LIPIDMAPS": "LMGP01010001", "INCHIKEY": "KILNVBDSWZSGLL-KXQOOQHDSA-N"}

    binder = _FakeBinder(result_ids={"PC 34:1": {"refmet_id": ["REFMET:RM0001"]}})
    ann = GoslinLipidAnnotator(binder=binder, enrichment=_FakeEnricher())  # pyright: ignore[reportArgumentType]
    out = ann.get_annotations({"name": "PC 34:1"}, name_field="name", category="biolink:SmallMolecule")
    assert "LMGP01010001" in out["goslin-lipid"]["LIPIDMAPS"]


def test_bulk_matches_rowwise_single_calls():
    binder = _FakeBinder(result_ids={"FA 16:0": {"refmet_id": ["REFMET:RM9"]}})
    ann = GoslinLipidAnnotator(binder=binder)  # pyright: ignore[reportArgumentType]
    df = pd.DataFrame({"name": ["FA 16:0", "caffeine"]})
    col = ann.get_annotations_bulk(df, name_field="name", category="biolink:SmallMolecule")
    assert "goslin-lipid" in col.iloc[0]
    assert col.iloc[1] == {}  # non-lipid row fell through
