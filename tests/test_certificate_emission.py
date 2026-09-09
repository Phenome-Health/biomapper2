"""The certificate on BOTH emission paths, and the shapes each surface actually needs.

Why both paths
--------------
``map_entity_to_kg`` and ``map_dataset_to_kg`` run *different* stage-5 blocks -- one enriches a
single node, the other batches. Every artifact the certificate work cites (the audit, the suite
arms, the published curve) comes from the dataset path, so wiring only the single-entity path would
have made the whole change invisible to the evidence. The state table below is therefore
parametrized over both, and asserts they agree row for row.

The certificate is also built OUTSIDE the ``chosen_kg_id is not None`` guard: inside it, the rows
the certificate most needs to describe -- the ones with no committed node -- would get no
certificate at all, and the two surfaces would disagree precisely there.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from biomapper2.core.certificate import SELECTION_CONFLICT_VALUES, CertificateState, StructureStatus
from biomapper2.mapper import Mapper

SMALL_MOLECULE = "biolink:SmallMolecule"
GENE = "biolink:Gene"
NODE = "CHEBI:15365"
WITH_KEY = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"], "HMDB": ["HMDB0001879"]}
WITHOUT_KEY = {"HMDB": ["HMDB0001879"]}


class _StubBiolink:
    def standardize_entity_type(self, entity_type: str) -> str:
        return entity_type

    def get_descendants(self, item: str) -> set[str]:
        return {SMALL_MOLECULE} if item == SMALL_MOLECULE else set()


class _StubLinker:
    """Stands in for Kestrel. Records calls so a test can prove the default path made none."""

    def __init__(self, equiv: dict[str, list[str]], lookup_ok: bool = True) -> None:
        self.equiv = equiv
        self.lookup_ok = lookup_ok
        self.calls = 0

    def link(self, item):
        fields = {"kg_ids": {}, "kg_ids_provided": {}, "kg_ids_assigned": {}}
        if isinstance(item, pd.DataFrame):
            return pd.DataFrame([fields] * len(item), index=item.index)
        return pd.Series(fields)

    def get_equivalent_ids_checked(self, kg_node_ids, prefixes=None):  # noqa: ARG002
        self.calls += 1
        return {kid: self.equiv for kid in kg_node_ids}, self.lookup_ok


def _stub_mapper(
    *,
    chosen_kg_id: str | None,
    review_flag: str | None = None,
    equiv: dict[str, list[str]] | None = None,
    lookup_ok: bool = True,
    kg_ids_assigned: dict[str, dict[str, list[str]]] | None = None,
    tier_b: Any = None,
) -> Mapper:
    """A Mapper with every network-touching collaborator replaced.

    Built with ``__new__`` rather than ``Mapper()`` because the real constructor downloads a Biolink
    model; this suite must stay offline.
    """
    mapper = Mapper.__new__(Mapper)
    mapper.biolink_client = _StubBiolink()  # pyright: ignore[reportAttributeAccessIssue]
    mapper.linker = _StubLinker(  # pyright: ignore[reportAttributeAccessIssue]
        equiv if equiv is not None else {}, lookup_ok
    )
    mapper.tier_b = tier_b

    resolved = {
        "chosen_kg_id": chosen_kg_id,
        "chosen_kg_id_provided": None,
        "chosen_kg_id_assigned": None,
        "chosen_kg_id_review": review_flag,
    }
    assigned = kg_ids_assigned or {}

    class _StubResolver:
        def resolve(self, item, category=None):  # noqa: ARG002
            if isinstance(item, pd.DataFrame):
                return pd.DataFrame([resolved] * len(item), index=item.index)
            return pd.Series(resolved)

        def is_small_molecule(self, category: str | None) -> bool:
            return category == SMALL_MOLECULE

    class _StubAnnotation:
        def annotate(self, item, **kwargs):  # noqa: ARG002
            fields = {"assigned_ids": {}}
            if isinstance(item, pd.DataFrame):
                return pd.DataFrame([fields] * len(item), index=item.index)
            return pd.Series(fields)

    class _StubNormalizer:
        def get_standard_prefix(self, vocab):  # noqa: ARG002
            return None

        def normalize(self, item, **kwargs):  # noqa: ARG002
            fields = {"curies": [], "curies_provided": [], "curies_assigned": {}}
            if isinstance(item, pd.DataFrame):
                return pd.DataFrame([fields] * len(item), index=item.index)
            return pd.Series(fields)

    mapper.resolver = _StubResolver()  # pyright: ignore[reportAttributeAccessIssue]
    mapper.annotation_engine = _StubAnnotation()  # pyright: ignore[reportAttributeAccessIssue]
    mapper.normalizer = _StubNormalizer()  # pyright: ignore[reportAttributeAccessIssue]
    # kg_ids_assigned is threaded through the linker stub's output for the dataset path.
    mapper.linker.link = _patched_link(assigned)  # type: ignore[method-assign]
    return mapper


def _patched_link(assigned: dict[str, dict[str, list[str]]]):
    def link(item):
        fields = {"kg_ids": {}, "kg_ids_provided": {}, "kg_ids_assigned": assigned}
        if isinstance(item, pd.DataFrame):
            return pd.DataFrame([fields] * len(item), index=item.index)
        return pd.Series(fields)

    return link


def _entity_path(mapper: Mapper, entity_type: str, name: str = "glucose") -> dict[str, Any]:
    result = mapper.map_entity_to_kg(
        item={"name": name},
        name_field="name",
        provided_id_fields=[],
        entity_type=entity_type,
    )
    assert isinstance(result, dict)
    return result


def _dataset_path(mapper: Mapper, entity_type: str, tmp_path, name: str = "glucose") -> pd.DataFrame:
    import biomapper2.mapper as mapper_module

    original = mapper_module.analyze_dataset_mapping
    mapper_module.analyze_dataset_mapping = lambda *a, **k: {}  # type: ignore[assignment]
    try:
        out_path, _ = mapper.map_dataset_to_kg(
            dataset=pd.DataFrame({"name": [name]}),
            entity_type=entity_type,
            name_column="name",
            provided_id_columns=[],
            output_dir=tmp_path,
        )
    finally:
        mapper_module.analyze_dataset_mapping = original  # type: ignore[assignment]
    return pd.read_csv(out_path, sep="\t")


# --------------------------------------------------------------------------------------------
# The state table, over both emission paths
# --------------------------------------------------------------------------------------------

_STATE_TABLE = [
    ("structure_present", SMALL_MOLECULE, NODE, WITH_KEY, True, CertificateState.UNCORROBORATED),
    ("structure_absent", SMALL_MOLECULE, NODE, WITHOUT_KEY, True, CertificateState.UNAVAILABLE),
    ("non_small_molecule", GENE, NODE, WITH_KEY, True, CertificateState.NOT_APPLICABLE),
    ("no_committed_node", SMALL_MOLECULE, None, {}, True, CertificateState.UNAVAILABLE),
    ("kestrel_outage", SMALL_MOLECULE, NODE, {}, False, CertificateState.UNAVAILABLE),
]


@pytest.mark.parametrize(
    ("label", "entity_type", "chosen", "equiv", "lookup_ok", "expected"),
    _STATE_TABLE,
    ids=[row[0] for row in _STATE_TABLE],
)
def test_both_emission_paths_agree_on_the_state(
    label: str, entity_type: str, chosen: str | None, equiv: dict, lookup_ok: bool, expected: CertificateState, tmp_path
) -> None:
    entity_result = _entity_path(_stub_mapper(chosen_kg_id=chosen, equiv=equiv, lookup_ok=lookup_ok), entity_type)
    dataset_result = _dataset_path(
        _stub_mapper(chosen_kg_id=chosen, equiv=equiv, lookup_ok=lookup_ok), entity_type, tmp_path
    )
    assert entity_result["resolution_certificate"]["state"] == expected.value, label
    assert dataset_result.loc[0, "certificate_state"] == expected.value, label


def test_a_row_with_no_committed_node_still_gets_a_certificate(tmp_path) -> None:
    """The regression this guards: building the certificate inside the null guard would leave
    exactly this population undescribed on one surface and described on the other."""
    entity_result = _entity_path(_stub_mapper(chosen_kg_id=None), SMALL_MOLECULE)
    dataset_result = _dataset_path(_stub_mapper(chosen_kg_id=None), SMALL_MOLECULE, tmp_path)
    assert entity_result["resolution_certificate"] is not None
    assert dataset_result.loc[0, "certificate_structure_status"] == StructureStatus.NOT_APPLICABLE.value


# --------------------------------------------------------------------------------------------
# G6 — the default path adds no I/O
# --------------------------------------------------------------------------------------------


def test_tier_a_makes_no_structure_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asserted on the StructureResolver session, not the linker: MW/PubChem calls never traverse
    ``Linker``, so a mocked-linker counter is blind to exactly the calls Tier A must not make."""
    from biomapper2.core import structure_resolver as structure_module

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Tier A made an external structure call")

    monkeypatch.setattr(structure_module.StructureResolver, "_fetch_mw_inchikey", _forbidden)
    monkeypatch.setattr(structure_module.StructureResolver, "_fetch_pubchem_inchikey", _forbidden)
    monkeypatch.setattr(structure_module.StructureResolver, "inchikey_blocks", _forbidden)
    monkeypatch.setattr(structure_module.StructureResolver, "inchikey_block", _forbidden)

    for equiv in (WITH_KEY, WITHOUT_KEY, {}):
        result = _entity_path(_stub_mapper(chosen_kg_id=NODE, equiv=equiv), SMALL_MOLECULE)
        assert result["resolution_certificate"]["provenance"]["tier_b_enabled"] is False


def test_tier_a_makes_exactly_one_kestrel_enrichment_call_and_no_more() -> None:
    """The certificate must not add a second /get-nodes round trip on top of stage 5's."""
    mapper = _stub_mapper(chosen_kg_id=NODE, equiv=WITH_KEY)
    _entity_path(mapper, SMALL_MOLECULE)
    assert mapper.linker.calls == 1  # type: ignore[attr-defined]


def test_tier_b_is_not_constructed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import biomapper2.mapper as mapper_module

    monkeypatch.setattr(mapper_module, "TIER_B_ENABLED", False)
    assert mapper_module.Mapper._build_tier_b() is None


def test_tier_b_is_constructed_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import biomapper2.mapper as mapper_module

    monkeypatch.setattr(mapper_module, "TIER_B_ENABLED", True)
    built = mapper_module.Mapper._build_tier_b()
    assert built is not None


# --------------------------------------------------------------------------------------------
# Flat TSV shape (item 20) and legacy-flag derivation (G2)
# --------------------------------------------------------------------------------------------


def test_mapped_tsv_carries_flat_scalar_certificate_columns(tmp_path) -> None:
    frame = _dataset_path(_stub_mapper(chosen_kg_id=NODE, equiv=WITH_KEY), SMALL_MOLECULE, tmp_path)
    assert frame.loc[0, "certificate_state"] == "uncorroborated"
    assert frame.loc[0, "certificate_node_inchikey_blocks"] == "BSYNRYMUTXBXSQ"
    assert str(frame.loc[0, "certificate_comparison_rule"]).startswith("inchikey_first_block_set_intersection")
    # No object column may survive to the writer: it would emit ResolutionCertificate(state=...) and
    # reintroduce the ast.literal_eval-only column this design exists to eliminate.
    assert not any("ResolutionCertificate" in str(v) for v in frame.iloc[0].tolist())
    assert "resolution_certificate" not in frame.columns


@pytest.mark.parametrize("flag", SELECTION_CONFLICT_VALUES)
def test_legacy_review_flag_survives_derivation_on_both_paths(flag: str | None, tmp_path) -> None:
    """G2. All three values the resolver actually emits, unchanged for one release."""
    entity_result = _entity_path(_stub_mapper(chosen_kg_id=NODE, review_flag=flag, equiv=WITH_KEY), SMALL_MOLECULE)
    dataset_result = _dataset_path(
        _stub_mapper(chosen_kg_id=NODE, review_flag=flag, equiv=WITH_KEY), SMALL_MOLECULE, tmp_path
    )
    assert entity_result["chosen_kg_id_review"] == flag
    assert entity_result["resolution_certificate"]["selection_conflict"] == flag
    emitted = dataset_result.loc[0, "chosen_kg_id_review"]
    assert (flag is None and pd.isna(emitted)) or emitted == flag
    assert dataset_result.loc[0, "certificate_state"] == "uncorroborated"


def test_the_certificate_changes_no_committed_id(tmp_path) -> None:
    """G1 / L19. The certificate is a LABEL: ``chosen_kg_id`` is emitted unchanged in every state,
    including the ones the certificate refuses to vouch for. Zero coverage delta on this axis."""
    for equiv in (WITH_KEY, WITHOUT_KEY, {}):
        entity_result = _entity_path(_stub_mapper(chosen_kg_id=NODE, equiv=equiv), SMALL_MOLECULE)
        dataset_result = _dataset_path(_stub_mapper(chosen_kg_id=NODE, equiv=equiv), SMALL_MOLECULE, tmp_path)
        assert entity_result["chosen_kg_id"] == NODE
        assert dataset_result.loc[0, "chosen_kg_id"] == NODE


def test_tier_b_changes_no_committed_id_either() -> None:
    """Even a ``contradicted`` verdict withholds nothing. Withholding stays a documented extension
    point, not built."""
    from biomapper2.core.certificate import TierBOutcome, TierBResult

    class _Contradicting:
        def lookup(self, name):  # noqa: ARG002
            return TierBResult(source="pubchem", inchikey_block="ZZZZZZZZZZZZZZ", outcome=TierBOutcome.RESOLVED)

    result = _entity_path(_stub_mapper(chosen_kg_id=NODE, equiv=WITH_KEY, tier_b=_Contradicting()), SMALL_MOLECULE)
    assert result["resolution_certificate"]["state"] == "contradicted"
    assert result["chosen_kg_id"] == NODE


def test_the_emitted_certificate_is_json_serializable_on_the_entity_path() -> None:
    """The NDJSON streaming endpoint json.dumps's this dict OUTSIDE its try/except, so a dataclass
    here raises mid-stream after a 200 has already gone out."""
    result = _entity_path(_stub_mapper(chosen_kg_id=NODE, equiv=WITH_KEY), SMALL_MOLECULE)
    json.dumps(result["resolution_certificate"])


def test_committed_node_sources_are_threaded_for_the_independence_check() -> None:
    """L26 needs to know which annotator supplied the committed node; that lives in
    ``kg_ids_assigned`` and is otherwise dropped before the certificate is built."""
    from biomapper2.core.certificate import TierBOutcome, TierBResult

    class _StubTierB:
        def lookup(self, name):  # noqa: ARG002
            return TierBResult(
                source="metabolomics-workbench",
                inchikey_block="BSYNRYMUTXBXSQ",
                outcome=TierBOutcome.RESOLVED,
            )

    mapper = _stub_mapper(
        chosen_kg_id=NODE,
        equiv=WITH_KEY,
        kg_ids_assigned={"metabolomics-workbench": {NODE: ["CHEBI:15365"]}},
        tier_b=_StubTierB(),
    )
    certificate = _entity_path(mapper, SMALL_MOLECULE)["resolution_certificate"]
    assert certificate["state"] == "corroborated"
    assert certificate["independent_of_selection"] is False


# --------------------------------------------------------------------------------------------
# Tier B's population: the lookup is scoped to the rows a verdict can be ABOUT
# --------------------------------------------------------------------------------------------


class _RecordingTierB:
    """Records every name it is asked about, so a test can assert on calls NOT made."""

    def __init__(self) -> None:
        self.names: list[str | None] = []

    def lookup(self, name):
        from biomapper2.core.certificate import TierBOutcome, TierBResult

        self.names.append(name)
        return TierBResult(source="pubchem", inchikey_block="BSYNRYMUTXBXSQ", outcome=TierBOutcome.RESOLVED)


_OUT_OF_POPULATION = [
    # (label, entity_type, chosen, equiv, lookup_ok)
    ("structure_absent", SMALL_MOLECULE, NODE, WITHOUT_KEY, True),
    ("non_small_molecule", GENE, NODE, WITH_KEY, True),
    ("no_committed_node", SMALL_MOLECULE, None, {}, True),
    ("kestrel_outage", SMALL_MOLECULE, NODE, {}, False),
]


@pytest.mark.parametrize(
    ("label", "entity_type", "chosen", "equiv", "lookup_ok"),
    _OUT_OF_POPULATION,
    ids=[row[0] for row in _OUT_OF_POPULATION],
)
def test_tier_b_is_not_looked_up_for_a_row_no_verdict_can_be_about(
    label: str, entity_type: str, chosen: str | None, equiv: dict, lookup_ok: bool, tmp_path
) -> None:
    """A row outside the population must cost ZERO rate-limited round trips, on both paths.

    ``structure_absent`` is the case that was missing and the one that matters most: it stays
    ``unavailable`` under every Tier B outcome (L21), so the request buys nothing -- and it is the
    LARGEST bucket exactly where the KG is thinnest, which is where a throttled MW/PubChem hop is
    most expensive. Spending it anyway would also put the name into the Tier B resolution-rate
    denominator that travels with every operating point, biasing a published rate with names the
    curve never plots.
    """
    entity_tier_b = _RecordingTierB()
    _entity_path(_stub_mapper(chosen_kg_id=chosen, equiv=equiv, lookup_ok=lookup_ok, tier_b=entity_tier_b), entity_type)
    dataset_tier_b = _RecordingTierB()
    _dataset_path(
        _stub_mapper(chosen_kg_id=chosen, equiv=equiv, lookup_ok=lookup_ok, tier_b=dataset_tier_b),
        entity_type,
        tmp_path,
    )
    assert entity_tier_b.names == [], label
    assert dataset_tier_b.names == [], label


def test_tier_b_is_still_looked_up_for_the_population_it_exists_for() -> None:
    """Guards the guard: a scoping predicate that excludes everything is a Tier B wired shut, and
    the operator running the supervised sweep would see an all-``off`` column with no way to tell it
    from a correct refusal."""
    tier_b = _RecordingTierB()
    certificate = _entity_path(
        _stub_mapper(chosen_kg_id=NODE, equiv=WITH_KEY, tier_b=tier_b), SMALL_MOLECULE, name="glucose"
    )["resolution_certificate"]
    assert tier_b.names == ["glucose"]
    assert certificate["state"] == "corroborated"
    assert certificate["independent_source"] == "pubchem"


def test_an_absent_structure_carries_no_independent_evidence_fields(tmp_path) -> None:
    """The emitted row must not show independent evidence beside a verdict that ignored it.

    Complements the call-count test above: this is the shape a reader (or Panel B's stratification)
    sees, so it is asserted on the emitted certificate rather than on the lookup.
    """
    entity_certificate = _entity_path(
        _stub_mapper(chosen_kg_id=NODE, equiv=WITHOUT_KEY, tier_b=_RecordingTierB()), SMALL_MOLECULE
    )["resolution_certificate"]
    assert entity_certificate["state"] == "unavailable"
    assert entity_certificate["structure_status"] == "structure_absent"
    assert entity_certificate["independent_source"] is None
    assert entity_certificate["independent_inchikey_block"] is None
    assert entity_certificate["independent_of_selection"] is None

    frame = _dataset_path(
        _stub_mapper(chosen_kg_id=NODE, equiv=WITHOUT_KEY, tier_b=_RecordingTierB()), SMALL_MOLECULE, tmp_path
    )
    assert frame.loc[0, "certificate_state"] == "unavailable"
    assert pd.isna(frame.loc[0, "certificate_independent_source"])
    assert pd.isna(frame.loc[0, "certificate_independent_inchikey_block"])
