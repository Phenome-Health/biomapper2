"""Tests for mapping datasets to the KG."""

from pathlib import Path

from biomapper2.mapper import Mapper


PROJECT_ROOT_PATH = Path(__file__).parents[1]


def test_map_dataset_metabolites_synthetic(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path, stats = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / "data" / "examples" / "metabolites_synthetic.tsv"),
        entity_type="metabolite",
        name_column="name",
        provided_id_columns=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
        array_delimiters=[",", ";"],
    )

    # Since all items in this set are known to map to Kraken, coverage should be 100%
    assert stats["performance"]["overall"]["coverage"] == 1.0


def test_map_dataset_olink_proteins(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path, stats = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / "data" / "examples" / "olink_protein_metadata.tsv"),
        entity_type="protein",
        name_column="Assay",
        provided_id_columns=["UniProt"],
        array_delimiters=["_"],
    )

    # Based on provided ids alone, we get 2922 / 2923 proteins in this dataset
    assert stats["performance"]["overall"]["coverage"] > 0.999


def test_map_dataset_diseases_groundtruth(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path, stats = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / "data" / "groundtruth" / "diseases_handcrafted.tsv"),
        entity_type="disease",
        name_column="name",
        provided_id_columns=[],
        array_delimiters=[],
    )

    assert stats["performance"]["overall"]["coverage"] > 0.8

    # TODO: up these after have better techniques for mapping disease names
    assert stats["performance"]["overall"]["per_groundtruth"]["precision"] >= 0.6
    assert stats["performance"]["overall"]["per_groundtruth"]["recall"] >= 0.6
    assert stats["performance"]["overall"]["per_groundtruth"]["f1_score"] >= 0.6


def test_map_dataset_metabolites_synthetic_partial_provided(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path, stats = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / "data" / "examples" / "metabolites_synthetic_partial_provided.tsv"),
        entity_type="metabolite",
        name_column="name",
        provided_id_columns=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
        array_delimiters=[",", ";"],
    )

    # Even though a few items don't have provided ids in this dataset, all should still map to Kraken (via assigned IDs)
    assert stats["performance"]["overall"]["coverage"] == 1.0
    assert stats["mapped_to_kg_provided"] == 27
    assert stats["mapped_to_kg_assigned"] == 3
    assert stats["mapped_to_kg_provided_and_assigned"] == 0
