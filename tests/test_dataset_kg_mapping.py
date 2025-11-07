"""Tests for mapping datasets to the KG."""
from pathlib import Path

from biomapper2.mapper import Mapper


PROJECT_ROOT_PATH = Path(__file__).parents[1]


def test_map_dataset_metabolon_groundtruth(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / 'data' / 'goldstandard' / 'metabolon_metadata_2025.tsv'),
        entity_type='metabolite',
        name_column='CHEMICAL_NAME',
        provided_id_columns=['INCHIKEY', 'SMILES', 'CAS', 'HMDB', 'KEGG', 'PUBCHEM'],
        array_delimiters=[',', ';'])

    #