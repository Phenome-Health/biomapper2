"""Tests for mapping datasets to the KG."""
from pathlib import Path

from biomapper2.mapper import Mapper


PROJECT_ROOT_PATH = Path(__file__).parents[1]


def test_map_dataset_metabolon_groundtruth(shared_mapper: Mapper):

    # Map the dataset
    results_tsv_path, stats = shared_mapper.map_dataset_to_kg(
        dataset_tsv_path=str(PROJECT_ROOT_PATH / 'data' / 'groundtruth' / 'metabolon_metadata_2025.tsv'),
        entity_type='metabolite',
        name_column='CHEMICAL_NAME',
        provided_id_columns=['INCHIKEY', 'SMILES', 'CAS', 'HMDB', 'KEGG', 'PUBCHEM'],
        array_delimiters=[',', ';'])

    # Since this is a groundtruth set where all items are known to map to Kraken, coverage should be 100%
    assert stats['performance']['overall']['coverage'] == 1.0

    # And precision/recall/f1-score should also be 100% (when calculated before resolving one-to-manys)
    assert stats['performance']['overall']['per_groundtruth']['precision'] == 1.0
    assert stats['performance']['overall']['per_groundtruth']['recall'] == 1.0
    assert stats['performance']['overall']['per_groundtruth']['f1_score'] == 1.0

