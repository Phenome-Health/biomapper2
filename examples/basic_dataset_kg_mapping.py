from pathlib import Path

from biomapper2.mapper import Mapper


PROJECT_ROOT_PATH = Path(__file__).parents[1]


mapper = Mapper()

# Example with a protein dataset
mapper.map_dataset_to_kg(dataset_tsv_path=str(PROJECT_ROOT_PATH / 'data' / 'examples' / 'olink_protein_metadata.tsv'),
                         entity_type='protein',
                         name_column='Assay',
                         provided_id_columns=['UniProt'])


# Example with a groundtruth disease dataset
mapper.map_dataset_to_kg(
    dataset_tsv_path=str(PROJECT_ROOT_PATH / 'data' / 'groundtruth' / 'diseases_handcrafted.tsv'),
    entity_type='disease',
    name_column='name',
    provided_id_columns=[],
    array_delimiters=[])


# Example with a groundtruth metabolites dataset
mapper.map_dataset_to_kg(dataset_tsv_path=str(PROJECT_ROOT_PATH / 'data' / 'groundtruth' / 'metabolon_metadata_2025.tsv'),
                         entity_type='metabolite',
                         name_column='CHEMICAL_NAME',
                         provided_id_columns=['INCHIKEY', 'SMILES', 'CAS', 'HMDB', 'KEGG', 'PUBCHEM'],
                         array_delimiters=[',', ';'])

