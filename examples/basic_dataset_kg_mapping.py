from pathlib import Path

from biomapper2.mapper import Mapper

PROJECT_ROOT_PATH = Path(__file__).parents[1]


mapper = Mapper()

# Example with a protein dataset
mapper.map_dataset_to_kg(
    dataset=str(PROJECT_ROOT_PATH / "data" / "examples" / "olink_protein_metadata.tsv"),
    entity_type="protein",
    name_column="Assay",
    provided_id_columns=["UniProt"],
    array_delimiters=["_"],
)


# Example with a groundtruth disease dataset
mapper.map_dataset_to_kg(
    dataset=str(PROJECT_ROOT_PATH / "data" / "groundtruth" / "diseases_handcrafted.tsv"),
    entity_type="disease",
    name_column="name",
    provided_id_columns=[],
    array_delimiters=[],
)


# Example with a metabolites dataset
mapper.map_dataset_to_kg(
    dataset=str(PROJECT_ROOT_PATH / "data" / "examples" / "metabolites_synthetic.tsv"),
    entity_type="metabolite",
    name_column="name",
    provided_id_columns=["INCHIKEY", "HMDB", "KEGG", "PUBCHEM", "CHEBI"],
    array_delimiters=[",", ";"],
)
