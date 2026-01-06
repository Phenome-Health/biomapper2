import logging

from biomapper2.config import PROJECT_ROOT
from biomapper2.mapper import Mapper
from biomapper2.utils import AnnotationMode
from biomapper2.visualizer import Visualizer

name = "name_col"
ids = "id_cols"
vocab = "vocab"
delimiters = "delimiters"

metabolite_vocabs = ["refmet", "HMDB"]
lipid_vocabs = ["HMDB", "LM", "refmet"]


datasets = {
    "arivale_proteins.tsv": {name: "gene_name", ids: ["uniprot", "gene_id"]},
    "arivale_labs.tsv": {name: "Display Name", ids: ["Labcorp LOINC ID", "Quest LOINC ID"]},
    "arivale_metabolites.tsv": {
        name: "CHEMICAL_NAME",
        ids: ["CAS", "KEGG", "HMDB", "PUBCHEM", "INCHIKEY", "SMILES"],
        vocab: metabolite_vocabs,
    },
    "arivale_lipids.tsv": {name: "CHEMICAL_NAME", ids: ["HMDB", "KEGG"], vocab: lipid_vocabs},
    "ukbb_proteins.tsv": {name: "Assay", ids: ["UniProt"], delimiters: ["_"]},
    "ukbb_labs.tsv": {name: "field_name", ids: []},
    "ukbb_metabolites.tsv": {
        name: "nightingale_name",
        ids: ["source_chebi_id", "source_hmdb_id", "source_pubchem_id"],
        vocab: metabolite_vocabs,
    },
    "hpp_proteins.tsv": {name: "nightingale_name", ids: []},
    "hpp_labs.csv": {name: "Description", ids: []},
    "hpp_metabolites.tsv": {name: "nightingale_description", ids: [], vocab: metabolite_vocabs},
    "hpp_lipids.tsv": {name: "Input.name", ids: [], vocab: lipid_vocabs},
}

datasets_dir = PROJECT_ROOT / "data" / "milestone"


mapper = Mapper()
mode: AnnotationMode = "all"
kg_name = "kraken"
results_dir = datasets_dir / "results" / kg_name

for dataset_filename, params in datasets.items():
    entity_type = dataset_filename.split("_")[1].split(".")[0]
    logging.info(f"On dataset {dataset_filename} (entity_type is: {entity_type}")
    tsv_path = datasets_dir / dataset_filename

    mapper.map_dataset_to_kg(
        dataset=tsv_path,
        entity_type=entity_type,
        name_column=params[name],
        provided_id_columns=params[ids],
        array_delimiters=params.get(delimiters),
        vocab=params.get(vocab),
        annotation_mode=mode,
        output_dir=results_dir,
    )


viz = Visualizer(
    config={"row_order": ["proteins", "metabolites", "lipids", "labs"], "col_order": ["arivale", "ukbb", "hpp"]}
)
viz_dir = results_dir / "viz"

stats_df = viz.aggregate_stats(stats_dir=results_dir)
suffix = f"- {kg_name.upper()} (mode={mode})"

viz.render_heatmap(df=stats_df, output_path=viz_dir / f"heatmap_{mode}", title=f"Mapping Coverage {suffix}")

viz.render_breakdown(df=stats_df, output_path=viz_dir / f"breakdown_{mode}", title=f"Mapping Breakdown {suffix}")
