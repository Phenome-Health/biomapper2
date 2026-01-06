import pandas as pd

from biomapper2.config import PROJECT_ROOT

datasets_dir = PROJECT_ROOT / "data" / "milestone"


# Prep UKBB clinical labs (using Trent's result file from original biomapper)
print("Prepping UKBB clinical labs tsv..")
biomapper_results_path = datasets_dir / "ukbb_chemistry_COMPLETE.tsv"

# Filter our QC field rows (since they aren't actually clinical labs)
fieldnames_path = datasets_dir / "clinicallab_fieldnames.tsv"  # Copied from web (per Lance)
fieldnames_df = pd.read_table(fieldnames_path)
field_names = set(fieldnames_df["Description"].unique())
print(f"Field names to filter to: {field_names}")
df = pd.read_table(biomapper_results_path)
df = df[df["field_name"].isin(field_names)]

# Remove Trent's mapping columns
df = df[["field_name"]]

df.to_csv(datasets_dir / "ukbb_labs.tsv", sep="\t")
print("Done prepping UKBB clinical labs tsv.")


# Prep UKBB metabolites (remove Trent's result cols)
print("Prepping UKBB metabolites tsv..")
df = pd.read_table(datasets_dir / "ukbb_metabolites_COMPLETE_manuallyedited.tsv")
df = df.drop(
    columns=[
        "kraken_node_id",
        "kraken_name",
        "kraken_category",
        "mapping_confidence",
        "mapping_type",
        "mapping_stage",
        "mapping_timestamp",
        "mapping_status",
        "available_chebi_id",
        "available_hmdb_id",
        "available_pubchem_id",
        "reason_unmatched",
        "integration_timestamp",
    ]
)
df.to_csv(datasets_dir / "ukbb_metabolites.tsv", sep="\t")
print("Done prepping UKBB metabolites tsv.")


# Prep HPP proteins (remove Trent's result cols)
print("Prepping HPP proteins tsv..")
df = pd.read_table(datasets_dir / "israeli10k_nightingale_proteins_mapped.tsv")
df = df[["nightingale_biomarker_id", "nightingale_name", "biomarker_description", "units"]]
df.to_csv(datasets_dir / "hpp_proteins.tsv", sep="\t")
print("Done prepping HPP proteins tsv.")

# Prep HPP metabolites (remove Trent's result cols)
print("Prepping HPP metabolites tsv..")
df = pd.read_table(datasets_dir / "israeli10k_metabolites_COMPLETE.tsv")
df = df[["nightingale_biomarker_id", "nightingale_name", "nightingale_description", "unit", "nightingale_category"]]
df.to_csv(datasets_dir / "hpp_metabolites.tsv", sep="\t")
print("Done prepping HPP metabolites tsv.")

# Prep HPP lipids (remove refmet annotation cols)
print("Prepping HPP lipids tsv..")
df = pd.read_table(datasets_dir / "Israeli10K_website_lipidomics_metadata_REFMETANNOT.tsv")
df = df[["Input.name"]]
df.to_csv(datasets_dir / "hpp_lipids.tsv", sep="\t")
print("Done prepping HPP lipids tsv.")
