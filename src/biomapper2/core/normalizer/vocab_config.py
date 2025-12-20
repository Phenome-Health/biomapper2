"""Vocabulary configuration for loading Biolink prefixes and validator mappings."""

import json
import logging
from pathlib import Path
from typing import Any

import requests
import yaml

from . import cleaners, validators


def load_prefix_info(biolink_version: str) -> dict[str, dict[str, str]]:
    """
    Load Biolink model prefix map and add custom entries.

    Args:
        biolink_version: Biolink model version string

    Returns:
        Dictionary mapping lowercase prefixes to {prefix, iri}
    """
    logging.debug(f"Grabbing biolink prefix map for version: {biolink_version}")
    url = (
        f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{biolink_version}/"
        f"project/prefixmap/biolink-model-prefix-map.json"
    )
    prefix_to_iri_map = _load_biolink_file(url, biolink_version)

    # Add prefixes as needed (ones we're making up, that don't exist in biolink)
    prefix_to_iri_map["USZIPCODE"] = "https://www.unitedstateszipcodes.org/"
    prefix_to_iri_map["SMILES"] = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    prefix_to_iri_map["CVCL"] = "https://web.expasy.org/cellosaurus/CVCL_"
    prefix_to_iri_map["VESICLEPEDIA"] = "http://microvesicles.org/exp_summary?exp_id="
    prefix_to_iri_map["NDFRT"] = "http://purl.bioontology.org/ontology/NDFRT/"
    prefix_to_iri_map["BVBRC"] = "https://www.bv-brc.org/view/Genome/"
    # Note: below doesn't go exactly to page for item, but closest I could find
    prefix_to_iri_map["GeoNames"] = "http://www.geonames.org/search.html?q="
    # Below IRI works, but weirdly SPOKE's identifiers for these nodes don't match what they have..
    prefix_to_iri_map["NHANES"] = "https://dsld.od.nih.gov/label/"
    # Not sure if it's right to use a 'mature' iri like this for all...
    prefix_to_iri_map["MIRDB"] = "https://mirdb.org/cgi-bin/mature_mir.cgi?name="
    prefix_to_iri_map["CYTOBAND"] = ""  # Haven't found good iri for these yet..
    prefix_to_iri_map["CHR"] = ""  # Country Health Rankings.. Haven't found good iri for these yet
    prefix_to_iri_map["AHRQ"] = ""  # AHRQ SDOH Database
    prefix_to_iri_map["HPS"] = ""  # Household Pulse Survey
    prefix_to_iri_map["mirbase"] = "https://mirbase.org/hairpin/"  # Biolink has mirbase, but their iri doesn't work
    # Note: Biolink has metacyc.reaction, but not pathway or ec
    prefix_to_iri_map["metacyc.pathway"] = "https://metacyc.org/pathway?orgid=META&id="
    prefix_to_iri_map["metacyc.ec"] = "https://biocyc.org/META/NEW-IMAGE?type=EC-NUMBER&object=EC-"
    prefix_to_iri_map["FIPS.PLACE"] = ""
    prefix_to_iri_map["FIPS.STATE"] = ""
    prefix_to_iri_map["PHARMVAR"] = ""  # This wants a number rather than the symbol..
    prefix_to_iri_map["CDCSVI"] = ""  # CDC Social Vulnerability Index
    prefix_to_iri_map["LM"] = "https://www.lipidmaps.org/databases/lmsd/LM"
    prefix_to_iri_map["SLM"] = "https://www.swisslipids.org/#/entity/SLM:"
    prefix_to_iri_map["LIPIDBANK"] = ""  # Could look harder for this iri..
    prefix_to_iri_map["PLANTFA"] = ""  # Could look harder for this iri..
    prefix_to_iri_map["RM"] = "https://www.metabolomicsworkbench.org/databases/refmet/refmet_details.php?REFMET_ID=RM"

    # KRAKEN Vocab Prefixes (Issue #12) - add custom prefixes not in Biolink
    prefix_to_iri_map["KEGG.ENZYME"] = "https://www.kegg.jp/entry/ec:"
    prefix_to_iri_map["KEGG.GLYCAN"] = "https://www.kegg.jp/entry/"
    prefix_to_iri_map["HGNC"] = "https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:"
    prefix_to_iri_map["MGI"] = "http://www.informatics.jax.org/marker/MGI:"
    prefix_to_iri_map["RGD"] = "https://rgd.mcw.edu/rgdweb/report/gene/main.html?id="
    prefix_to_iri_map["RXNORM"] = "https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm="
    prefix_to_iri_map["RXCUI"] = "https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm="
    prefix_to_iri_map["DrugCentral"] = "https://drugcentral.org/drugcard/"
    prefix_to_iri_map["ATC"] = "https://www.whocc.no/atc_ddd_index/?code="
    prefix_to_iri_map["RHEA"] = "https://www.rhea-db.org/rhea/"
    prefix_to_iri_map["UNII"] = "https://precision.fda.gov/uniisearch/srs/unii/"
    prefix_to_iri_map["OMIM.PS"] = "https://www.omim.org/phenotypicSeries/PS"
    prefix_to_iri_map["orphanet"] = "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=en&Expert="
    prefix_to_iri_map["PR"] = "https://proconsortium.org/app/entry/PR:"
    prefix_to_iri_map["SMPDB"] = "https://smpdb.ca/view/"
    # Tier 2: Anatomy/Phenotype Ontologies
    prefix_to_iri_map["AEO"] = "http://purl.obolibrary.org/obo/AEO_"
    prefix_to_iri_map["BSPO"] = "http://purl.obolibrary.org/obo/BSPO_"
    prefix_to_iri_map["PATO"] = "http://purl.obolibrary.org/obo/PATO_"
    prefix_to_iri_map["PO"] = "http://purl.obolibrary.org/obo/PO_"
    prefix_to_iri_map["SO"] = "http://purl.obolibrary.org/obo/SO_"
    prefix_to_iri_map["NBO"] = "http://purl.obolibrary.org/obo/NBO_"
    prefix_to_iri_map["OBI"] = "http://purl.obolibrary.org/obo/OBI_"
    prefix_to_iri_map["UO"] = "http://purl.obolibrary.org/obo/UO_"
    prefix_to_iri_map["FAO"] = "http://purl.obolibrary.org/obo/FAO_"
    prefix_to_iri_map["DDANAT"] = "http://purl.obolibrary.org/obo/DDANAT_"
    prefix_to_iri_map["GENEPIO"] = "http://purl.obolibrary.org/obo/GENEPIO_"
    prefix_to_iri_map["MAXO"] = "http://purl.obolibrary.org/obo/MAXO_"
    prefix_to_iri_map["ZFA"] = "http://purl.obolibrary.org/obo/ZFA_"
    prefix_to_iri_map["FBbt"] = "http://purl.obolibrary.org/obo/FBbt_"
    prefix_to_iri_map["EMAPA"] = "http://purl.obolibrary.org/obo/EMAPA_"
    prefix_to_iri_map["EHDAA2"] = "http://purl.obolibrary.org/obo/EHDAA2_"
    prefix_to_iri_map["MOD"] = "http://purl.obolibrary.org/obo/MOD_"
    prefix_to_iri_map["MI"] = "http://purl.obolibrary.org/obo/MI_"
    # Tier 3: Specialized/Medical
    prefix_to_iri_map["PathWhiz"] = "https://smpdb.ca/pathwhiz/pathways/"
    prefix_to_iri_map["PathWhiz.Bound"] = "https://smpdb.ca/pathwhiz/reactions/"
    prefix_to_iri_map["PathWhiz.Compound"] = "https://smpdb.ca/pathwhiz/metabolites/"
    prefix_to_iri_map["PathWhiz.ElementCollection"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.NucleicAcid"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.ProteinComplex"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.Reaction"] = "https://smpdb.ca/pathwhiz/reactions/"
    prefix_to_iri_map["MEDDRA"] = "http://purl.bioontology.org/ontology/MEDDRA/"
    prefix_to_iri_map["medgen"] = "https://www.ncbi.nlm.nih.gov/medgen/"
    prefix_to_iri_map["ICD10PCS"] = "https://www.icd10data.com/ICD10PCS/Codes/"
    prefix_to_iri_map["icd11.foundation"] = "https://icd.who.int/browse11/l-m/en#/"
    prefix_to_iri_map["HCPCS"] = "https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/"
    prefix_to_iri_map["NDDF"] = ""
    prefix_to_iri_map["VANDF"] = ""
    prefix_to_iri_map["GTOPDB"] = "https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId="
    prefix_to_iri_map["PDQ"] = "https://www.cancer.gov/publications/pdq"
    prefix_to_iri_map["CHV"] = ""
    prefix_to_iri_map["CDNO"] = "http://purl.obolibrary.org/obo/CDNO_"
    prefix_to_iri_map["ECTO"] = "http://purl.obolibrary.org/obo/ECTO_"
    prefix_to_iri_map["FOODON"] = "http://purl.obolibrary.org/obo/FOODON_"
    prefix_to_iri_map["PSY"] = ""
    prefix_to_iri_map["ttd.target"] = "https://db.idrblab.net/ttd/data/target/details/"
    # Tier 4: Model Organism Databases
    prefix_to_iri_map["FB"] = "https://flybase.org/reports/"
    prefix_to_iri_map["WB"] = "https://wormbase.org/species/c_elegans/gene/"
    prefix_to_iri_map["ZFIN"] = "https://zfin.org/"
    prefix_to_iri_map["SGD"] = "https://www.yeastgenome.org/locus/"
    prefix_to_iri_map["PomBase"] = "https://www.pombase.org/gene/"
    prefix_to_iri_map["dictyBase"] = "http://dictybase.org/gene/"
    prefix_to_iri_map["dictybase.gene"] = "http://dictybase.org/gene/"
    prefix_to_iri_map["AraPort"] = "https://www.arabidopsis.org/servlets/TairObject?accession="
    prefix_to_iri_map["CGNC"] = "https://vertebrate.genenames.org/data/gene-symbol-report/#!/cgnc_id/"
    prefix_to_iri_map["ecogene"] = "https://ecocyc.org/gene?orgid=ECOLI&id="
    prefix_to_iri_map["EnsemblGenomes"] = "https://www.ensemblgenomes.org/id/"
    # Final 4 edge case prefixes
    prefix_to_iri_map["OBA"] = "http://purl.obolibrary.org/obo/OBA_"
    prefix_to_iri_map["OBO"] = "http://purl.obolibrary.org/obo/"
    prefix_to_iri_map["CHEMBL.MECHANISM"] = "https://www.ebi.ac.uk/chembl/mechanism/"

    # Override prefixes as needed (if Biolink's iri is broken)
    prefix_to_iri_map["OMIM"] = "http://purl.bioontology.org/ontology/OMIM/"  # Works for regular ids and MTHU ids
    prefix_to_iri_map["REACT"] = "https://reactome.org/content/detail/"  # Works for Complexes and Pathways (I think)

    # Return a mapping of lowercase prefixes to their normalized form (varying capitalization) and IRIs
    vocab_info_map = {
        cleaners.clean_vocab_prefix(prefix): {"prefix": prefix, "iri": iri} for prefix, iri in prefix_to_iri_map.items()
    }

    return vocab_info_map


def load_validator_map() -> dict[str, dict[str, Any]]:
    """
    Load vocabulary validator/cleaner function mappings.

    Returns:
        Dictionary mapping vocab names to validator, cleaner, and alias configs
    """
    return {
        "ahrq": {"validator": validators.is_ahrq_id},
        "bfo": {"validator": validators.is_bfo_id},
        "bvbrc": {"validator": validators.is_bvbrc_id},
        "cas": {"validator": validators.is_cas_id},
        "cdcsvi": {"validator": validators.is_cdcsvi_id},
        "chebi": {"validator": validators.is_chebi_id},
        "chembl.compound": {"validator": validators.is_chembl_compound_id},
        "chembl.target": {"validator": validators.is_chembl_target_id},
        "chr": {"validator": validators.is_chr_id},
        "cl": {"validator": validators.is_cl_id},
        "clo": {"validator": validators.is_clo_id, "aliases": ["celllineontology"]},
        "complexportal": {"validator": validators.is_complexportal_id},
        "cvcl": {"validator": validators.is_cellosaurus_id},
        "cytoband": {"validator": validators.is_cytoband_id},
        "dbsnp": {"validator": validators.is_dbsnp_id},
        "doid": {"validator": validators.is_doid_id},
        "drugbank": {"validator": validators.is_drugbank_id},
        "ec": {"validator": validators.is_ec_id, "aliases": ["explorenz"]},
        "efo": {"validator": validators.is_efo_id},
        "ensembl": {"validator": validators.is_ensembl_gene_id},
        "envo": {"validator": validators.is_envo_id},
        "fips.place": {"validator": validators.is_fips_compound_id},
        "fips.state": {"validator": validators.is_fips_state_id},
        "geonames": {"validator": validators.is_geonames_id},
        "go": {"validator": validators.is_go_id},
        "hmdb": {"validator": validators.is_hmdb_id, "cleaner": cleaners.clean_hmdb_id},
        "hps": {"validator": validators.is_hps_id},
        "icd9": {"validator": validators.is_icd9_id},
        "icd10": {"validator": validators.is_icd10_id},
        "inchikey": {"validator": validators.is_inchikey_id},
        "kegg.compound": {"validator": validators.is_kegg_compound_id},
        "kegg.drug": {"validator": validators.is_kegg_drug_id},
        "kegg.reaction": {"validator": validators.is_kegg_reaction_id},
        "lipidbank": {"validator": validators.is_lipidbank_id},
        "loinc": {"validator": validators.is_loinc_id},
        "lm": {
            "validator": validators.is_lipidmaps_id,
            "cleaner": lambda x: x.removeprefix("LM"),
            "aliases": ["lipidmaps"],
        },
        "mesh": {"validator": validators.is_mesh_id},
        "metacyc.ec": {"validator": validators.is_metacyc_ec_id},
        "metacyc.pathway": {"validator": validators.is_metacyc_pathway_id},
        "metacyc.reaction": {"validator": validators.is_metacyc_reaction_id},
        "mirbase": {"validator": validators.is_mirbase_id},
        "mirdb": {"validator": validators.is_mirdb_id},
        "mondo": {"validator": validators.is_mondo_id},
        "ncbigene": {"validator": validators.is_ncbigene_id, "aliases": ["entrez", "entrezgene"]},
        "ncbitaxon": {"validator": validators.is_ncbitaxon_id, "aliases": ["ncbitaxonomy"]},
        "ncit": {"validator": validators.is_ncit_id},
        "ndfrt": {"validator": validators.is_ndfrt_id},
        "nhanes": {"validator": validators.is_nhanes_id},
        "omim": {"validator": validators.is_omim_id},
        "pfam": {"validator": validators.is_pfam_id},
        "pharmvar": {"validator": validators.is_pharmvar_id},
        "pubchem.compound": {"validator": validators.is_pubchem_compound_id},
        "plantfa": {"validator": validators.is_plantfa_id},
        "react": {"validator": validators.is_reactome_id, "aliases": ["reactome"]},
        "rm": {"validator": validators.is_refmet_id, "cleaner": lambda x: x.removeprefix("RM"), "aliases": ["refmet"]},
        "slm": {
            "validator": validators.is_slm_id,
            "cleaner": lambda x: x.removeprefix("SLM:"),
            "aliases": ["swisslipids"],
        },
        "smiles": {"validator": validators.is_smiles_string, "cleaner": cleaners.get_canonical_smiles},
        "snomedct": {"validator": validators.is_snomedct_id, "aliases": ["snomed"]},
        "uberon": {"validator": validators.is_uberon_id},
        "umls": {"validator": validators.is_umls_id},
        "uniprotkb": {"validator": validators.is_uniprot_id, "aliases": ["uniprot"]},
        "uszipcode": {"validator": validators.is_uszipcode_id, "cleaner": cleaners.clean_zipcode},
        "vesiclepedia": {"validator": validators.is_vesiclepedia_id},
        "wikipathways": {"validator": validators.is_wikipathways_id, "cleaner": cleaners.clean_wikipathways_id},
        # =========================================================================
        # KRAKEN Vocab Validators (Issue #12)
        # =========================================================================
        # --- Tier 1: Core Metabolomics/Proteomics/Drugs ---
        "hgnc": {"validator": validators.is_numeric_id},
        "mgi": {"validator": validators.is_numeric_id},
        "rgd": {"validator": validators.is_numeric_id},
        "rxnorm": {"validator": validators.is_numeric_id},
        "rxcui": {"validator": validators.is_numeric_id},
        "drugcentral": {"validator": validators.is_numeric_id},
        "atc": {"validator": validators.is_atc_id},
        "rhea": {"validator": validators.is_numeric_id},
        "unii": {"validator": validators.is_unii_id},
        "hp": {"validator": validators.is_seven_digit_id, "aliases": ["hpo"]},
        "omim.ps": {"validator": validators.is_omim_ps_id},
        "orphanet": {"validator": validators.is_numeric_id, "aliases": ["orpha"]},
        "pr": {"validator": validators.is_pr_id},
        "smpdb": {"validator": validators.is_smpdb_id},
        "kegg.enzyme": {"validator": validators.is_ec_id},  # Reuse existing EC validator
        "kegg.glycan": {"validator": validators.is_kegg_glycan_id},
        # --- Tier 2: Anatomy/Phenotype Ontologies ---
        "aeo": {"validator": validators.is_seven_digit_id},
        "bspo": {"validator": validators.is_seven_digit_id},
        "pato": {"validator": validators.is_seven_digit_id},
        "po": {"validator": validators.is_seven_digit_id},
        "so": {"validator": validators.is_seven_digit_id},
        "nbo": {"validator": validators.is_seven_digit_id},
        "obi": {"validator": validators.is_seven_digit_id},
        "uo": {"validator": validators.is_seven_digit_id},
        "fao": {"validator": validators.is_seven_digit_id},
        "ddanat": {"validator": validators.is_seven_digit_id},
        "genepio": {"validator": validators.is_seven_digit_id},
        "maxo": {"validator": validators.is_seven_digit_id},
        "fma": {"validator": validators.is_numeric_id},
        "zfa": {"validator": validators.is_zfa_id},
        "fbbt": {"validator": validators.is_fbbt_id},
        "emapa": {"validator": validators.is_numeric_id},
        "ehdaa2": {"validator": validators.is_seven_digit_id},
        "mod": {"validator": validators.is_mod_id},
        "mi": {"validator": validators.is_mi_id},
        # --- Tier 3: Specialized/Medical ---
        "pathwhiz": {"validator": validators.is_pathwhiz_id},
        "pathwhiz.bound": {"validator": validators.is_numeric_id},
        "pathwhiz.compound": {"validator": validators.is_numeric_id},
        "pathwhiz.elementcollection": {"validator": validators.is_numeric_id},
        "pathwhiz.nucleicacid": {"validator": validators.is_numeric_id},
        "pathwhiz.proteincomplex": {"validator": validators.is_numeric_id},
        "pathwhiz.reaction": {"validator": validators.is_numeric_id},
        "meddra": {"validator": validators.is_meddra_id},
        "medgen": {"validator": validators.is_numeric_id},
        "icd10pcs": {"validator": validators.is_icd10pcs_id},
        "icd11.foundation": {"validator": validators.is_numeric_id},
        "hcpcs": {"validator": validators.is_hcpcs_id},
        "nddf": {"validator": validators.is_numeric_id},
        "vandf": {"validator": validators.is_vandf_id},
        "gtopdb": {"validator": validators.is_gtopdb_id},
        "pdq": {"validator": validators.is_pdq_id},
        "chv": {"validator": validators.is_chv_id},
        "cdno": {"validator": validators.is_seven_digit_id},
        "ecto": {"validator": validators.is_numeric_id},
        "foodon": {"validator": validators.is_foodon_id},
        "psy": {"validator": validators.is_numeric_id},
        "ttd.target": {"validator": validators.is_ttd_target_id},
        # --- Final 4: Edge case prefixes ---
        "kegg": {"validator": validators.is_kegg_generic_id},  # Generic KEGG pathway IDs
        "chembl.mechanism": {"validator": validators.is_chembl_mechanism_id},
        "oba": {"validator": validators.is_oba_id},  # Ontology for Biomedical Annotations
        "obo": {"validator": validators.is_obo_id},  # Open Biological Ontology cross-refs
        # --- Tier 4: Model Organism Databases ---
        "fb": {"validator": validators.is_flybase_id, "aliases": ["flybase"]},
        "wb": {"validator": validators.is_wormbase_gene_id, "aliases": ["wormbase"]},
        "zfin": {"validator": validators.is_zfin_id},
        "sgd": {"validator": validators.is_sgd_id},
        "pombase": {"validator": validators.is_pombase_id},
        "dictybase": {"validator": validators.is_dictybase_id},
        "dictybase.gene": {"validator": validators.is_dictybase_gene_id},
        "araport": {"validator": validators.is_araport_id},
        "cgnc": {"validator": validators.is_numeric_id},
        "ecogene": {"validator": validators.is_ecogene_id},
        "ensemblgenomes": {"validator": validators.is_ensemblgenomes_id},
    }


def _load_biolink_file(url: str, biolink_version: str) -> dict:
    """
    Download and cache Biolink model file.

    Args:
        url: URL to Biolink JSON/YAML file
        biolink_version: Version string for cache naming

    Returns:
        Parsed JSON content
    """
    project_root = Path(__file__).parents[4]

    cache_dir = project_root / "cache"
    file_name = url.split("/")[-1]
    file_name_json = file_name.split(".")[0] + f"_{biolink_version}" + ".json"
    local_path = cache_dir / file_name_json
    logging.debug(f"Local file path is: {local_path}")

    # Download the file if we don't already have it cached
    if not local_path.exists():
        logging.info(f"Downloading YAML file from {url}. local path is: {local_path}")
        response = requests.get(url)
        response.raise_for_status()
        if file_name.endswith(".yaml"):
            response_json = yaml.safe_load(response.text)
        else:
            response_json = response.json()

        # Cache the response
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w+") as cache_file:
            json.dump(response_json, cache_file, indent=2)

    # Read and return the cached JSON
    with open(local_path) as cache_file:
        contents = json.load(cache_file)
        return contents
