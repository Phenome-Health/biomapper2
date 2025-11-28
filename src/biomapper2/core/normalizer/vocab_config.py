"""Vocabulary configuration for loading Biolink prefixes and validator mappings."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import requests
import yaml

from . import validators, cleaners


def load_prefix_info(biolink_version: str) -> Dict[str, Dict[str, str]]:
    """
    Load Biolink model prefix map and add custom entries.

    Args:
        biolink_version: Biolink model version string

    Returns:
        Dictionary mapping lowercase prefixes to {prefix, iri}
    """
    logging.debug(f"Grabbing biolink prefix map for version: {biolink_version}")
    url = f"https://raw.githubusercontent.com/biolink/biolink-model/refs/tags/v{biolink_version}/project/prefixmap/biolink-model-prefix-map.json"
    prefix_to_iri_map = _load_biolink_file(url, biolink_version)

    # Remove prefixes as needed
    if "KEGG" in prefix_to_iri_map:
        del prefix_to_iri_map["KEGG"]  # We want to use only KEGG.COMPOUND, KEGG.REACTION, etc.

    # Add prefixes as needed (ones we're making up, that don't exist in biolink)
    prefix_to_iri_map["USZIPCODE"] = "https://www.unitedstateszipcodes.org/"
    prefix_to_iri_map["SMILES"] = "https://pubchem.ncbi.nlm.nih.gov/compound/"
    prefix_to_iri_map["CVCL"] = "https://web.expasy.org/cellosaurus/CVCL_"
    prefix_to_iri_map["VESICLEPEDIA"] = "http://microvesicles.org/exp_summary?exp_id="
    prefix_to_iri_map["NDFRT"] = "http://purl.bioontology.org/ontology/NDFRT/"
    prefix_to_iri_map["BVBRC"] = "https://www.bv-brc.org/view/Genome/"
    prefix_to_iri_map["GeoNames"] = (
        "http://www.geonames.org/search.html?q="  # Note: this doesn't go exactly to page for item, but closest I could find
    )
    prefix_to_iri_map["NHANES"] = (
        "https://dsld.od.nih.gov/label/"  # These IRIs work, but weirdly SPOKE's identifiers for these nodes don't match what they have..
    )
    prefix_to_iri_map["MIRDB"] = (
        "https://mirdb.org/cgi-bin/mature_mir.cgi?name="  # Not sure if it's right to use a 'mature' iri like this for all...
    )
    prefix_to_iri_map["CYTOBAND"] = ""  # Haven't found good iri for these yet..
    prefix_to_iri_map["CHR"] = ""  # Country Health Rankings.. Haven't found good iri for these yet
    prefix_to_iri_map["AHRQ"] = ""  # AHRQ SDOH Database
    prefix_to_iri_map["HPS"] = ""  # Household Pulse Survey
    prefix_to_iri_map["mirbase"] = (
        "https://mirbase.org/hairpin/"  # Biolink has mirbase in here, but their iri doesn't work
    )
    prefix_to_iri_map["metacyc.pathway"] = (
        "https://metacyc.org/pathway?orgid=META&id="  # Biolink has metacyc.reaction, but not pathway
    )
    prefix_to_iri_map["metacyc.ec"] = (
        "https://biocyc.org/META/NEW-IMAGE?type=EC-NUMBER&object=EC-"  # Biolink has metacyc.reaction, but not ec (these are like provisional ec codes, not yet in explorenz)
    )
    prefix_to_iri_map["FIPS.PLACE"] = ""
    prefix_to_iri_map["FIPS.STATE"] = ""
    prefix_to_iri_map["PHARMVAR"] = ""  # This wants a number rather than the symbol..
    prefix_to_iri_map["CDCSVI"] = ""  # CDC Social Vulnerability Index
    prefix_to_iri_map["LM"] = "https://www.lipidmaps.org/databases/lmsd/LM"
    prefix_to_iri_map["SLM"] = "https://www.swisslipids.org/#/entity/SLM:"
    prefix_to_iri_map["LIPIDBANK"] = ""  # Could look harder for this iri..
    prefix_to_iri_map["PLANTFA"] = ""  # Could look harder for this iri..
    prefix_to_iri_map["RM"] = "https://www.metabolomicsworkbench.org/databases/refmet/refmet_details.php?REFMET_ID=RM"

    # Override prefixes as needed (if Biolink's iri is broken)
    prefix_to_iri_map["OMIM"] = "http://purl.bioontology.org/ontology/OMIM/"  # Works for regular ids and MTHU ids
    prefix_to_iri_map["REACT"] = "https://reactome.org/content/detail/"  # Works for Complexes and Pathways (I think)

    # Return a mapping of lowercase prefixes to their normalized form (varying capitalization) and IRIs
    vocab_info_map = {
        cleaners.clean_vocab_prefix(prefix): {"prefix": prefix, "iri": iri} for prefix, iri in prefix_to_iri_map.items()
    }

    return vocab_info_map


def load_validator_map() -> Dict[str, Dict[str, Any]]:
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
    with open(local_path, "r") as cache_file:
        contents = json.load(cache_file)
        return contents
