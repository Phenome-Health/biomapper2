"""Vocabulary configuration for loading Biolink prefixes and validator mappings."""

from typing import Any

from ...biolink_client import BiolinkClient
from ...utils import ALIASES_PROP, CLEANER_PROP, VALIDATOR_PROP
from . import cleaners, validators


def load_prefix_info(biolink_client: BiolinkClient) -> dict[str, dict[str, str]]:
    """
    Load Biolink model prefix map and add custom entries.

    Args:
        biolink_client: Biolink Client

    Returns:
        Dictionary mapping lowercase prefixes to {prefix, iri}
    """
    prefix_to_iri_map = biolink_client.get_prefix_map()

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

    # KRAKEN Vocab Prefixes (Issue #12) - add custom prefixes NOT in Biolink
    prefix_to_iri_map["KEGG.ENZYME"] = "https://www.kegg.jp/entry/ec:"
    prefix_to_iri_map["ATC"] = "https://www.whocc.no/atc_ddd_index/?code="
    prefix_to_iri_map["AEO"] = "http://purl.obolibrary.org/obo/AEO_"
    prefix_to_iri_map["UO"] = "http://purl.obolibrary.org/obo/UO_"
    prefix_to_iri_map["EHDAA2"] = "http://purl.obolibrary.org/obo/EHDAA2_"
    prefix_to_iri_map["MOD"] = "http://purl.obolibrary.org/obo/MOD_"
    prefix_to_iri_map["PathWhiz.Bound"] = "https://smpdb.ca/pathwhiz/reactions/"
    prefix_to_iri_map["PathWhiz.Compound"] = "https://smpdb.ca/pathwhiz/metabolites/"
    prefix_to_iri_map["PathWhiz.ElementCollection"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.NucleicAcid"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.ProteinComplex"] = "https://smpdb.ca/pathwhiz/"
    prefix_to_iri_map["PathWhiz.Reaction"] = "https://smpdb.ca/pathwhiz/reactions/"
    prefix_to_iri_map["ICD10PCS"] = "https://www.icd10data.com/ICD10PCS/Codes/"
    prefix_to_iri_map["icd11.foundation"] = "https://icd.who.int/browse11/l-m/en#/"
    prefix_to_iri_map["PDQ"] = "https://www.cancer.gov/publications/pdq"
    prefix_to_iri_map["CHV"] = ""
    prefix_to_iri_map["CDNO"] = "http://purl.obolibrary.org/obo/CDNO_"
    prefix_to_iri_map["PSY"] = ""
    prefix_to_iri_map["ttd.target"] = "https://db.idrblab.net/ttd/data/target/details/"
    prefix_to_iri_map["dictybase.gene"] = "http://dictybase.org/gene/"
    prefix_to_iri_map["AraPort"] = "https://www.arabidopsis.org/servlets/TairObject?accession="
    prefix_to_iri_map["CGNC"] = "https://vertebrate.genenames.org/data/gene-symbol-report/#!/cgnc_id/"
    prefix_to_iri_map["ecogene"] = "https://ecocyc.org/gene?orgid=ECOLI&id="
    prefix_to_iri_map["EnsemblGenomes"] = "https://www.ensemblgenomes.org/id/"
    prefix_to_iri_map["OBA"] = "http://purl.obolibrary.org/obo/OBA_"
    prefix_to_iri_map["OBO"] = "http://purl.obolibrary.org/obo/"

    # Override prefixes only when Biolink's IRI is broken
    prefix_to_iri_map["OMIM"] = "http://purl.bioontology.org/ontology/OMIM/"  # Works for regular ids and MTHU ids
    prefix_to_iri_map["REACT"] = "https://reactome.org/content/detail/"  # Works for Complexes and Pathways

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
    validator = VALIDATOR_PROP
    cleaner = CLEANER_PROP
    aliases = ALIASES_PROP
    # Validators organized alphabetically for easy lookup
    return {
        "aeo": {validator: validators.is_seven_digit_id},
        "ahrq": {validator: validators.is_ahrq_id},
        "araport": {validator: validators.is_araport_id},
        "atc": {validator: validators.is_atc_id},
        "bfo": {validator: validators.is_bfo_id},
        "bspo": {validator: validators.is_seven_digit_id},
        "bvbrc": {validator: validators.is_bvbrc_id},
        "cas": {validator: validators.is_cas_id},
        "cdcsvi": {validator: validators.is_cdcsvi_id},
        "cdno": {validator: validators.is_seven_digit_id},
        "cgnc": {validator: validators.is_numeric_id},
        "chebi": {validator: validators.is_chebi_id},
        "chembl.compound": {validator: validators.is_chembl_compound_id},
        "chembl.mechanism": {validator: validators.is_chembl_mechanism_id},
        "chembl.target": {validator: validators.is_chembl_target_id},
        "chr": {validator: validators.is_chr_id},
        "chv": {validator: validators.is_chv_id},
        "cl": {validator: validators.is_cl_id},
        "clo": {validator: validators.is_clo_id, aliases: ["celllineontology"]},
        "complexportal": {validator: validators.is_complexportal_id},
        "cvcl": {validator: validators.is_cellosaurus_id},
        "cytoband": {validator: validators.is_cytoband_id},
        "dbsnp": {validator: validators.is_dbsnp_id},
        "ddanat": {validator: validators.is_seven_digit_id},
        "dictybase": {validator: validators.is_dictybase_id},
        "dictybase.gene": {validator: validators.is_dictybase_gene_id},
        "doid": {validator: validators.is_doid_id},
        "drugbank": {validator: validators.is_drugbank_id},
        "drugcentral": {validator: validators.is_numeric_id},
        "ec": {validator: validators.is_ec_id, aliases: ["explorenz"]},
        "ecto": {validator: validators.is_numeric_id},
        "efo": {validator: validators.is_efo_id},
        "ehdaa2": {validator: validators.is_seven_digit_id},
        "emapa": {validator: validators.is_numeric_id},
        "ensembl": {validator: validators.is_ensembl_gene_id, aliases: ["gene"]},
        "ensemblgenomes": {validator: validators.is_ensemblgenomes_id},
        "envo": {validator: validators.is_envo_id},
        "fao": {validator: validators.is_seven_digit_id},
        "fb": {validator: validators.is_flybase_id, aliases: ["flybase"]},
        "fbbt": {validator: validators.is_fbbt_id},
        "fips.place": {validator: validators.is_fips_compound_id},
        "fips.state": {validator: validators.is_fips_state_id},
        "fma": {validator: validators.is_numeric_id},
        "foodon": {validator: validators.is_foodon_id},
        "genepio": {validator: validators.is_seven_digit_id},
        "geonames": {validator: validators.is_geonames_id},
        "go": {validator: validators.is_go_id},
        "gtopdb": {validator: validators.is_gtopdb_id},
        "hcpcs": {validator: validators.is_hcpcs_id},
        "hgnc": {validator: validators.is_numeric_id},
        "hmdb": {validator: validators.is_hmdb_id, cleaner: cleaners.clean_hmdb_id},
        "hp": {validator: validators.is_seven_digit_id, aliases: ["hpo"]},
        "hps": {validator: validators.is_hps_id},
        "icd9": {validator: validators.is_icd9_id},
        "icd10": {validator: validators.is_icd10_id},
        "icd10pcs": {validator: validators.is_icd10pcs_id},
        "icd11.foundation": {validator: validators.is_numeric_id},
        "inchikey": {validator: validators.is_inchikey_id},
        "kegg": {validator: validators.is_kegg_generic_id},
        "kegg.compound": {validator: validators.is_kegg_compound_id},
        "kegg.drug": {validator: validators.is_kegg_drug_id},
        "kegg.enzyme": {validator: validators.is_ec_id},
        "kegg.glycan": {validator: validators.is_kegg_glycan_id},
        "kegg.reaction": {validator: validators.is_kegg_reaction_id},
        "lipidbank": {validator: validators.is_lipidbank_id},
        "lm": {
            validator: validators.is_lipidmaps_id,
            cleaner: lambda x: x.removeprefix("LM"),
            aliases: ["lipidmaps"],
        },
        "loinc": {validator: validators.is_loinc_id},
        "maxo": {validator: validators.is_seven_digit_id},
        "meddra": {validator: validators.is_meddra_id},
        "medgen": {validator: validators.is_numeric_id},
        "mesh": {validator: validators.is_mesh_id},
        "metacyc.ec": {validator: validators.is_metacyc_ec_id},
        "metacyc.pathway": {validator: validators.is_metacyc_pathway_id},
        "metacyc.reaction": {validator: validators.is_metacyc_reaction_id},
        "mgi": {validator: validators.is_numeric_id},
        "mi": {validator: validators.is_mi_id},
        "mirbase": {validator: validators.is_mirbase_id},
        "mirdb": {validator: validators.is_mirdb_id},
        "mod": {validator: validators.is_mod_id},
        "mondo": {validator: validators.is_mondo_id},
        "nbo": {validator: validators.is_seven_digit_id},
        "ncbigene": {validator: validators.is_ncbigene_id, aliases: ["entrez", "entrezgene", "gene"]},
        "ncbitaxon": {validator: validators.is_ncbitaxon_id, aliases: ["ncbitaxonomy"]},
        "ncit": {validator: validators.is_ncit_id},
        "nddf": {validator: validators.is_numeric_id},
        "ndfrt": {validator: validators.is_ndfrt_id},
        "nhanes": {validator: validators.is_nhanes_id},
        "oba": {validator: validators.is_oba_id},
        "obi": {validator: validators.is_seven_digit_id},
        "obo": {validator: validators.is_obo_id},
        "omim": {validator: validators.is_omim_id},
        "omim.ps": {validator: validators.is_omim_ps_id},
        "orphanet": {validator: validators.is_numeric_id, aliases: ["orpha"]},
        "pathwhiz": {validator: validators.is_pathwhiz_id},
        "pathwhiz.bound": {validator: validators.is_numeric_id},
        "pathwhiz.compound": {validator: validators.is_numeric_id},
        "pathwhiz.elementcollection": {validator: validators.is_numeric_id},
        "pathwhiz.nucleicacid": {validator: validators.is_numeric_id},
        "pathwhiz.proteincomplex": {validator: validators.is_numeric_id},
        "pathwhiz.reaction": {validator: validators.is_numeric_id},
        "pato": {validator: validators.is_seven_digit_id},
        "pdq": {validator: validators.is_pdq_id},
        "pfam": {validator: validators.is_pfam_id},
        "pharmvar": {validator: validators.is_pharmvar_id},
        "plantfa": {validator: validators.is_plantfa_id},
        "po": {validator: validators.is_seven_digit_id},
        "pombase": {validator: validators.is_pombase_id},
        "pr": {validator: validators.is_pr_id},
        "psy": {validator: validators.is_numeric_id},
        "pubchem.compound": {validator: validators.is_pubchem_compound_id},
        "react": {validator: validators.is_reactome_id, aliases: ["reactome"]},
        "rgd": {validator: validators.is_numeric_id},
        "rhea": {validator: validators.is_numeric_id},
        "rm": {validator: validators.is_refmet_id, cleaner: lambda x: x.removeprefix("RM"), aliases: ["refmet"]},
        "rxcui": {validator: validators.is_numeric_id},
        "rxnorm": {validator: validators.is_numeric_id},
        "sgd": {validator: validators.is_sgd_id},
        "slm": {
            validator: validators.is_slm_id,
            cleaner: lambda x: x.removeprefix("SLM:"),
            aliases: ["swisslipids"],
        },
        "smiles": {validator: validators.is_smiles_string, cleaner: cleaners.get_canonical_smiles},
        "smpdb": {validator: validators.is_smpdb_id},
        "snomedct": {validator: validators.is_snomedct_id, aliases: ["snomed"]},
        "so": {validator: validators.is_seven_digit_id},
        "ttd.target": {validator: validators.is_ttd_target_id},
        "uberon": {validator: validators.is_uberon_id},
        "umls": {validator: validators.is_umls_id},
        "unii": {validator: validators.is_unii_id},
        "uniprotkb": {validator: validators.is_uniprot_id, aliases: ["uniprot"]},
        "uo": {validator: validators.is_seven_digit_id},
        "uszipcode": {validator: validators.is_uszipcode_id, cleaner: cleaners.clean_zipcode},
        "vandf": {validator: validators.is_vandf_id},
        "vesiclepedia": {validator: validators.is_vesiclepedia_id},
        "wb": {validator: validators.is_wormbase_gene_id, aliases: ["wormbase"]},
        "wikipathways": {validator: validators.is_wikipathways_id, cleaner: cleaners.clean_wikipathways_id},
        "zfa": {validator: validators.is_zfa_id},
        "zfin": {validator: validators.is_zfin_id},
    }
