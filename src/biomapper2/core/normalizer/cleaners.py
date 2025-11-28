"""ID cleaning functions for standardizing identifiers."""

import logging
import re

from rdkit import Chem


def clean_vocab_prefix(vocab: str) -> str:
    """Remove non-alphanumeric characters (except periods) from vocabulary name."""
    return re.sub(r"[^a-z0-9.]", "", vocab.lower())


def clean_zipcode(local_id: str) -> str:
    """Strip prefix off of zipcodes like AZ-85039."""
    return local_id.split("-")[-1]


def clean_wikipathways_id(local_id: str) -> str:
    """Get rid of version suffix info, like in WP5395_r126912."""
    return local_id.split("_")[0]


def clean_hmdb_id(local_id: str) -> str:
    """Clean HMDB identifiers."""
    # Remove any double-HMDB prefix
    if local_id.startswith("HMDBHMDB"):
        local_id = local_id.removeprefix("HMDB")

    # Convert any 5-digit HMDB local IDs to current 7-digit form
    if local_id.startswith("HMDB") and len(local_id) == 9:
        id_digits = local_id.removeprefix("HMDB")
        return f"HMDB00{id_digits}"
    else:
        return local_id


def get_canonical_smiles(smiles_string: str) -> str:
    """
    Convert a SMILES string to its canonical form using RDKit

    Args:
        smiles_string (str): Input SMILES string

    Returns:
        str: Canonical SMILES string, or None if invalid
    """
    try:
        # Parse the SMILES string into a molecule object
        mol = Chem.MolFromSmiles(smiles_string)

        # Check if parsing was successful
        if mol is None:
            logging.warning(
                f"SMILES string '{smiles_string}' is invalid according to rdkit. "
                f"Cannot convert it to canonical form."
            )
            return smiles_string

        # Generate canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles

    except Exception as e:
        logging.warning(f"Error getting canonical SMILES for '{smiles_string}'; will use input SMILES string. {e}")
        return smiles_string
