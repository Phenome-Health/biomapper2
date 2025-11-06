import re
from pathlib import Path
from typing import List, Optional, Union, Any

import pandas as pd
import numpy as np

from .kg_mapper import map_to_kg
from .utils import setup_logging

setup_logging()


def map_dataset_to_kg(dataset_tsv_path: Union[str, Path],
                      entity_type: str,
                      name_column: str,
                      provided_id_columns: List[str],
                      array_delimiters: Optional[List[str]] = None):
    array_delimiters = array_delimiters if array_delimiters else [',', ';']
    # TODO: Optionally allow people to input a Dataframe directly, as opposed to TSV path?

    # Load tsv into pandas, skipping comment lines (Arivale uses '#' comments)
    df = pd.read_csv(dataset_tsv_path, sep='\t', comment='#')

    # Do some basic cleanup to try to ensure empty cells are represented consistently
    df[provided_id_columns] = df[provided_id_columns].replace('-', np.nan)
    df[provided_id_columns] = df[provided_id_columns].replace('NO_MATCH', np.nan)
    # TODO: split string arrays at this point..
    # TODO: deal with float local ids ending in .0 automatically?

    # Loop through and map each item
    mapped_records = []
    for _, row in df.iterrows():
        record = row.to_dict()

        # Split any delimited IDs in the provided ID columns into lists
        for id_col in provided_id_columns:
            if id_col in record and pd.notna(record[id_col]):
                # Split and strip whitespace
                ids = re.split(f"[{''.join(array_delimiters)}]", record[id_col])
                record[id_col] = [item.strip() for item in ids if item.strip()]

        # Map the record (this returns the expanded/mapped version)
        mapped_record = map_to_kg(
            record=record,
            name_field=name_column,
            provided_id_fields=provided_id_columns,
            entity_type=entity_type
        )
        mapped_records.append(mapped_record)

    # Convert back to DataFrame and write to TSV
    output_df = pd.DataFrame(mapped_records)
    output_tsv_path = dataset_tsv_path.replace('.tsv', '_MAPPED.tsv')
    output_df.to_csv(output_tsv_path, sep='\t', index=False)

# Run stats on accuracy and vizs

# Just use what I have from other files (don't worry about changing structure for now)


def clean_float_id(local_id: Any):
    if pd.isna(local_id):
        return local_id
    try:
        # If it's a float that equals its int version, convert to int string
        float_val = float(local_id)
        if float_val == int(float_val):
            return str(int(float_val))
    except (ValueError, TypeError):
        pass
    return local_id
