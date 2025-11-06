import copy
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import pandas as pd
import numpy as np

from .core.normalizer import Normalizer
from .core.annotator import annotate
from .core.linker import link, get_kg_ids, get_kg_id_fields
from .core.resolver import resolve
from .utils import setup_logging


setup_logging()


class Mapper:

    def __init__(self):
        self.normalizer = Normalizer()  # Instantiate the ID normalizer (should only be done once, up front)


    def map_entity_to_kg(self,
                         item: pd.Series | Dict[str, Any],
                         name_field: str,
                         provided_id_fields: List[str],
                         entity_type: str,
                         array_delimiters: Optional[List[str]] = None,
                         stop_on_invalid_id: bool = False) -> Dict[str, Any]:
        logging.debug(f"Item at beginning of map_entity_to_kg() is {item}")
        array_delimiters = array_delimiters if array_delimiters is not None else [',', ';']
        mapped_item = copy.deepcopy(item)  # Use a copy to avoid editing input item

        # Do Step 1: annotation of IDs
        assigned_ids = annotate(mapped_item, name_field, provided_id_fields, entity_type)
        mapped_item['assigned_ids'] = assigned_ids

        # Do Step 2: normalization of IDs (curie formation)
        normalizer_result_tuple = self.normalizer.normalize(mapped_item, provided_id_fields, stop_on_invalid_id)
        curies, curies_provided, curies_assigned, invalid_ids, invalid_ids_provided, invalid_ids_assigned = normalizer_result_tuple
        mapped_item['curies'] = curies
        mapped_item['curies_provided'] = curies_provided
        mapped_item['curies_assigned'] = curies_assigned
        mapped_item['invalid_ids'] = invalid_ids
        mapped_item['invalid_ids_provided'] = invalid_ids_provided
        mapped_item['invalid_ids_assigned'] = invalid_ids_assigned

        # Do Step 3: linking to KG nodes
        kg_ids, kg_ids_provided, kg_ids_assigned = link(mapped_item)
        mapped_item['kg_ids'] = kg_ids
        mapped_item['kg_ids_provided'] = kg_ids_provided
        mapped_item['kg_ids_assigned'] = kg_ids_assigned

        # Do Step 4: resolving one-to-many KG matches
        chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned = resolve(mapped_item)
        mapped_item['chosen_kg_id'] = chosen_kg_id
        mapped_item['chosen_kg_id_provided'] = chosen_kg_id_provided
        mapped_item['chosen_kg_id_assigned'] = chosen_kg_id_assigned

        return mapped_item


    def map_dataset_to_kg(self,
                          dataset_tsv_path: str,
                          entity_type: str,
                          name_column: str,
                          provided_id_columns: List[str],
                          array_delimiters: Optional[List[str]] = None) -> pd.DataFrame:
        logging.info(f"Beginning to map dataset to KG ({dataset_tsv_path})")
        array_delimiters = array_delimiters if array_delimiters is not None else [',', ';']

        # TODO: Optionally allow people to input a Dataframe directly, as opposed to TSV path?

        # Load tsv into pandas, skipping comment lines (Arivale uses '#' comments)
        df = pd.read_csv(dataset_tsv_path, sep='\t', comment='#')

        # Do some basic cleanup to try to ensure empty cells are represented consistently
        df[provided_id_columns] = df[provided_id_columns].replace('-', np.nan)
        df[provided_id_columns] = df[provided_id_columns].replace('NO_MATCH', np.nan)

        # Do Step 1: annotate all rows with IDs
        df['assigned_ids'] = df.apply(lambda row: annotate(item=row,
                                                           name_field=name_column,
                                                           provided_id_fields=provided_id_columns,
                                                           entity_type=entity_type),
                                      axis=1)
        logging.info(f"After step 1 (annotation), df is: \n{df}")

        # Do Step 2: normalize IDs in all rows to form proper curies
        df[['curies', 'curies_provided', 'curies_assigned', 'invalid_ids', 'invalid_ids_provided',
            'invalid_ids_assigned']] = df.apply(lambda row: self.normalizer.normalize(item=row,
                                                                                      provided_id_fields=provided_id_columns,
                                                                                      array_delimiters=array_delimiters),
                                                axis=1,
                                                result_type='expand')
        logging.info(f"After step 2 (normalization), df is: \n{df}")

        # Do Step 3: link curies to KG nodes
        # First look up all curies in bulk (way more efficient than sending in separate requests)
        curie_to_kg_id_map = get_kg_ids(list(set(df.curies.explode().dropna())))
        # Then form our new columns using that curie-->kg id map
        df[['kg_ids', 'kg_ids_provided',
            'kg_ids_assigned']] = df.apply(lambda row: get_kg_id_fields(item=row,
                                                                        curie_to_kg_id_map=curie_to_kg_id_map),
                                                                        axis=1,
                                                                        result_type='expand')
        logging.info(f"After step 3 (linking), df is: \n{df}")

        # Do Step 4: resolve one-to-many KG matches
        df[['chosen_kg_id', 'chosen_kg_id_provided', 'chosen_kg_id_assigned']] = df.apply(lambda row: resolve(item=row),
                                                                                          axis=1,
                                                                                          result_type='expand')
        logging.info(f"After step 4 (resolution), df is: \n{df}")

        # Dump the final dataframe to a TSV
        output_tsv_path = dataset_tsv_path.replace('.tsv', '_MAPPED.tsv')  # TODO: let this be configurable?
        logging.info(f"Dumping output TSV to {output_tsv_path}")
        df.to_csv(output_tsv_path, sep='\t', index=False)
