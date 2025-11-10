"""
Main mapper module for entity and dataset knowledge graph mapping.

Provides the Mapper class for harmonizing biological entities to knowledge graphs
through annotation, normalization, linking, and resolution steps.
"""
import ast
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from .core.normalizer import Normalizer
from .core.annotator import annotate
from .core.linker import link, get_kg_ids, get_kg_id_fields
from .core.resolver import resolve
from .utils import setup_logging, safe_divide, calculate_f1_score

setup_logging()


class Mapper:
    """
    Maps biological entities and datasets to knowledge graph nodes.

    Performs four-step mapping pipeline:
    1. Annotation - assign additional IDs via external APIs
    2. Normalization - convert local IDs to Biolink-standard curies
    3. Linking - map curies to knowledge graph nodes
    4. Resolution - resolve one-to-many mappings
    """

    def __init__(self, biolink_version: Optional[str] = None):
        self.normalizer = Normalizer()  # Instantiate the ID normalizer (should only be done once, up front)


    def map_entity_to_kg(self,
                         item: pd.Series | Dict[str, Any],
                         name_field: str,
                         provided_id_fields: List[str],
                         entity_type: str,
                         array_delimiters: Optional[List[str]] = None,
                         stop_on_invalid_id: bool = False) -> pd.Series | Dict[str, Any]:
        """
        Map a single entity to knowledge graph nodes.

        Args:
            item: Entity with name and ID fields
            name_field: Field containing entity name
            provided_id_fields: List of fields containing local identifiers
            entity_type: Type of entity (e.g., 'metabolite', 'protein')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            stop_on_invalid_id: Halt execution on invalid IDs (default: False)

        Returns:
            Mapped entity with added fields: curies, kg_ids, chosen_kg_id, etc.
        """
        logging.debug(f"Item at beginning of map_entity_to_kg() is {item}")
        array_delimiters = array_delimiters if array_delimiters is not None else [',', ';']
        mapped_item = copy.deepcopy(item)  # Use a copy to avoid editing input item

        # Do Step 1: annotation of IDs
        assigned_ids = annotate(mapped_item, name_field, provided_id_fields, entity_type)
        mapped_item['assigned_ids'] = assigned_ids

        # Do Step 2: normalization of IDs (curie formation)
        normalizer_result_tuple = self.normalizer.normalize(mapped_item, provided_id_fields, array_delimiters, stop_on_invalid_id)
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
                          array_delimiters: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Map all entities in a dataset to knowledge graph nodes.

        Args:
            dataset_tsv_path: Path to TSV file containing dataset
            entity_type: Type of entities (e.g., 'metabolite', 'protein')
            name_column: Column containing entity names
            provided_id_columns: Columns containing local identifiers
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])

        Returns:
            Tuple of (output_tsv_path, stats_summary)
        """
        logging.info(f"Beginning to map dataset to KG ({dataset_tsv_path})")
        array_delimiters = array_delimiters if array_delimiters is not None else [',', ';']

        # TODO: Optionally allow people to input a Dataframe directly, as opposed to TSV path?

        # Load tsv into pandas
        df = pd.read_csv(dataset_tsv_path, sep='\t', dtype={id_col: str for id_col in provided_id_columns})

        # Do some basic cleanup to try to ensure empty cells are represented consistently
        df[provided_id_columns] = df[provided_id_columns].replace('-', np.nan)
        df[provided_id_columns] = df[provided_id_columns].replace('NO_MATCH', np.nan)

        # Do Step 1: annotate all rows with IDs
        df['assigned_ids'] = df.apply(
            lambda row: annotate(item=row,
                                 name_field=name_column,
                                 provided_id_fields=provided_id_columns,
                                 entity_type=entity_type),
            axis=1
        )
        logging.info(f"After step 1 (annotation), df is: \n{df}")

        # Do Step 2: normalize IDs in all rows to form proper curies
        df[['curies', 'curies_provided', 'curies_assigned', 'invalid_ids', 'invalid_ids_provided', 'invalid_ids_assigned']] = df.apply(
            lambda row: self.normalizer.normalize(item=row,
                                                  provided_id_fields=provided_id_columns,
                                                  array_delimiters=array_delimiters),
            axis=1,
            result_type='expand'
        )
        logging.info(f"After step 2 (normalization), df is: \n{df}")

        # Do Step 3: link curies to KG nodes
        # First look up all curies in bulk (way more efficient than sending in separate requests)
        curie_to_kg_id_map = get_kg_ids(list(set(df.curies.explode().dropna())))
        # Then form our new columns using that curie-->kg id map
        df[['kg_ids', 'kg_ids_provided', 'kg_ids_assigned']] = df.apply(
            lambda row: get_kg_id_fields(item=row,
                                         curie_to_kg_id_map=curie_to_kg_id_map),
            axis=1,
            result_type='expand'
        )
        logging.info(f"After step 3 (linking), df is: \n{df}")

        # Do Step 4: resolve one-to-many KG matches
        df[['chosen_kg_id', 'chosen_kg_id_provided', 'chosen_kg_id_assigned']] = df.apply(
            lambda row: resolve(item=row),
            axis=1,
            result_type='expand'
        )
        logging.info(f"After step 4 (resolution), df is: \n{df}")

        # Dump the final dataframe to a TSV
        output_tsv_path = dataset_tsv_path.replace('.tsv', '_MAPPED.tsv')  # TODO: let this be configurable?
        logging.info(f"Dumping output TSV to {output_tsv_path}")
        df.to_csv(output_tsv_path, sep='\t', index=False)

        stats_summary = self.analyze_dataset_mapping(output_tsv_path)

        return output_tsv_path, stats_summary


    @staticmethod
    def analyze_dataset_mapping(results_tsv_path: str) -> Dict[str, Any]:
        """
        Analyze dataset mapping results and generate summary statistics.

        Args:
            results_tsv_path: Path to mapped dataset TSV

        Returns:
            Dictionary containing coverage, precision, recall, and F1 metrics
        """
        logging.info(f"Analyzing dataset KG mapping in {results_tsv_path}")

        cols_to_literal_eval = [
            'curies', 'curies_provided', 'curies_assigned',
            'invalid_ids', 'invalid_ids_provided', 'invalid_ids_assigned',
            'kg_ids', 'kg_ids_provided', 'kg_ids_assigned'
        ]
        converters = {col: ast.literal_eval for col in cols_to_literal_eval}
        df = pd.read_table(results_tsv_path, converters=converters)

        # Make sure we load any groundtruth column properly
        if 'kg_ids_groundtruth' in df.columns:
            df.kg_ids_groundtruth = df.kg_ids_groundtruth.apply(ast.literal_eval)

        # Calculate some summary stats
        total_items = len(df)
        has_valid_ids = df.curies.apply(lambda x: len(x) > 0).sum()
        has_valid_ids_provided = df.curies_provided.apply(lambda x: len(x) > 0).sum()
        has_valid_ids_assigned = df.curies_assigned.apply(lambda x: len(x) > 0).sum()
        has_only_provided_ids = has_valid_ids - has_valid_ids_assigned
        has_only_assigned_ids = has_valid_ids - has_valid_ids_provided
        has_both_provided_and_assigned_ids = has_valid_ids - has_only_provided_ids - has_only_assigned_ids
        has_no_ids = ((df.curies.apply(len) == 0) & (df.invalid_ids.apply(len) == 0)).sum()
        has_invalid_ids = df.invalid_ids.apply(lambda x: len(x) > 0).sum()
        has_invalid_ids_provided = df.invalid_ids_provided.apply(lambda x: len(x) > 0).sum()
        has_invalid_ids_assigned = df.invalid_ids_assigned.apply(lambda x: len(x) > 0).sum()
        mapped_to_kg = df.kg_ids.apply(lambda x: len(x) > 0).sum()
        mapped_to_kg_provided = df.kg_ids_provided.apply(lambda x: len(x) > 0).sum()
        mapped_to_kg_assigned = df.kg_ids_assigned.apply(lambda x: len(x) > 0).sum()
        mapped_to_kg_provided_and_assigned = df.apply(
            lambda r: (len(r.kg_ids_provided) > 0) & (len(r.kg_ids_assigned) > 0),
            axis=1).sum()
        assigned_mappings_correct_per_provided = df.apply(
            lambda r: len(set(r.kg_ids_provided) & set(r.kg_ids_assigned)) > 0,
            axis=1).sum()
        assigned_mappings_correct_per_provided_chosen = ((df.chosen_kg_id_provided == df.chosen_kg_id_assigned) & df.chosen_kg_id_provided.notna() & df.chosen_kg_id_assigned.notna()).sum()
        has_invalid_ids_and_not_mapped_to_kg = ((df.invalid_ids.apply(len) > 0) & (df.kg_ids.apply(len) == 0)).sum()
        one_to_many_mask = df.kg_ids.apply(lambda x: len(x) > 1)
        many_to_one_mask = df.chosen_kg_id.notna() & df.chosen_kg_id.duplicated(keep=False)
        one_to_many_mappings = one_to_many_mask.sum()
        many_to_one_mappings = many_to_one_mask.sum()
        multi_mappings = (one_to_many_mask | many_to_one_mask).sum()
        one_to_one_mappings = mapped_to_kg - multi_mappings

        # Do some sanity checks
        assert multi_mappings <= mapped_to_kg
        assert one_to_many_mappings <= multi_mappings
        assert many_to_one_mappings <= multi_mappings
        assert multi_mappings + one_to_one_mappings == mapped_to_kg
        assert has_only_provided_ids + has_only_assigned_ids + has_both_provided_and_assigned_ids == has_valid_ids
        assert assigned_mappings_correct_per_provided <= mapped_to_kg_provided

        # Compile final stats summary
        stats = {
            'mapped_dataset': results_tsv_path,
            'total_items': total_items,
            'mapped_to_kg': int(mapped_to_kg),

            'mapped_to_kg_provided': int(mapped_to_kg_provided),
            'mapped_to_kg_assigned': int(mapped_to_kg_assigned),
            'mapped_to_kg_provided_and_assigned': int(mapped_to_kg_provided_and_assigned),

            'one_to_one_mappings': int(one_to_one_mappings),
            'multi_mappings': int(multi_mappings),
            'one_to_many_mappings': int(one_to_many_mappings),
            'many_to_one_mappings': int(many_to_one_mappings),

            'has_valid_ids': int(has_valid_ids),
            'has_valid_ids_provided': int(has_valid_ids_provided),
            'has_valid_ids_assigned': int(has_valid_ids_assigned),
            'has_only_provided_ids': int(has_only_provided_ids),
            'has_only_assigned_ids': int(has_only_assigned_ids),
            'has_both_provided_and_assigned_ids': int(has_both_provided_and_assigned_ids),

            'assigned_mappings_correct_per_provided': int(assigned_mappings_correct_per_provided),
            'assigned_mappings_correct_per_provided_chosen': int(assigned_mappings_correct_per_provided_chosen),

            'has_invalid_ids': int(has_invalid_ids),
            'has_invalid_ids_provided': int(has_invalid_ids_provided),
            'has_invalid_ids_assigned': int(has_invalid_ids_assigned),
            'has_no_ids': int(has_no_ids),
            'has_invalid_ids_and_not_mapped_to_kg': int(has_invalid_ids_and_not_mapped_to_kg),
        }

        # Calculate performance stats for 'assigned' ids vs. provided
        precision_per_provided = safe_divide(assigned_mappings_correct_per_provided, mapped_to_kg_provided_and_assigned)
        recall_per_provided = safe_divide(assigned_mappings_correct_per_provided, mapped_to_kg_provided)
        precision_per_provided_chosen = safe_divide(assigned_mappings_correct_per_provided_chosen, mapped_to_kg_provided_and_assigned)
        recall_per_provided_chosen = safe_divide(assigned_mappings_correct_per_provided_chosen, mapped_to_kg_provided)

        # Compile performance stats
        performance: Dict[str, Any] = {
            'overall': {
                'coverage': safe_divide(mapped_to_kg, total_items),
                'coverage_explanation': f"{mapped_to_kg} / {total_items}"
            },
            'assigned_ids': {
                'coverage': safe_divide(mapped_to_kg_assigned, total_items),
                'coverage_explanation': f"{mapped_to_kg_assigned} / {total_items}",
                'per_provided_ids': {
                    'precision': precision_per_provided,
                    'precision_explanation': f"{assigned_mappings_correct_per_provided} / {mapped_to_kg_provided_and_assigned}",
                    'recall': recall_per_provided,
                    'recall_explanation': f"{assigned_mappings_correct_per_provided} / {mapped_to_kg_provided}",
                    'f1_score': calculate_f1_score(precision_per_provided, recall_per_provided),
                    'after_resolving_one_to_manys': {
                        'precision': precision_per_provided_chosen,
                        'precision_explanation': f"{assigned_mappings_correct_per_provided_chosen} / {mapped_to_kg_provided_and_assigned}",
                        'recall': recall_per_provided_chosen,
                        'recall_explanation': f"{assigned_mappings_correct_per_provided_chosen} / {mapped_to_kg_provided}",
                        'f1_score': calculate_f1_score(precision_per_provided_chosen, recall_per_provided_chosen)
                    }
                }
            }
        }

        # Do evaluation vs. groundtruth, if available
        if 'kg_ids_groundtruth' in df:
            assert df.kg_ids_groundtruth.notnull().all()  # TODO: adjust later so we don't have to enforce this.. (just use rows w/ groundtruth mappings available)
            canonical_map = get_kg_ids(list(set(df.kg_ids_groundtruth.explode().dropna())))
            df['kg_ids_groundtruth_canonical'] = df.apply(lambda r: [canonical_map[kg_id] for kg_id in r.kg_ids_groundtruth],
                                                          axis=1)

            # TODO: can we require more than set intersection to count this as 'correct'? requiring equivalence might be complicated..
            mappings_correct_per_groundtruth = df.apply(lambda r: len(set(r.kg_ids) & set(r.kg_ids_groundtruth_canonical)) > 0,
                                                        axis=1).sum()
            precision = safe_divide(mappings_correct_per_groundtruth, mapped_to_kg)
            recall = safe_divide(mappings_correct_per_groundtruth, total_items)

            # Tack these metrics onto our stats
            stats['mappings_correct_per_groundtruth'] = int(mappings_correct_per_groundtruth)
            performance['overall']['per_groundtruth'] = {
                'precision': precision,
                'precision_explanation': f"{mappings_correct_per_groundtruth} / {mapped_to_kg}",
                'recall': recall,
                'recall_explanation': f"{mappings_correct_per_groundtruth} / {total_items}",
                'f1_score': calculate_f1_score(precision, recall)
            }

        # Tack the performance metrics onto our other stats
        stats['performance'] = performance

        # Save all result stats
        logging.info(f"Dataset summary stats are: {json.dumps(stats, indent=2)}")
        results_filepath_root = results_tsv_path.replace('.tsv', '')
        with open(f"{results_filepath_root}_a_summary_stats.json", 'w+') as stats_file:
            json.dump(stats, stats_file, indent=2)

        # Record the items that had valid curies but that weren't in the KG, for easy reference
        kg_misses = df[(df.curies.apply(len) > 0) & (df.kg_ids.apply(len) == 0)]
        kg_misses.to_csv(f"{results_filepath_root}_b_curie_misses.tsv", sep='\t')

        # Record the items that didn't get mapped to the KG, for easy reference
        unmapped = df[df.kg_ids.apply(len) == 0]
        unmapped.to_csv(f"{results_filepath_root}_c_unmapped.tsv", sep='\t')

        # Record the items that DID map to the KG, for easy reference
        mapped = df[df.kg_ids.apply(len) > 0]
        mapped.to_csv(f"{results_filepath_root}_d_mapped.tsv", sep='\t')

        # Record the items with invalid IDs, for easy reference
        invalid_ids = df[df.invalid_ids.apply(lambda x: len(x) > 0)]
        invalid_ids.to_csv(f"{results_filepath_root}_e_invalid_ids.tsv", sep='\t')

        # Record the one-to-many items, for easy reference
        one_to_many_items = df[df.kg_ids.apply(lambda x: len(x) > 1)]
        one_to_many_items.to_csv(f"{results_filepath_root}_f_one_to_many.tsv", sep='\t')

        # Record the many-to-one items, for easy reference
        many_to_one_items = df[df.chosen_kg_id.notna() & df.chosen_kg_id.duplicated(keep=False)]
        many_to_one_items.to_csv(f"{results_filepath_root}_g_many_to_one.tsv", sep='\t')

        return stats

