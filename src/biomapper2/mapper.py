"""
Main mapper module for entity and dataset knowledge graph mapping.

Provides the Mapper class for harmonizing biological entities to knowledge graphs
through annotation, normalization, linking, and resolution steps.
"""

import ast
import copy
import json
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from .core.annotation_engine import AnnotationEngine
from .core.linker import Linker
from .core.normalizer import Normalizer
from .core.resolver import Resolver
from .utils import calculate_f1_score, merge_into_entity, safe_divide, setup_logging

setup_logging()


class Mapper:
    """
    Maps biological entities and datasets of entities to knowledge graph nodes.

    Performs four-step mapping pipeline:
    1. Annotation - assign additional vocab IDs via external APIs
    2. Normalization - convert un-normalized vocab IDs to Biolink-standard curies
    3. Linking - map curies to knowledge graph nodes
    4. Resolution - resolve one-to-many mappings
    """

    def __init__(self, biolink_version: str | None = None):
        # Instantiate the mapping modules (should only be done once, up front)
        self.annotation_engine = AnnotationEngine()
        self.normalizer = Normalizer(biolink_version=biolink_version)
        self.linker = Linker()
        self.resolver = Resolver()

    def map_entity_to_kg(
        self,
        item: pd.Series | dict[str, Any],
        name_field: str,
        provided_id_fields: list[str],
        entity_type: str,
        array_delimiters: list[str] | None = None,
        stop_on_invalid_id: bool = False,
        annotation_mode: Literal["all", "missing", "none"] = "missing",
    ) -> pd.Series | dict[str, Any]:
        """
        Map a single entity to knowledge graph nodes.

        Args:
            item: Entity with name and ID fields
            name_field: Field containing entity name
            provided_id_fields: List of fields containing vocab identifiers
            entity_type: Type of entity (e.g., 'metabolite', 'protein')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            stop_on_invalid_id: Halt execution on invalid IDs (default: False)
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)

        Returns:
            Mapped entity with added fields: curies, kg_ids, chosen_kg_id, etc.
        """
        logging.debug(f"Item at beginning of map_entity_to_kg() is {item}")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]
        mapped_item = copy.deepcopy(item)  # Use a copy to avoid editing input item

        # Do Step 1: annotate with vocab IDs
        annotation_result = self.annotation_engine.annotate(
            item=mapped_item,
            name_field=name_field,
            provided_id_fields=provided_id_fields,
            entity_type=entity_type,
            mode=annotation_mode,
        )
        assert isinstance(annotation_result, pd.Series)
        mapped_item = merge_into_entity(mapped_item, annotation_result)

        # Do Step 2: normalize vocab IDs to form proper curies
        normalization_result = self.normalizer.normalize(
            item=mapped_item,
            provided_id_fields=provided_id_fields,
            array_delimiters=array_delimiters,
            stop_on_invalid_id=stop_on_invalid_id,
        )
        assert isinstance(normalization_result, pd.Series)
        mapped_item = merge_into_entity(mapped_item, normalization_result)

        # Do Step 3: link curies to KG nodes
        linked_result = self.linker.link(mapped_item)
        assert isinstance(linked_result, pd.Series)
        mapped_item = merge_into_entity(mapped_item, linked_result)

        # Do Step 4: resolve one-to-many KG matches
        resolved_result = self.resolver.resolve(mapped_item)
        assert isinstance(resolved_result, pd.Series)
        mapped_item = merge_into_entity(mapped_item, resolved_result)

        return mapped_item

    def map_dataset_to_kg(
        self,
        dataset: str | pd.DataFrame,  # changed from dataset_tsv_path
        entity_type: str,
        name_column: str,
        provided_id_columns: list[str],
        array_delimiters: list[str] | None = None,
        annotation_mode: Literal["all", "missing", "none"] = "missing",
        output_prefix: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Map all entities in a dataset to knowledge graph nodes.

        Args:
            dataset: Path to TSV/CSV file or pandas DataFrame for processing
            entity_type: Type of entities (e.g., 'metabolite', 'protein')
            name_column: Column containing entity names
            provided_id_columns: Columns containing (un-normalized) vocab identifiers
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            output_prefix: Optional path to save the output TSV file

        Returns:
            Tuple of (output_tsv_path, stats_summary)
        """
        logging.info(f"Beginning to map dataset to KG ({dataset})")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]

        # TODO: Optionally allow people to input a Dataframe directly, as opposed to TSV path? #3
        # Addressed below:
        # Check if dataset is a pandas dataframe of a path to tsv/csv file
        # TODO: how to handle other data types, like .txt?

        # TODO: let file output location be configurable? #11
        # Issue: if dataset is a pandas df, need to create some default filename
        # naively create a default output filename (input_df_MAPPED) if output_prefix not provided

        if isinstance(dataset, pd.DataFrame):
            df = dataset
            output_tsv_path = "input_df_MAPPED.tsv" if output_prefix is None else f"{output_prefix}_MAPPED.tsv"
        elif isinstance(dataset, str):
            # Load tsv into pandas
            output_tsv_path = dataset.replace(".tsv", "_MAPPED.tsv").replace(".csv", "_MAPPED.tsv")
            if dataset.endswith(".tsv"):
                df = pd.read_csv(dataset, sep="\t", dtype={id_col: str for id_col in provided_id_columns})
            elif dataset.endswith(".csv"):
                df = pd.read_csv(dataset, dtype={id_col: str for id_col in provided_id_columns})
            else:
                raise ValueError(f"Unsupported file extension for dataset: {dataset}")

        # Do some basic cleanup to try to ensure empty cells are represented consistently
        df[provided_id_columns] = df[provided_id_columns].replace("-", np.nan)
        df[provided_id_columns] = df[provided_id_columns].replace("NO_MATCH", np.nan)
        num_rows_start = len(df)

        # Do Step 1: annotate all rows with vocab IDs
        annotation_df = self.annotation_engine.annotate(
            item=df,
            name_field=name_column,
            provided_id_fields=provided_id_columns,
            entity_type=entity_type,
            mode=annotation_mode,
        )
        df = df.join(annotation_df)
        logging.info(f"After step 1 (annotation), df is: \n{df}")

        # Do Step 2: normalize vocab IDs in all rows to form proper curies
        normalization_df = self.normalizer.normalize(
            item=df, provided_id_fields=provided_id_columns, array_delimiters=array_delimiters
        )
        df = df.join(normalization_df)
        logging.info(f"After step 2 (normalization), df is: \n{df}")

        # Do Step 3: link curies to KG nodes
        linked_df = self.linker.link(df)
        df = df.join(linked_df)
        logging.info(f"After step 3 (linking), df is: \n{df}")

        # Do Step 4: resolve one-to-many KG matches
        resolved_df = self.resolver.resolve(df)
        df = df.join(resolved_df)
        logging.info(f"After step 4 (resolution), df is: \n{df}")

        # Do a little validation of results dataframe
        num_rows_end = len(df)
        if num_rows_start != num_rows_end:
            raise ValueError(
                f"At end of map_dataset_to_kg(), dataframe has {num_rows_end} rows but started with {num_rows_start} "
                f"rows. Row count should not change."
            )

        # Dump the final dataframe to a TSV

        logging.info(f"Dumping output TSV to {output_tsv_path}")
        df.to_csv(output_tsv_path, sep="\t", index=False)

        stats_summary = self.analyze_dataset_mapping(output_tsv_path)

        return output_tsv_path, stats_summary

    def analyze_dataset_mapping(self, results_tsv_path: str) -> dict[str, Any]:
        """
        Analyze dataset mapping results and generate summary statistics.

        Args:
            results_tsv_path: Path to mapped dataset TSV

        Returns:
            Dictionary containing coverage, precision, recall, and F1 metrics
        """
        logging.info(f"Analyzing dataset KG mapping in {results_tsv_path}")

        cols_to_literal_eval = [
            "curies",
            "curies_provided",
            "curies_assigned",
            "invalid_ids",
            "invalid_ids_provided",
            "invalid_ids_assigned",
            "kg_ids",
            "kg_ids_provided",
            "kg_ids_assigned",
        ]
        converters = {col: ast.literal_eval for col in cols_to_literal_eval}
        df = pd.read_table(results_tsv_path, converters=converters)

        # Make sure we load any groundtruth column properly
        if "kg_ids_groundtruth" in df.columns:
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
        mapped_to_kg_both = df.apply(
            lambda r: (len(r.kg_ids_provided) > 0) & (len(r.kg_ids_assigned) > 0), axis=1
        ).sum()
        assigned_correct_per_provided = df.apply(
            lambda r: len(set(r.kg_ids_provided) & set(r.kg_ids_assigned)) > 0, axis=1
        ).sum()
        assigned_correct_per_provided_chosen = (
            (df.chosen_kg_id_provided == df.chosen_kg_id_assigned)
            & df.chosen_kg_id_provided.notna()
            & df.chosen_kg_id_assigned.notna()
        ).sum()
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
        assert assigned_correct_per_provided <= mapped_to_kg_provided

        # Compile final stats summary
        stats = {
            "mapped_dataset": results_tsv_path,
            "total_items": total_items,
            "mapped_to_kg": int(mapped_to_kg),
            "mapped_to_kg_provided": int(mapped_to_kg_provided),
            "mapped_to_kg_assigned": int(mapped_to_kg_assigned),
            "mapped_to_kg_provided_and_assigned": int(mapped_to_kg_both),
            "one_to_one_mappings": int(one_to_one_mappings),
            "multi_mappings": int(multi_mappings),
            "one_to_many_mappings": int(one_to_many_mappings),
            "many_to_one_mappings": int(many_to_one_mappings),
            "has_valid_ids": int(has_valid_ids),
            "has_valid_ids_provided": int(has_valid_ids_provided),
            "has_valid_ids_assigned": int(has_valid_ids_assigned),
            "has_only_provided_ids": int(has_only_provided_ids),
            "has_only_assigned_ids": int(has_only_assigned_ids),
            "has_both_provided_and_assigned_ids": int(has_both_provided_and_assigned_ids),
            "assigned_mappings_correct_per_provided": int(assigned_correct_per_provided),
            "assigned_mappings_correct_per_provided_chosen": int(assigned_correct_per_provided_chosen),
            "has_invalid_ids": int(has_invalid_ids),
            "has_invalid_ids_provided": int(has_invalid_ids_provided),
            "has_invalid_ids_assigned": int(has_invalid_ids_assigned),
            "has_no_ids": int(has_no_ids),
            "has_invalid_ids_and_not_mapped_to_kg": int(has_invalid_ids_and_not_mapped_to_kg),
        }

        # Calculate performance stats for 'assigned' ids vs. provided
        precision_per_provided = safe_divide(assigned_correct_per_provided, mapped_to_kg_both)
        recall_per_provided = safe_divide(assigned_correct_per_provided, mapped_to_kg_provided)
        precision_per_provided_chosen = safe_divide(assigned_correct_per_provided_chosen, mapped_to_kg_both)
        recall_per_provided_chosen = safe_divide(assigned_correct_per_provided_chosen, mapped_to_kg_provided)

        # Compile performance stats
        performance: dict[str, Any] = {
            "overall": {
                "coverage": safe_divide(mapped_to_kg, total_items),
                "coverage_explanation": f"{mapped_to_kg} / {total_items}",
            },
            "assigned_ids": {
                "coverage": safe_divide(mapped_to_kg_assigned, total_items),
                "coverage_explanation": f"{mapped_to_kg_assigned} / {total_items}",
                "per_provided_ids": {
                    "precision": precision_per_provided,
                    "precision_explanation": f"{assigned_correct_per_provided} / {mapped_to_kg_both}",
                    "recall": recall_per_provided,
                    "recall_explanation": f"{assigned_correct_per_provided} / {mapped_to_kg_provided}",
                    "f1_score": calculate_f1_score(precision_per_provided, recall_per_provided),
                    "after_resolving_one_to_manys": {
                        "precision": precision_per_provided_chosen,
                        "precision_explanation": f"{assigned_correct_per_provided_chosen} / {mapped_to_kg_both}",
                        "recall": recall_per_provided_chosen,
                        "recall_explanation": f"{assigned_correct_per_provided_chosen} / {mapped_to_kg_provided}",
                        "f1_score": calculate_f1_score(precision_per_provided_chosen, recall_per_provided_chosen),
                    },
                },
            },
        }

        # Do evaluation vs. groundtruth, if available
        if "kg_ids_groundtruth" in df:
            # TODO: adjust later so we don't have to enforce this.. (just use rows w/ groundtruth mappings available)
            assert df.kg_ids_groundtruth.notnull().all()
            canonical_map = self.linker.get_kg_ids(list(set(df.kg_ids_groundtruth.explode().dropna())))
            df["kg_ids_groundtruth_canonical"] = df.apply(
                lambda r: [canonical_map[kg_id] for kg_id in r.kg_ids_groundtruth], axis=1
            )

            # TODO: can we require more than set intersection to count this as 'correct'? might be complicated..
            mappings_correct_per_groundtruth = df.apply(
                lambda r: len(set(r.kg_ids) & set(r.kg_ids_groundtruth_canonical)) > 0, axis=1
            ).sum()
            precision = safe_divide(mappings_correct_per_groundtruth, mapped_to_kg)
            recall = safe_divide(mappings_correct_per_groundtruth, total_items)

            # Tack these metrics onto our stats
            stats["mappings_correct_per_groundtruth"] = int(mappings_correct_per_groundtruth)
            performance["overall"]["per_groundtruth"] = {
                "precision": precision,
                "precision_explanation": f"{mappings_correct_per_groundtruth} / {mapped_to_kg}",
                "recall": recall,
                "recall_explanation": f"{mappings_correct_per_groundtruth} / {total_items}",
                "f1_score": calculate_f1_score(precision, recall),
            }

        # Tack the performance metrics onto our other stats
        stats["performance"] = performance

        # Save all result stats
        logging.info(f"Dataset summary stats are: {json.dumps(stats, indent=2)}")
        results_filepath_root = results_tsv_path.replace(".tsv", "")
        with open(f"{results_filepath_root}_a_summary_stats.json", "w+") as stats_file:
            json.dump(stats, stats_file, indent=2)

        # Record the items that had valid curies but that weren't in the KG, for easy reference
        kg_misses = df[(df.curies.apply(len) > 0) & (df.kg_ids.apply(len) == 0)]
        kg_misses.to_csv(f"{results_filepath_root}_b_curie_misses.tsv", sep="\t")

        # Record the items that didn't get mapped to the KG, for easy reference
        unmapped = df[df.kg_ids.apply(len) == 0]
        unmapped.to_csv(f"{results_filepath_root}_c_unmapped.tsv", sep="\t")

        # Record the items that DID map to the KG, for easy reference
        mapped = df[df.kg_ids.apply(len) > 0]
        mapped.to_csv(f"{results_filepath_root}_d_mapped.tsv", sep="\t")

        # Record the items with invalid IDs, for easy reference
        invalid_ids = df[df.invalid_ids.apply(lambda x: len(x) > 0)]
        invalid_ids.to_csv(f"{results_filepath_root}_e_invalid_ids.tsv", sep="\t")

        # Record the one-to-many items, for easy reference
        one_to_many_items = df[df.kg_ids.apply(lambda x: len(x) > 1)]
        one_to_many_items.to_csv(f"{results_filepath_root}_f_one_to_many.tsv", sep="\t")

        # Record the many-to-one items, for easy reference
        many_to_one_items = df[df.chosen_kg_id.notna() & df.chosen_kg_id.duplicated(keep=False)]
        many_to_one_items.to_csv(f"{results_filepath_root}_g_many_to_one.tsv", sep="\t")

        return stats
