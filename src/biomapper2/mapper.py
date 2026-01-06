"""
Main mapper module for entity and dataset knowledge graph mapping.

Provides the Mapper class for harmonizing biological entities to knowledge graphs
through annotation, normalization, linking, and resolution steps.
"""

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .biolink_client import BiolinkClient
from .config import PROJECT_ROOT
from .core.analysis import analyze_dataset_mapping
from .core.annotation_engine import AnnotationEngine
from .core.linker import Linker
from .core.normalizer import Normalizer
from .core.resolver import Resolver
from .utils import AnnotationMode, merge_into_entity, setup_logging

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
        self.biolink_client = BiolinkClient(biolink_version=biolink_version)
        self.annotation_engine = AnnotationEngine(biolink_client=self.biolink_client)
        self.normalizer = Normalizer(biolink_client=self.biolink_client)
        self.linker = Linker()
        self.resolver = Resolver()

    def map_entity_to_kg(
        self,
        item: pd.Series | dict[str, Any],
        name_field: str,
        provided_id_fields: list[str],
        entity_type: str,
        vocab: str | list[str] | None = None,
        array_delimiters: list[str] | None = None,
        stop_on_invalid_id: bool = False,
        annotation_mode: AnnotationMode = "missing",
        annotators: list[str] | None = None,
    ) -> pd.Series | dict[str, Any]:
        """
        Map a single entity to knowledge graph nodes.

        Args:
            item: Entity with name and ID fields
            name_field: Field containing entity name
            provided_id_fields: List of fields containing vocab identifiers
            entity_type: Type of entity (e.g., 'metabolite', 'protein')
            vocab: Allowed vocab name(s) to map to (e.g., 'refmet', 'mondo')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            stop_on_invalid_id: Halt execution on invalid IDs (default: False)
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            annotators: Optional list of annotators to use (by slug). If None, annotators are selected automatically.

        Returns:
            Mapped entity with added fields: curies, kg_ids, chosen_kg_id, etc.
        """
        logging.debug(f"Item at beginning of map_entity_to_kg() is {item}")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]
        mapped_item = copy.deepcopy(item)  # Use a copy to avoid editing input item

        # Validate/standardize the input entity type and vocab(s) on Biolink
        entity_type = self.biolink_client.standardize_entity_type(entity_type)
        prefixes = self.normalizer.get_standard_prefix(vocab)

        # Do Step 1: annotate with vocab IDs
        annotation_result = self.annotation_engine.annotate(
            item=mapped_item,
            name_field=name_field,
            provided_id_fields=provided_id_fields,
            category=entity_type,
            prefixes=prefixes,
            mode=annotation_mode,
            annotators=annotators,
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
        dataset: str | Path | pd.DataFrame,
        entity_type: str,
        name_column: str,
        provided_id_columns: list[str],
        vocab: str | list[str] | None = None,
        array_delimiters: list[str] | None = None,
        output_prefix: str | None = None,
        output_dir: str | Path = PROJECT_ROOT / "results",
        annotation_mode: AnnotationMode = "missing",
        annotators: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Map all entities in a dataset to knowledge graph nodes.

        Args:
            dataset: Path to TSV/CSV file or pandas DataFrame for processing
            entity_type: Type of entities (e.g., 'metabolite', 'protein')
            name_column: Column containing entity names
            provided_id_columns: Columns containing (un-normalized) vocab identifiers
            vocab: Allowed vocab name(s) to map to (e.g., 'CHEBI', 'MONDO')
            array_delimiters: Characters used to split delimited ID strings (default: [',', ';'])
            annotation_mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            output_prefix: Optional prefix for the output TSV file name
            output_dir: Optional path to directory to save output/result files in
            annotators: Optional list of annotators to use (by slug). If None, annotators are selected automatically.

        Returns:
            Tuple of (output_tsv_path, stats_summary)
        """
        logging.info(f"Beginning to map dataset to KG ({dataset})")
        array_delimiters = array_delimiters if array_delimiters is not None else [",", ";"]

        # Validate/standardize the input entity type and vocab(s) on Biolink
        entity_type = self.biolink_client.standardize_entity_type(entity_type)
        prefixes = self.normalizer.get_standard_prefix(vocab)

        # Ensure the results directory for output files exists
        output_dir = Path(output_dir)
        logging.info(f"Output dir path is: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: how to handle other data types, like .txt?
        # TODO: let file output location be configurable? #11
        # Issue: if dataset is a pandas df, need to create some default filename
        # naively create a default output filename (input_df_MAPPED) if output_prefix not provided

        output_suffix = "_MAPPED.tsv"
        if isinstance(dataset, pd.DataFrame):
            df = dataset
            output_tsv_name = f"input_df{output_suffix}" if output_prefix is None else f"{output_prefix}{output_suffix}"
        elif isinstance(dataset, (str, Path)):
            dataset = str(dataset)
            # Load tsv into pandas
            output_tsv_name = Path(dataset).name.replace(".tsv", output_suffix).replace(".csv", output_suffix)
            if dataset.endswith(".tsv"):
                df = pd.read_csv(dataset, sep="\t", dtype={id_col: str for id_col in provided_id_columns}, comment="#")
            elif dataset.endswith(".csv"):
                df = pd.read_csv(dataset, dtype={id_col: str for id_col in provided_id_columns}, comment="#")
            else:
                raise ValueError(f"Unsupported file extension for dataset: {dataset}")
        else:
            raise ValueError(
                f"Unsupported type of '{type(dataset)}' for 'dataset' parameter; "
                f"only str, Path, or pd.DataFrame are supported"
            )
        logging.info(f"Output tsv name is: {output_tsv_name}")
        output_tsv_path = output_dir / output_tsv_name
        logging.info(f"output tsv path is: {output_tsv_path}")

        # Do some basic cleanup to try to ensure empty cells are represented consistently
        df[provided_id_columns] = df[provided_id_columns].replace("-", np.nan)
        df[provided_id_columns] = df[provided_id_columns].replace("NO_MATCH", np.nan)
        num_rows_start = len(df)

        # Do Step 1: annotate all rows with vocab IDs
        annotation_df = self.annotation_engine.annotate(
            item=df,
            name_field=name_column,
            provided_id_fields=provided_id_columns,
            category=entity_type,
            prefixes=prefixes,
            mode=annotation_mode,
            annotators=annotators,
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

        stats_summary = analyze_dataset_mapping(output_tsv_path, self.linker, annotation_mode)

        return str(output_tsv_path), stats_summary
