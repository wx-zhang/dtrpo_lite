import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import datasets
import trl
from trl import ScriptArguments as DPOScriptArguments
import yaml

from datasets import concatenate_datasets, DatasetDict, interleave_datasets
from data.parsing_functions import *

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: Optional[str] = None
    columns: Optional[list[str]] = None
    weight: Optional[float] = None
    format: str = None
    parsing_fcn: str = None
    total_size: int = 300000


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(DPOScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name (optional when dataset_mixture is provided)."},
    )

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Configuration for creating dataset mixtures with advanced options like shuffling."
        },
    )

    dataset_mixture_yaml: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a YAML file containing the dataset mixture configuration. "
            "This is an alternative to specifying the `dataset_mixture` argument directly."
        },
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to use streaming datasets. This is useful for large datasets that cannot fit in memory."
        },
    )

    def __post_init__(self):
        if self.dataset_mixture_yaml is not None:
            yaml_path = self.dataset_mixture_yaml
            with open(yaml_path, "r") as f:
                self.dataset_mixture = yaml.safe_load(f)

        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError(
                "Either `dataset_name` or `dataset_mixture` must be provided"
            )

        if self.dataset_mixture is not None:
            if (
                not isinstance(self.dataset_mixture, dict)
                or "datasets" not in self.dataset_mixture
            ):
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", None),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                            format=dataset_config.get("format", None),
                            parsing_fcn=dataset_config.get("parsing_fcn", None),
                            total_size=dataset_config.get("total_size", None),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [
                set(dataset.columns)
                for dataset in datasets_list
                if dataset.columns is not None
            ]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")

        if args.dataset_name.endswith(".jsonl"):
            path = args.dataset_name
            ds = datasets.load_dataset(
                "json",
                data_files=path,
                streaming=args.dataset_streaming,
            )

        else:
            ds = datasets.load_dataset(args.dataset_name, args.dataset_config)

        return ds
    elif args.dataset_mixture:
        logger.info(
            f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets"
        )
        seed = args.dataset_mixture.seed
        datasets_list = []
        weighted_streaming = []

        for dataset_config in args.dataset_mixture.datasets:
            # ===============
            # Load dataset
            # ===============
            logger.info(
                f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config})"
            )

            path = dataset_config.id
            if dataset_config.id.endswith(".jsonl"):
                ds = datasets.load_dataset(
                    "json",
                    data_files=path,
                    split=dataset_config.split,
                    streaming=args.dataset_streaming,
                )

            else:
                try:
                    ds = datasets.load_dataset(
                        path,
                        dataset_config.config,
                        split=dataset_config.split,
                        streaming=args.dataset_streaming,
                    )
                except:
                    ds = datasets.load_from_disk(path)
                    if dataset_config.split is not None:
                        ds = ds[dataset_config.split]

            # subsample dataset for non-streaming datasets
            # for streaming datasets, we will use interleave_datasets to subsample
            if dataset_config.weight is not None:
                if not args.dataset_streaming:
                    ds = ds.shuffle(seed=seed).select(
                        range(int(len(ds) * dataset_config.weight))
                    )
                    logger.info(
                        f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                    )
                else:
                    weighted_streaming.append(dataset_config.weight)

            # parse dataset
            if dataset_config.format == "text_dict":

                def parse_text(example):
                    return json.loads(example["text"])

                ds = ds.map(lambda ex: {"dict": parse_text(ex)})
                ds = ds.map(lambda ex: ex["dict"], remove_columns=ds.column_names)
                logger.info(f"Load dataset '{dataset_config.id}' internal json")

            if dataset_config.parsing_fcn is not None:
                parsing_fcn = globals()[dataset_config.parsing_fcn]
                ds = ds.map(
                    parsing_fcn,
                    fn_kwargs={"dataset_id": dataset_config.id},
                    remove_columns=ds.column_names,
                )
                ds = ds.filter(lambda x: x["valid"] is True)

                logger.info(
                    f"Parse dataset '{dataset_config.id}' with parsing function {dataset_config.parsing_fcn}"
                )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)

            logger.info(f"Dataset Added: {ds}")
            datasets_list.append(ds)

        if datasets_list:
            # ===============
            # Combine datasets
            # ===============
            if args.dataset_streaming and len(weighted_streaming) > 0:
                assert len(weighted_streaming) == len(datasets_list)
                weighted_streaming = [
                    i / sum(weighted_streaming) for i in weighted_streaming
                ]
                combined_dataset = interleave_datasets(
                    datasets_list,
                    probabilities=weighted_streaming,
                    seed=seed,
                    stopping_strategy="all_exhausted",
                )
            else:
                combined_dataset = concatenate_datasets(datasets_list)

            # ===============
            # Shuffle and subsample
            # ===============
            if not args.dataset_streaming:
                combined_dataset = combined_dataset.shuffle(seed=seed)
                logger.info(
                    f"Created dataset mixture with {len(combined_dataset)} examples"
                )
            if (
                dataset_config.total_size is not None
                and len(combined_dataset) > dataset_config.total_size
                and not args.dataset_streaming
            ):
                combined_dataset = combined_dataset.select(
                    range(int(dataset_config.total_size))
                )

            # ===============
            # Split dataset
            # ===============
            if args.dataset_mixture.test_split_size is not None and not args.dataset_streaming:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
