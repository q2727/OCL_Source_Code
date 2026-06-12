from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithm import OCL
from .baselines import KModes
from .data import list_available_datasets, load_dataset

SUPPORTED_METHODS = ("kmd", "ocl")


@dataclass(slots=True)
class RunSummary:
    dataset: str
    method: str
    ca_mean: float
    ca_std: float
    ari_mean: float
    ari_std: float
    nmi_mean: float
    nmi_std: float


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _parse_methods(methods: str) -> list[str]:
    parsed = [method.strip().lower() for method in methods.split(",") if method.strip()]
    if not parsed:
        raise ValueError("At least one method must be provided.")
    if "all" in parsed:
        return list(SUPPORTED_METHODS)

    unsupported = sorted(set(parsed) - set(SUPPORTED_METHODS))
    if unsupported:
        supported = ", ".join([*SUPPORTED_METHODS, "all"])
        raise ValueError(f"Unsupported method(s): {', '.join(unsupported)}. Use: {supported}.")
    return parsed


def _build_model(method: str, seed: int, max_outer_loops: int, max_init_loops: int):
    if method == "ocl":
        return OCL(max_outer_loops=max_outer_loops, max_init_loops=max_init_loops, seed=seed)
    if method == "kmd":
        return KModes(max_loops=max_init_loops, seed=seed)
    raise ValueError(f"Unsupported method: {method}")


def _run_dataset(
    dataset_name: str,
    method: str,
    data_root: Path,
    runs: int,
    seed: int,
    max_outer_loops: int,
    max_init_loops: int,
) -> tuple[RunSummary, list[list[int]]]:
    dataset = load_dataset(dataset_name, data_root=data_root)

    ca_scores: list[float] = []
    ari_scores: list[float] = []
    nmi_scores: list[float] = []
    last_orders: list[list[int]] = []

    for run_idx in range(runs):
        model = _build_model(
            method=method,
            seed=seed + run_idx,
            max_outer_loops=max_outer_loops,
            max_init_loops=max_init_loops,
        )
        result = model.fit_predict(
            dataset.features,
            true_labels=dataset.labels,
            n_clusters=dataset.true_k,
        )
        ca_scores.append(float(result.ca))
        ari_scores.append(float(result.ari))
        nmi_scores.append(float(result.nmi))
        if method == "ocl":
            last_orders = result.learned_orders

    summary = RunSummary(
        dataset=dataset.name,
        method=method.upper(),
        ca_mean=float(np.mean(ca_scores)),
        ca_std=float(np.std(ca_scores)),
        ari_mean=float(np.mean(ari_scores)),
        ari_std=float(np.std(ari_scores)),
        nmi_mean=float(np.mean(nmi_scores)),
        nmi_std=float(np.std(nmi_scores)),
    )
    return summary, last_orders


def _print_summaries(summaries: list[RunSummary]) -> None:
    show_method = len({summary.method for summary in summaries}) > 1
    if show_method:
        print("Dataset | Method | CA | ARI | NMI\n--- | --- | --- | --- | ---")
    else:
        print("Dataset | CA | ARI | NMI\n--- | --- | --- | ---")

    for summary in summaries:
        row = [summary.dataset]
        if show_method:
            row.append(summary.method)
        row.extend(
            [
                _format_metric(summary.ca_mean, summary.ca_std),
                _format_metric(summary.ari_mean, summary.ari_std),
                _format_metric(summary.nmi_mean, summary.nmi_std),
            ]
        )
        print(" | ".join(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python port of OCL.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Single dataset abbreviation, e.g. NS or VT.")
    group.add_argument("--all", action="store_true", help="Run all datasets under --data-root.")
    parser.add_argument(
        "--data-root",
        default="OCL_Source_Code/Data",
        help="Directory containing the benchmark MAT files.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs.")
    parser.add_argument(
        "--methods",
        default="ocl",
        help="Comma-separated methods to run: ocl, kmd, or all. Default: ocl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=25,
        help="Base random seed. Repeated runs use seed + run_index.",
    )
    parser.add_argument(
        "--max-outer-loops",
        type=int,
        default=50,
        help="Maximum number of OCL outer iterations.",
    )
    parser.add_argument(
        "--max-init-loops",
        type=int,
        default=50,
        help="Maximum number of k-modes initialization/baseline iterations.",
    )
    parser.add_argument(
        "--show-orders",
        action="store_true",
        help="Print the learned order of each attribute from the last run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root)
    datasets = (
        list_available_datasets(data_root)
        if args.all
        else [str(args.dataset).upper()]
    )
    methods = _parse_methods(args.methods)

    summaries: list[RunSummary] = []
    for dataset_name in datasets:
        for method in methods:
            summary, orders = _run_dataset(
                dataset_name=dataset_name,
                method=method,
                data_root=data_root,
                runs=args.runs,
                seed=args.seed,
                max_outer_loops=args.max_outer_loops,
                max_init_loops=args.max_init_loops,
            )
            summaries.append(summary)
            if args.show_orders and method == "ocl":
                print(f"\nLearned orders for {dataset_name}:")
                for attr_idx, attr_order in enumerate(orders, start=1):
                    print(f"  attr_{attr_idx}: {attr_order}")

    print()
    _print_summaries(summaries)


if __name__ == "__main__":
    main()
