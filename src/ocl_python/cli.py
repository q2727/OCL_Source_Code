from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithm import OCL
from .baselines import KModes
from .data import list_available_datasets, load_dataset

SUPPORTED_METHODS = ("kmd", "ocl", "ocl1", "ocl2", "ocl3", "lnro", "rnro", "wocl")

# Nominal attribute counts per dataset (from paper Table 2).
# First s_n columns are assumed nominal, the rest ordinal.
DATASET_NOMINAL_COUNTS: dict[str, int] = {
    "NS": 1, "AP": 6, "CS": 1, "HR": 2, "BC": 5,
    "LG": 15, "CR": 7, "OB": 6, "BM": 6,
}


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
    cmp_mean: float
    cmp_std: float


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


def _build_model(
    method: str,
    seed: int,
    max_outer_loops: int,
    max_init_loops: int,
    *,
    weight_alpha: float,
    weight_gamma: float,
    weight_min: float | None,
    weight_delay: int,
    weight_mix: float,
    weight_guard: str,
    weight_entropy_min: float,
    weight_objective_tol: float,
):
    if method == "ocl":
        return OCL(max_outer_loops=max_outer_loops, max_init_loops=max_init_loops, seed=seed)
    if method in ("ocl1", "ocl2", "ocl3"):
        return OCL(max_outer_loops=max_outer_loops, max_init_loops=max_init_loops, seed=seed, variant=method)
    if method in ("lnro", "rnro"):
        return OCL(max_outer_loops=max_outer_loops, max_init_loops=max_init_loops, seed=seed, variant=method)
    if method == "wocl":
        return OCL(
            max_outer_loops=max_outer_loops,
            max_init_loops=max_init_loops,
            seed=seed,
            variant="wocl",
            weight_alpha=weight_alpha,
            weight_gamma=weight_gamma,
            weight_min=weight_min,
            weight_delay=weight_delay,
            weight_mix=weight_mix,
            weight_guard=weight_guard,
            weight_entropy_min=weight_entropy_min,
            weight_objective_tol=weight_objective_tol,
        )
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
    weight_alpha: float,
    weight_gamma: float,
    weight_min: float | None,
    weight_delay: int,
    weight_mix: float,
    weight_guard: str,
    weight_entropy_min: float,
    weight_objective_tol: float,
) -> tuple[RunSummary, list[list[int]], list[float] | None]:
    dataset = load_dataset(dataset_name, data_root=data_root)

    ca_scores: list[float] = []
    ari_scores: list[float] = []
    nmi_scores: list[float] = []
    cmp_scores: list[float] = []
    last_orders: list[list[int]] = []
    last_weights: list[float] | None = None

    for run_idx in range(runs):
        model = _build_model(
            method=method,
            seed=seed + run_idx,
            max_outer_loops=max_outer_loops,
            max_init_loops=max_init_loops,
            weight_alpha=weight_alpha,
            weight_gamma=weight_gamma,
            weight_min=weight_min,
            weight_delay=weight_delay,
            weight_mix=weight_mix,
            weight_guard=weight_guard,
            weight_entropy_min=weight_entropy_min,
            weight_objective_tol=weight_objective_tol,
        )
        if method in ("lnro", "rnro") and dataset_name in DATASET_NOMINAL_COUNTS:
            s_n = DATASET_NOMINAL_COUNTS[dataset_name]
            model.nominal_attrs = list(range(s_n))
        result = model.fit_predict(
            dataset.features,
            true_labels=dataset.labels,
            n_clusters=dataset.true_k,
        )
        ca_scores.append(float(result.ca))
        ari_scores.append(float(result.ari))
        nmi_scores.append(float(result.nmi))
        cmp_scores.append(float(result.cmp))
        if method in ("ocl", "ocl1", "ocl2", "lnro", "wocl"):
            last_orders = result.learned_orders
        if method == "wocl":
            last_weights = result.attribute_weights

    summary = RunSummary(
        dataset=dataset.name,
        method=method.upper(),
        ca_mean=float(np.mean(ca_scores)),
        ca_std=float(np.std(ca_scores)),
        ari_mean=float(np.mean(ari_scores)),
        ari_std=float(np.std(ari_scores)),
        nmi_mean=float(np.mean(nmi_scores)),
        nmi_std=float(np.std(nmi_scores)),
        cmp_mean=float(np.mean(cmp_scores)),
        cmp_std=float(np.std(cmp_scores)),
    )
    return summary, last_orders, last_weights


def _parse_weight_min(value: str) -> float | None:
    if value.lower() == "auto":
        return None
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("--weight-min must be non-negative or 'auto'.")
    return parsed


def _print_summaries(summaries: list[RunSummary]) -> None:
    show_method = len({summary.method for summary in summaries}) > 1
    if show_method:
        print("Dataset | Method | CA | ARI | NMI | CMP\n--- | --- | --- | --- | --- | ---")
    else:
        print("Dataset | CA | ARI | NMI | CMP\n--- | --- | --- | --- | ---")

    for summary in summaries:
        row = [summary.dataset]
        if show_method:
            row.append(summary.method)
        row.extend(
            [
                _format_metric(summary.ca_mean, summary.ca_std),
                _format_metric(summary.ari_mean, summary.ari_std),
                _format_metric(summary.nmi_mean, summary.nmi_std),
                _format_metric(summary.cmp_mean, summary.cmp_std),
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
        help="Comma-separated methods to run: ocl, wocl, kmd, or all. Default: ocl.",
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
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=0.5,
        help="WOCL smoothing factor for attribute-weight updates. Default: 0.5.",
    )
    parser.add_argument(
        "--weight-gamma",
        type=float,
        default=1.0,
        help="WOCL sharpness exponent for entropy-gain scores. Default: 1.0.",
    )
    parser.add_argument(
        "--weight-min",
        type=_parse_weight_min,
        default=None,
        help="WOCL minimum attribute weight, or 'auto' for 0.01 / n_attrs. Default: auto.",
    )
    parser.add_argument(
        "--weight-delay",
        type=int,
        default=0,
        help="WOCL outer iterations to wait before updating attribute weights. Default: 0.",
    )
    parser.add_argument(
        "--weight-mix",
        type=float,
        default=1.0,
        help="WOCL distance mix: 0 uses uniform OCL distance, 1 uses learned weights. Default: 1.0.",
    )
    parser.add_argument(
        "--weight-guard",
        choices=("none", "entropy", "objective", "objective_entropy"),
        default="none",
        help="WOCL guard for accepting weight updates. Default: none.",
    )
    parser.add_argument(
        "--weight-entropy-min",
        type=float,
        default=0.7,
        help="Minimum normalized weight entropy when --weight-guard entropy is used. Default: 0.7.",
    )
    parser.add_argument(
        "--weight-objective-tol",
        type=float,
        default=0.0,
        help="Relative tolerance for accepting objective guard updates. Default: 0.0.",
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
            summary, orders, weights = _run_dataset(
                dataset_name=dataset_name,
                method=method,
                data_root=data_root,
                runs=args.runs,
                seed=args.seed,
                max_outer_loops=args.max_outer_loops,
                max_init_loops=args.max_init_loops,
                weight_alpha=args.weight_alpha,
                weight_gamma=args.weight_gamma,
                weight_min=args.weight_min,
                weight_delay=args.weight_delay,
                weight_mix=args.weight_mix,
                weight_guard=args.weight_guard,
                weight_entropy_min=args.weight_entropy_min,
                weight_objective_tol=args.weight_objective_tol,
            )
            summaries.append(summary)
            if args.show_orders and method in ("ocl", "wocl"):
                print(f"\nLearned orders for {dataset_name} ({method.upper()}):")
                for attr_idx, attr_order in enumerate(orders, start=1):
                    print(f"  attr_{attr_idx}: {attr_order}")
                if weights is not None:
                    formatted = ", ".join(f"{weight:.4f}" for weight in weights)
                    print(f"  attribute_weights: [{formatted}]")

    print()
    _print_summaries(summaries)


if __name__ == "__main__":
    main()
