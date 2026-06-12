# OCL Python

This repository contains the Python implementation of OCL for categorical data clustering and the currently supported reproduction experiments for `Categorical Data Clustering via Value Order Estimated Distance Metric Learning`.

## Contents

- `src/ocl_python/`: Python package for data loading, metrics, OCL, and the supported `KMD` baseline.
- `OCL_Source_Code/Data/`: 12 benchmark `.mat` datasets used by the current main experiment run.
- `OCL_Source_Code/Supp_data/`: 6 supplementary `.mat` datasets available in this repository.
- `docs/current_experiment_report_zh.md`: latest Chinese experiment report, including runnable commands, KMD/OCL results, trend analysis, and mismatch analysis against arXiv v5.

## Setup

```bash
uv sync
```

## Run Experiments

Run OCL on one dataset:

```bash
uv run ocl-run --dataset NS --runs 10 --seed 25
```

Run OCL on all 12 main datasets:

```bash
uv run ocl-run --all --runs 10 --seed 25
```

Run the currently supported baseline and OCL:

```bash
uv run ocl-run --all --runs 10 --seed 25 --methods all
```

Run the supplementary datasets:

```bash
uv run ocl-run --all --data-root OCL_Source_Code/Supp_data --runs 10 --seed 25 --methods all
```

Show learned orders from the last OCL run:

```bash
uv run ocl-run --dataset VT --runs 1 --seed 25 --show-orders
```

## Current Scope

The runnable methods are:

- `OCL`: Python port of the value-order learning clustering algorithm.
- `KMD`: traditional k-modes baseline with Hamming distance.

The implemented metrics are `CA`, `ARI`, and `NMI`. The latest experiment report explains why exact numeric equality with the paper is not expected and which datasets currently deviate most from the paper trend.
