from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


DEFAULT_DATASET_KEYS: dict[str, str] = {
    "AC": "AC",
    "AP": "AP_c",
    "BC": "BC",
    "CS": "CS",
    "DS": "DS_c",
    "HR": "HR",
    "LG": "LG",
    "NS": "NS",
    "SB": "SB",
    "TT": "TT",
    "VT": "VT",
    "ZO": "ZO",
}


@dataclass(slots=True)
class Dataset:
    name: str
    data: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    true_k: int


def _candidate_keys(mat_dict: dict[str, np.ndarray]) -> list[str]:
    keys: list[str] = []
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            keys.append(key)
    return keys


def _choose_key(dataset_name: str, mat_dict: dict[str, np.ndarray], mat_key: str | None) -> str:
    if mat_key is not None:
        if mat_key not in mat_dict:
            raise KeyError(f"Requested MAT variable {mat_key!r} is not present.")
        return mat_key

    if dataset_name in DEFAULT_DATASET_KEYS and DEFAULT_DATASET_KEYS[dataset_name] in mat_dict:
        return DEFAULT_DATASET_KEYS[dataset_name]

    candidates = _candidate_keys(mat_dict)
    if not candidates:
        raise ValueError("No 2D array was found in the MAT file.")
    if len(candidates) == 1:
        return candidates[0]

    # Fallback: the benchmark matrix is usually the largest 2D numeric array.
    return max(candidates, key=lambda key: int(np.prod(mat_dict[key].shape)))


def list_available_datasets(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    return sorted(path.stem for path in root.glob("*.mat"))


def load_dataset(
    dataset_name: str,
    data_root: str | Path = "OCL_Source_Code/Data",
    mat_key: str | None = None,
) -> Dataset:
    name = dataset_name.upper()
    path = Path(data_root) / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    mat_dict = loadmat(path)
    chosen_key = _choose_key(name, mat_dict, mat_key)
    matrix = np.asarray(mat_dict[chosen_key])
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}.")

    matrix = np.asarray(matrix, dtype=np.int64)
    features = matrix[:, :-1]
    labels = matrix[:, -1].astype(np.int64)
    return Dataset(
        name=name,
        data=matrix,
        features=features,
        labels=labels,
        true_k=int(labels.max()),
    )


def encode_features(features: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Encode each categorical attribute into contiguous codes [0, m_i)."""

    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    encoded = np.empty_like(features, dtype=np.int64)
    original_values: list[np.ndarray] = []
    for col in range(features.shape[1]):
        values, inverse = np.unique(features[:, col], return_inverse=True)
        encoded[:, col] = inverse
        original_values.append(values)
    return encoded, original_values


def decode_orders(order_labels: Iterable[np.ndarray]) -> list[list[int]]:
    return [list(map(int, labels.tolist())) for labels in order_labels]
