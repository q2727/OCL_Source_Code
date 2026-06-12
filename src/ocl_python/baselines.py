from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import encode_features
from .metrics import adjusted_rand_index, clustering_accuracy, normalized_mutual_information


@dataclass(slots=True)
class KModesResult:
    assignments: np.ndarray
    ca: float | None = None
    ari: float | None = None
    nmi: float | None = None


def _choose_unique_modes(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n_samples = X.shape[0]
    unique_rows = np.unique(X, axis=0)
    if unique_rows.shape[0] < k:
        raise ValueError(
            f"k={k} requires at least {k} unique rows, got {unique_rows.shape[0]}."
        )

    modes: list[np.ndarray] = []
    signatures: set[tuple[int, ...]] = set()
    while len(modes) < k:
        row = X[int(rng.randint(0, n_samples))]
        signature = tuple(int(v) for v in row.tolist())
        if signature in signatures:
            continue
        signatures.add(signature)
        modes.append(row.copy())
    return np.vstack(modes)


class KModes:
    """Traditional k-modes baseline with Hamming distance."""

    def __init__(self, max_loops: int = 50, seed: int | None = None) -> None:
        self.max_loops = max_loops
        self.seed = seed

    def fit_predict(
        self,
        features: np.ndarray,
        true_labels: np.ndarray | None = None,
        n_clusters: int | None = None,
    ) -> KModesResult:
        X, original_values = encode_features(np.asarray(features))
        n_samples, n_attrs = X.shape
        if n_clusters is None:
            if true_labels is None:
                raise ValueError("n_clusters must be provided when true_labels is absent.")
            n_clusters = int(np.unique(true_labels).size)
        k = int(n_clusters)

        num_values = np.array([len(values) for values in original_values], dtype=np.int64)
        rng = np.random.RandomState(self.seed)
        modes = _choose_unique_modes(X, k, rng)
        assignments = np.full(n_samples, -1, dtype=np.int64)

        for _ in range(self.max_loops + 1):
            distances = np.zeros((n_samples, k), dtype=np.float64)
            for attr in range(n_attrs):
                distances += (X[:, attr][:, None] != modes[:, attr][None, :]).astype(
                    np.float64
                )
            distances /= float(n_attrs)

            winners = distances.argmin(axis=1)
            if np.array_equal(winners, assignments):
                break
            assignments = winners

            for cluster in range(k):
                mask = assignments == cluster
                if not np.any(mask):
                    continue
                x_sub = X[mask]
                for attr in range(n_attrs):
                    counts = np.bincount(x_sub[:, attr], minlength=int(num_values[attr]))
                    modes[cluster, attr] = int(np.argmax(counts))

        result = KModesResult(assignments=assignments + 1)
        if true_labels is not None:
            result.ca = clustering_accuracy(result.assignments, true_labels)
            result.ari = adjusted_rand_index(result.assignments, true_labels)
            result.nmi = normalized_mutual_information(true_labels, result.assignments)
        return result
