from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import decode_orders, encode_features
from .metrics import adjusted_rand_index, clustering_accuracy, clustering_compactness, normalized_mutual_information


@dataclass(slots=True)
class OCLResult:
    assignments: np.ndarray
    learned_orders: list[list[int]]
    objective_history: list[float]
    order_update_iterations: list[int]
    ca: float | None = None
    ari: float | None = None
    nmi: float | None = None
    cmp: float | None = None


def _distance_matrix(num_values: int) -> np.ndarray:
    if num_values <= 1:
        return np.zeros((num_values, num_values), dtype=np.float64)
    grid = np.arange(num_values, dtype=np.float64)
    return np.abs(grid[:, None] - grid[None, :]) / float(num_values - 1)


def _choose_unique_modes(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n_samples = X.shape[0]
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


def _kmodes_initialization(
    X: np.ndarray,
    k: int,
    num_values: np.ndarray,
    rng: np.random.RandomState,
    max_loops: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_attrs = X.shape
    modes = _choose_unique_modes(X, k, rng)
    partition_old = np.full(n_samples, -1, dtype=np.int64)

    for _ in range(max_loops + 1):
        distances = np.zeros((n_samples, k), dtype=np.float64)
        for attr in range(n_attrs):
            distances += (X[:, attr][:, None] != modes[:, attr][None, :]).astype(np.float64)
        distances /= float(n_attrs)
        partition_new = distances.argmin(axis=1)
        if np.array_equal(partition_new, partition_old):
            break
        partition_old = partition_new
        for cluster in range(k):
            mask = partition_new == cluster
            if not np.any(mask):
                continue
            x_sub = X[mask]
            for attr in range(n_attrs):
                counts = np.bincount(x_sub[:, attr], minlength=int(num_values[attr]))
                modes[cluster, attr] = int(np.argmax(counts))
    return modes, partition_old


def _mode_probabilities(
    X: np.ndarray,
    assignments: np.ndarray,
    k: int,
    num_values: np.ndarray,
    previous: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    n_attrs = X.shape[1]
    if previous is None:
        probs = [np.zeros((k, int(num_values[attr])), dtype=np.float64) for attr in range(n_attrs)]
    else:
        probs = [attr_probs.copy() for attr_probs in previous]

    for cluster in range(k):
        mask = assignments == cluster
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue
        cluster_data = X[mask]
        for attr in range(n_attrs):
            counts = np.bincount(cluster_data[:, attr], minlength=int(num_values[attr]))
            probs[attr][cluster] = counts / float(cluster_size)
    return probs


def _mode_point_probabilities(modes: np.ndarray, num_values: np.ndarray) -> list[np.ndarray]:
    k, n_attrs = modes.shape
    probs = [np.zeros((k, int(num_values[attr])), dtype=np.float64) for attr in range(n_attrs)]
    for cluster in range(k):
        for attr in range(n_attrs):
            probs[attr][cluster, int(modes[cluster, attr])] = 1.0
    return probs


def _objective_from_distances(distances: np.ndarray, assignments: np.ndarray) -> float:
    return float(distances[np.arange(distances.shape[0]), assignments].sum())


def _sample_cluster_distances(
    X: np.ndarray,
    mode_probs: list[np.ndarray],
    distance_matrices: list[np.ndarray],
) -> np.ndarray:
    """Expected distance to each cluster, weighted by the value-probability
    distribution (full OCL — probability-aware distance)."""
    n_samples, n_attrs = X.shape
    k = mode_probs[0].shape[0]
    distances = np.zeros((n_samples, k), dtype=np.float64)
    for attr in range(n_attrs):
        attr_dists = distance_matrices[attr][X[:, attr]]
        distances += attr_dists @ mode_probs[attr].T
    return distances / float(n_attrs)


def _hard_mode_distances(
    X: np.ndarray,
    mode_probs: list[np.ndarray],
    distance_matrices: list[np.ndarray],
) -> np.ndarray:
    """Distance to each cluster's *mode* (OCL-I: equidistant order distance
    without probability weighting)."""
    n_samples, n_attrs = X.shape
    k = mode_probs[0].shape[0]
    distances = np.zeros((n_samples, k), dtype=np.float64)
    for attr in range(n_attrs):
        # Most frequent value per cluster
        modes_attr = np.argmax(mode_probs[attr], axis=1)
        for cluster in range(k):
            distances[:, cluster] += distance_matrices[attr][
                X[:, attr], modes_attr[cluster]
            ]
    return distances / float(n_attrs)


def _mixed_nom_ord_distances(
    X: np.ndarray,
    mode_probs: list[np.ndarray],
    distance_matrices: list[np.ndarray],
    nominal_attrs: list[int],
) -> np.ndarray:
    """Hamming for nominal attributes, order distance for ordinal (RNRO)."""
    n_samples, n_attrs = X.shape
    k = mode_probs[0].shape[0]
    distances = np.zeros((n_samples, k), dtype=np.float64)
    nominal_set = set(nominal_attrs)
    for attr in range(n_attrs):
        if attr in nominal_set:
            modes_attr = np.argmax(mode_probs[attr], axis=1)
            for cluster in range(k):
                distances[:, cluster] += (
                    X[:, attr] != modes_attr[cluster]
                ).astype(np.float64)
        else:
            attr_dists = distance_matrices[attr][X[:, attr]]
            distances += attr_dists @ mode_probs[attr].T
    return distances / float(n_attrs)


def _hamming_distances(
    X: np.ndarray,
    mode_probs: list[np.ndarray],
) -> np.ndarray:
    """Traditional Hamming distance to cluster modes (OCL-III: no order info)."""
    n_samples, n_attrs = X.shape
    k = mode_probs[0].shape[0]
    distances = np.zeros((n_samples, k), dtype=np.float64)
    for attr in range(n_attrs):
        modes_attr = np.argmax(mode_probs[attr], axis=1)
        for cluster in range(k):
            distances[:, cluster] += (
                X[:, attr] != modes_attr[cluster]
            ).astype(np.float64)
    return distances / float(n_attrs)


def _interleaved_rank_order(significance: np.ndarray) -> np.ndarray:
    sorted_nodes = np.argsort(-significance, kind="mergesort")
    new_order = sorted_nodes.copy()
    num_values = len(sorted_nodes)
    middle = int(np.ceil(num_values / 2.0) - 1)
    new_order[middle] = sorted_nodes[0]
    for step in range(1, int((num_values - 1) / 2) + 1):
        new_order[middle - step] = sorted_nodes[step * 2 - 1]
        new_order[middle + step] = sorted_nodes[step * 2]

    rank_order = np.zeros(num_values, dtype=np.float64)
    for rank, value in enumerate(new_order):
        rank_order[value] = rank
    return rank_order


def _choose_orders(
    assignments: np.ndarray,
    mode_probs: list[np.ndarray],
    distance_matrices: list[np.ndarray],
) -> list[np.ndarray]:
    n_samples = assignments.size
    k = mode_probs[0].shape[0]
    cluster_weights = np.bincount(assignments, minlength=k).astype(np.float64) / float(n_samples)
    final_orders: list[np.ndarray] = []

    for attr, attr_probs in enumerate(mode_probs):
        num_values = attr_probs.shape[1]
        cluster_rankings = np.zeros((k, num_values), dtype=np.float64)
        for cluster in range(k):
            significance = np.zeros(num_values, dtype=np.float64)
            for value in range(num_values):
                cost = float(distance_matrices[attr][value] @ attr_probs[cluster])
                if cost == 0.0:
                    cost = 1e-6
                significance[value] = attr_probs[cluster, value] / cost
            cluster_rankings[cluster] = _interleaved_rank_order(significance)

        combined = cluster_weights @ cluster_rankings
        final_orders.append(np.argsort(combined, kind="mergesort"))
    return final_orders


def _choose_orders_nominal(
    assignments: np.ndarray,
    mode_probs: list[np.ndarray],
    distance_matrices: list[np.ndarray],
    nominal_attrs: list[int],
) -> list[np.ndarray]:
    """Like _choose_orders but only reorders nominal attributes (LNRO)."""
    n_samples = assignments.size
    k = mode_probs[0].shape[0]
    cluster_weights = np.bincount(assignments, minlength=k).astype(np.float64) / float(n_samples)
    n_attrs = len(mode_probs)
    final_orders: list[np.ndarray] = []

    for attr, attr_probs in enumerate(mode_probs):
        num_values = attr_probs.shape[1]
        if attr not in nominal_attrs:
            final_orders.append(np.arange(num_values, dtype=np.int64))
            continue
        cluster_rankings = np.zeros((k, num_values), dtype=np.float64)
        for cluster in range(k):
            significance = np.zeros(num_values, dtype=np.float64)
            for value in range(num_values):
                cost = float(distance_matrices[attr][value] @ attr_probs[cluster])
                if cost == 0.0:
                    cost = 1e-6
                significance[value] = attr_probs[cluster, value] / cost
            cluster_rankings[cluster] = _interleaved_rank_order(significance)
        combined = cluster_weights @ cluster_rankings
        final_orders.append(np.argsort(combined, kind="mergesort"))
    return final_orders


def _apply_orders(
    X: np.ndarray,
    orders: list[np.ndarray],
    order_labels: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    reordered = X.copy()
    new_order_labels = [labels.copy() for labels in order_labels]
    for attr, mapping in enumerate(orders):
        reordered[:, attr] = mapping[reordered[:, attr]]

        updated_labels = np.empty_like(order_labels[attr])
        for old_code, new_code in enumerate(mapping):
            updated_labels[new_code] = order_labels[attr][old_code]
        new_order_labels[attr] = updated_labels
    return reordered, new_order_labels


def _inner_loop(
    X: np.ndarray,
    mode_probs: list[np.ndarray],
    k: int,
    num_values: np.ndarray,
    init_assignments: np.ndarray,
    distance_matrices: list[np.ndarray],
    objective_history: list[float],
    distance_fn=_sample_cluster_distances,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the inner assignment–update loop to convergence.

    ``distance_fn`` controls which distance metric is used:
    - ``_sample_cluster_distances`` → full OCL (probability-aware)
    - ``_hard_mode_distances``        → OCL-I (equidistant, no prob weighting)
    - ``_hamming_distances``          → OCL-III (Hamming, no order info)
    """
    assignments = init_assignments.copy()
    while True:
        if distance_fn is _hamming_distances:
            distances = distance_fn(X, mode_probs)
        else:
            distances = distance_fn(X, mode_probs, distance_matrices)
        winners = distances.argmin(axis=1)
        changed = not np.array_equal(winners, assignments)
        assignments = winners
        mode_probs = _mode_probabilities(X, assignments, k, num_values, previous=mode_probs)
        objective_history.append(_objective_from_distances(distances, assignments))
        if not changed:
            break
    return assignments, mode_probs


class OCL:
    """Order-learning clustering for categorical data.

    Parameters
    ----------
    max_outer_loops : int
        Maximum number of outer (order-learning) iterations.
    max_init_loops : int
        Maximum k-modes iterations during initialisation.
    seed : int or None
        Random seed.
    variant : str
        Ablation variant:

        - ``"full"`` — complete OCL with probability-aware order distance
          and iterative order learning.
        - ``"ocl1"`` — OCL-I: equidistant order distance **without**
          probability weighting (hard mode distance).
        - ``"ocl2"`` — OCL-II: same as OCL-I but order is computed **once**
          and never updated (single outer pass).
        - ``"ocl3"`` — OCL-III: traditional Hamming distance, no order
          information (essentially KMD).
    """

    def __init__(
        self,
        max_outer_loops: int = 50,
        max_init_loops: int = 50,
        seed: int | None = None,
        variant: str = "full",
        nominal_attrs: list[int] | None = None,
    ) -> None:
        if variant not in ("full", "ocl1", "ocl2", "ocl3", "lnro", "rnro"):
            raise ValueError(f"Unknown variant {variant!r}.")
        self.max_outer_loops = max_outer_loops
        self.max_init_loops = max_init_loops
        self.seed = seed
        self.variant = variant
        self.nominal_attrs = nominal_attrs or []

    def fit_predict(
        self,
        features: np.ndarray,
        true_labels: np.ndarray | None = None,
        n_clusters: int | None = None,
    ) -> OCLResult:
        encoded_features, original_values = encode_features(np.asarray(features))
        X = encoded_features.copy()
        n_samples, n_attrs = X.shape
        if n_clusters is None:
            if true_labels is None:
                raise ValueError("n_clusters must be provided when true_labels is absent.")
            n_clusters = int(np.max(true_labels))
        k = int(n_clusters)

        num_values = np.array([len(values) for values in original_values], dtype=np.int64)
        distance_matrices = [_distance_matrix(int(m)) for m in num_values]
        rng = np.random.RandomState(self.seed)

        # --- Select the distance function based on variant ---
        if self.variant == "ocl3":
            distance_fn = _hamming_distances
            outer_loop_limit = 0   # No order learning at all
        elif self.variant == "ocl1":
            distance_fn = _hard_mode_distances
            outer_loop_limit = self.max_outer_loops
        elif self.variant == "ocl2":
            distance_fn = _hard_mode_distances
            outer_loop_limit = 0   # Order computed once, then fixed
        elif self.variant == "rnro":
            distance_fn = _sample_cluster_distances  # placeholder, overridden below
            outer_loop_limit = 0   # No order learning, use mixed distance
        elif self.variant == "lnro":
            distance_fn = _sample_cluster_distances
            outer_loop_limit = self.max_outer_loops
        else:  # "full"
            distance_fn = _sample_cluster_distances
            outer_loop_limit = self.max_outer_loops

        modes, previous_outer_assignments = _kmodes_initialization(
            X, k, num_values, rng, self.max_init_loops
        )
        mode_probs = _mode_point_probabilities(modes, num_values)
        current_assignments = np.full(n_samples, -1, dtype=np.int64)
        objective_history: list[float] = []
        order_update_iterations: list[int] = []
        order_labels = [values.copy() for values in original_values]

        # RNRO: mixed distance — Hamming for nominal, order dist for ordinal
        if self.variant == "rnro":
            def _rnro_dist(X, mp, dm):
                return _mixed_nom_ord_distances(X, mp, dm, self.nominal_attrs)
            distance_fn = _rnro_dist

        outer_loops = 0
        while outer_loops <= outer_loop_limit:
            outer_loops += 1
            current_assignments, mode_probs = _inner_loop(
                X,
                mode_probs,
                k,
                num_values,
                current_assignments,
                distance_matrices,
                objective_history,
                distance_fn=distance_fn,
            )
            if outer_loop_limit == 0:
                break

            if np.array_equal(previous_outer_assignments, current_assignments):
                break

            previous_outer_assignments = current_assignments.copy()
            if self.variant == "lnro":
                learned_orders = _choose_orders_nominal(
                    current_assignments, mode_probs, distance_matrices, self.nominal_attrs
                )
            else:
                learned_orders = _choose_orders(current_assignments, mode_probs, distance_matrices)
            X, order_labels = _apply_orders(X, learned_orders, order_labels)
            order_update_iterations.append(len(objective_history))

        result = OCLResult(
            assignments=current_assignments + 1,
            learned_orders=decode_orders(order_labels),
            objective_history=objective_history,
            order_update_iterations=order_update_iterations,
        )

        if true_labels is not None:
            result.ca = clustering_accuracy(result.assignments, true_labels)
            result.ari = adjusted_rand_index(result.assignments, true_labels)
            result.nmi = normalized_mutual_information(true_labels, result.assignments)
        result.cmp = clustering_compactness(encoded_features, result.assignments, num_values)
        return result
