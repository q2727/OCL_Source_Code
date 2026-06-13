from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    pred = np.asarray(pred_labels, dtype=np.int64).ravel()
    true = np.asarray(true_labels, dtype=np.int64).ravel()
    if pred.shape != true.shape:
        raise ValueError("pred_labels and true_labels must have the same shape")

    pred_codes = np.unique(pred, return_inverse=True)[1]
    true_codes = np.unique(true, return_inverse=True)[1]
    n_classes = max(pred_codes.max(), true_codes.max()) + 1

    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(contingency, (pred_codes, true_codes), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / pred.size)


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.shape != b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)

    n_a = int(a.max()) + 1
    n_b = int(b.max()) + 1
    contingency = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)

    def comb2(values: np.ndarray) -> float:
        values = values.astype(np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    sum_comb = comb2(contingency)
    row_comb = comb2(contingency.sum(axis=1))
    col_comb = comb2(contingency.sum(axis=0))
    total_comb = float(contingency.sum() * (contingency.sum() - 1) / 2.0)
    if total_comb == 0.0:
        return 0.0

    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denominator = max_index - expected
    if denominator == 0.0:
        return 0.0
    return float((sum_comb - expected) / denominator)


def clustering_compactness(
    X: np.ndarray,
    assignments: np.ndarray,
    num_values: np.ndarray,
) -> float:
    """CMP: Clustering coMPactness — an internal entropy-based index (Eq. 17).

    For each cluster and each attribute, the normalised entropy of the
    value distribution is computed.  CMP is the average across all clusters
    and attributes.  **Lower is better** (range [0, 1]).

    .. math::
        CMP = \\frac{1}{s \\cdot k} \\sum_{j=1}^{k} \\sum_{r=1}^{s}
              \\frac{H(C_j, r)}{\\log V_r}

    where :math:`H(C_j, r)` is the Shannon entropy of the value
    distribution of attribute *r* within cluster *j*, and :math:`V_r`
    is the number of distinct values of attribute *r*.
    """
    X_arr = np.asarray(X, dtype=np.int64)
    assignments_arr = np.asarray(assignments, dtype=np.int64).ravel()
    num_values_arr = np.asarray(num_values, dtype=np.int64).ravel()

    n_attrs = X_arr.shape[1]
    k = int(assignments_arr.max())

    total_norm_entropy = 0.0
    for cluster in range(1, k + 1):
        mask = assignments_arr == cluster
        if not np.any(mask):
            continue
        cluster_data = X_arr[mask]
        for attr in range(n_attrs):
            counts = np.bincount(
                cluster_data[:, attr], minlength=int(num_values_arr[attr])
            ).astype(np.float64)
            probs = counts / counts.sum()
            nz = probs > 0.0
            entropy = -np.sum(probs[nz] * np.log(probs[nz]))
            max_entropy = np.log(float(num_values_arr[attr]))
            if max_entropy > 0.0:
                total_norm_entropy += entropy / max_entropy

    return float(total_norm_entropy / (n_attrs * k))


def normalized_mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.shape != b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)

    n_a = int(a.max()) + 1
    n_b = int(b.max()) + 1
    contingency = np.zeros((n_a, n_b), dtype=np.float64)
    np.add.at(contingency, (a, b), 1.0)
    contingency /= contingency.sum()

    px = contingency.sum(axis=1)
    py = contingency.sum(axis=0)
    nz = contingency > 0.0
    mi = float(
        np.sum(contingency[nz] * np.log2(contingency[nz] / (px[:, None] * py[None, :])[nz]))
    )

    def entropy(probabilities: np.ndarray) -> float:
        probs = probabilities[probabilities > 0.0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    hx = entropy(px)
    hy = entropy(py)
    if hx == 0.0 or hy == 0.0:
        return 0.0
    return float(max(0.0, math.sqrt((mi / hx) * (mi / hy))))
