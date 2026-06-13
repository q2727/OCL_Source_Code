"""t-SNE visualization on VT dataset — paper Figure 5 style."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from ocl_python import KModes, OCL, load_dataset
from ocl_python.data import encode_features

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def order_distance_matrix(X_ordered: np.ndarray) -> np.ndarray:
    """Pairwise order distance matrix using OCL-learned positions."""
    n, s = X_ordered.shape
    D = np.zeros((n, n), dtype=np.float64)
    for attr in range(s):
        col = X_ordered[:, attr].astype(np.float64)
        max_v = float(col.max())
        if max_v > 0:
            D += np.abs(col[:, None] - col[None, :]) / max_v
    return D / float(s)


def hamming_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance matrix."""
    n, s = X.shape
    D = np.zeros((n, n), dtype=np.float64)
    for attr in range(s):
        col = X[:, attr].astype(np.float64)
        D += (col[:, None] != col[None, :]).astype(np.float64)
    return D / float(s)


def main():
    ds = load_dataset("VT")
    labels = ds.labels.ravel().astype(np.int64)

    # OCL: learn orders → compute order distance → t-SNE
    ocl = OCL(seed=SEED)
    ocl_res = ocl.fit_predict(ds.features, ds.labels)

    X_enc, orig_vals = encode_features(np.asarray(ds.features))
    mappings = []
    for attr, order in enumerate(ocl_res.learned_orders):
        mapping = np.zeros(len(order), dtype=np.int64)
        for rank, val in enumerate(order):
            code = int(np.where(orig_vals[attr] == val)[0][0])
            mapping[code] = rank
        mappings.append(mapping)

    X_ordered = X_enc.copy()
    for attr, mapping in enumerate(mappings):
        X_ordered[:, attr] = mapping[X_ordered[:, attr]]

    D_ocl = order_distance_matrix(X_ordered)
    tsne_ocl = TSNE(
        n_components=2, metric="precomputed", init="random", random_state=SEED, perplexity=30
    )
    emb_ocl = tsne_ocl.fit_transform(D_ocl)

    # KMD: Hamming distance → t-SNE
    kmd = KModes(seed=SEED)
    kmd_res = kmd.fit_predict(ds.features, ds.labels)
    X_raw, _ = encode_features(np.asarray(ds.features))
    D_kmd = hamming_distance_matrix(X_raw)
    tsne_kmd = TSNE(
        n_components=2, metric="precomputed", init="random", random_state=SEED, perplexity=30
    )
    emb_kmd = tsne_kmd.fit_transform(D_kmd)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"t-SNE Visualization — VT Dataset  "
        f"(OCL CA={ocl_res.ca:.3f} vs KMD CA={kmd_res.ca:.3f})",
        fontsize=14, fontweight="bold",
    )

    markers = ["o", "s"]
    for ax, emb, title, ca in [
        (axes[0], emb_kmd, "KMD (Hamming distance)", kmd_res.ca),
        (axes[1], emb_ocl, "OCL (Order distance)", ocl_res.ca),
    ]:
        for c in range(1, int(labels.max()) + 1):
            mask = labels == c
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                marker=markers[(c - 1) % 2], s=30, alpha=0.7,
                label=f"Cluster {c}",
            )
        ax.set_title(f"{title}\nCA = {ca:.3f}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsne_vt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved tsne_vt.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    main()
