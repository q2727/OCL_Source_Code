"""Convergence curves — paper Figure 3: SB, NS, HR."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocl_python import OCL, load_dataset

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["SB", "NS", "HR"]
SEEDS = [25, 100, 500]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Convergence Curves of OCL", fontsize=14, fontweight="bold")

    for ax, ds_name in zip(axes, DATASETS):
        ds = load_dataset(ds_name)
        for seed in SEEDS:
            m = OCL(seed=seed)
            r = m.fit_predict(ds.features, ds.labels)
            obj = np.array(r.objective_history)
            iters = np.arange(1, len(obj) + 1)
            ax.plot(iters, obj, linewidth=1.3, alpha=0.8, label=f"seed={seed}")
            for ou in r.order_update_iterations:
                ax.axvline(x=ou, color="red", linestyle="--", alpha=0.3, linewidth=0.6)

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Objective L")
        ax.set_title(f"{ds_name}  (k={ds.true_k})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Single legend entry for red dashed line
    from matplotlib.lines import Line2D
    leg = Line2D([0], [0], color="red", linestyle="--", alpha=0.5)
    axes[0].legend([leg], ["Order update"], loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved convergence.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    main()
