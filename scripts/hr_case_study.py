"""HR case study — paper Figure 6: learned orders on Hayes-Roth dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ocl_python import OCL, load_dataset

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HR_ATTR_NAMES = ["Hobby", "Education Level", "Age", "Marital Status"]
HR_VALUE_LABELS = {
    0: {1: "chess", 2: "sport", 3: "stamp"},
    1: {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"},
    2: {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"},
    3: {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"},
}
SEED = 42


def main():
    ds = load_dataset("HR")
    m = OCL(seed=SEED)
    r = m.fit_predict(ds.features, ds.labels)

    orders = r.learned_orders

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Learned Value Orders on HR (Hayes-Roth) — OCL seed={SEED}  "
        f"CA={r.ca:.4f}  ARI={r.ari:.4f}  NMI={r.nmi:.4f}",
        fontsize=13, fontweight="bold",
    )

    for attr_idx, ax in enumerate(axes):
        learned = orders[attr_idx]
        n = len(learned)
        orig_labels = [HR_VALUE_LABELS[attr_idx][v] for v in range(1, n + 1)]
        learned_labels = [HR_VALUE_LABELS[attr_idx][v] for v in learned]

        y_orig = np.arange(n)
        y_learned = np.array(
            [learned.index(v) for v in range(1, n + 1)]
        )

        for j in range(n):
            moved = y_learned[j] != j
            ax.annotate(
                "", xy=(1, y_learned[j]), xytext=(0, y_orig[j]),
                arrowprops=dict(
                    arrowstyle="->", color="C0" if moved else "gray",
                    lw=2.5, alpha=0.7,
                ),
            )
            ax.text(-0.15, y_orig[j], orig_labels[j], ha="right", va="center",
                    fontsize=10, fontweight="bold")
            ax.text(1.15, y_learned[j], learned_labels[j], ha="left", va="center",
                    fontsize=10, fontweight="bold",
                    color="C0" if moved else "gray")

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(HR_ATTR_NAMES[attr_idx], fontsize=12, fontweight="bold")

        if attr_idx == 0:
            ax.text(-0.4, -1.0, "Original", ha="center", fontsize=9,
                    style="italic", color="gray")
            ax.text(1.4, -1.0, "Learned", ha="center", fontsize=9,
                    style="italic", color="C0")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hr_learned_orders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved hr_learned_orders.png")
    print("\nOrder changes:")
    for attr_idx in range(4):
        learned = orders[attr_idx]
        orig = [HR_VALUE_LABELS[attr_idx][v] for v in range(1, len(learned) + 1)]
        learned_l = [HR_VALUE_LABELS[attr_idx][v] for v in learned]
        changed = orig != learned_l
        print(f"  {HR_ATTR_NAMES[attr_idx]}: "
              f"{' > '.join(orig)}  =>  {' > '.join(learned_l)}"
              f"{'  ***' if changed else ''}")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    main()
