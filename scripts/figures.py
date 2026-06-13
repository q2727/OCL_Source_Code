"""Generate key figures for PPT."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["AC", "AP", "BC", "CS", "DS", "HR", "LG", "NS", "SB", "TT", "VT", "ZO"]


def plot_ocl_vs_kmd():
    """OCL vs KMD bar chart across 12 datasets."""
    ocl_ca = [0.7896, 0.6190, 0.6818, 0.6338, 0.6742, 0.3621, 0.5324, 0.3292, 0.9617, 0.5630, 0.8943, 0.7733]
    kmd_ca = [0.7186, 0.5360, 0.5318, 0.5525, 0.6742, 0.3848, 0.4554, 0.3384, 0.7532, 0.5391, 0.8655, 0.7010]

    x = np.arange(len(DATASETS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, kmd_ca, w, label="KMD (Hamming)", color="#e74c3c", alpha=0.85)
    ax.bar(x + w/2, ocl_ca, w, label="OCL", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("CA")
    ax.set_title("OCL vs KMD — Clustering Accuracy on 12 Datasets")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ocl_vs_kmd.png", dpi=150)
    plt.close(fig)


def plot_ablation():
    """Ablation waterfall: OCL > OCL-I > OCL-II > OCL-III."""
    variants = ["OCL", "OCL-I", "OCL-II", "OCL-III"]
    ca_avg = [0.6512, 0.6174, 0.6074, 0.5875]
    ari_avg = [0.2730, 0.2346, 0.2202, 0.1970]
    drops_ca = [0, 0.0338, 0.0100, 0.0199]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart
    x = np.arange(len(variants))
    ax1.bar(x, ca_avg, color=["#3498db", "#5dade2", "#85c1e9", "#aed6f1"], edgecolor="white")
    for i, (v, ca) in enumerate(zip(variants, ca_avg)):
        ax1.text(i, ca + 0.005, f"{ca:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=11)
    ax1.set_ylabel("CA (12-dataset avg)")
    ax1.set_ylim(0.55, 0.68)
    ax1.grid(axis="y", alpha=0.3)

    # Right: waterfall
    cumulative = [0.6512]
    for d in drops_ca[1:]:
        cumulative.append(cumulative[-1] - d)
    colors = ["#3498db", "#e74c3c", "#e67e22", "#95a5a6"]
    labels = ["OCL", "-prob weight", "-iter order", "-order info"]
    for i in range(len(labels)):
        bottom = cumulative[i] if i < len(cumulative) else 0
        ax2.bar(0, drops_ca[i], bottom=bottom if i > 0 else 0,
                color=colors[i], label=labels[i], edgecolor="white", width=0.5)
    ax2.set_xticks([])
    ax2.set_ylabel("CA")
    ax2.set_title("Contribution of Each Component")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.57, 0.67)

    fig.suptitle("Ablation Study — 12 Dataset Average", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation.png", dpi=150)
    plt.close(fig)


def plot_lnro_rnro():
    """LNRO/RNRO comparison on 9 datasets."""
    ds9 = ["NS", "AP", "CS", "HR", "BC", "LG", "CR", "OB", "BM"]
    ocl = [0.3213, 0.6190, 0.6183, 0.3864, 0.6762, 0.5405, 0.5332, 0.3848, 0.5925]
    lnro = [0.3243, 0.6032, 0.6187, 0.3760, 0.5140, 0.4624, 0.5329, 0.3776, 0.6051]
    rnro = [0.3253, 0.5367, 0.5492, 0.3965, 0.5228, 0.4245, 0.5354, 0.3716, 0.5654]

    x = np.arange(len(ds9))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, ocl, w, label="OCL", color="#3498db", alpha=0.85)
    ax.bar(x, lnro, w, label="LNRO", color="#2ecc71", alpha=0.85)
    ax.bar(x + w, rnro, w, label="RNRO", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ds9)
    ax.set_ylabel("CA")
    ax.set_title("Nominal/Ordinal Ablation (Table 10) — OCL vs LNRO vs RNRO")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lnro_rnro.png", dpi=150)
    plt.close(fig)


def plot_paper_vs_ours():
    """Scatter: paper OCL CA vs our OCL CA."""
    paper_ca = [0.8206, 0.6222, 0.6650, 0.6313, 0.7458, 0.4326, 0.5426, 0.3573, 0.9830, 0.5785, 0.8943, 0.7792]
    our_ca = [0.7896, 0.6190, 0.6818, 0.6338, 0.6742, 0.3621, 0.5324, 0.3292, 0.9617, 0.5630, 0.8943, 0.7733]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(paper_ca, our_ca, s=60, c="#3498db", edgecolors="white", linewidth=0.5, zorder=3)
    ax.plot([0.3, 1.0], [0.3, 1.0], "k--", alpha=0.3)
    for i, ds in enumerate(DATASETS):
        dx = our_ca[i] - paper_ca[i]
        color = "#e74c3c" if abs(dx) > 0.03 else "#27ae60"
        ax.annotate(ds, (paper_ca[i], our_ca[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)
    ax.set_xlabel("Paper CA")
    ax.set_ylabel("Our CA")
    ax.set_title("Paper vs Ours — OCL Clustering Accuracy")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "paper_vs_ours.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    plot_ocl_vs_kmd()
    plot_ablation()
    plot_lnro_rnro()
    plot_paper_vs_ours()
    print("Done: ocl_vs_kmd.png, ablation.png, lnro_rnro.png, paper_vs_ours.png")
