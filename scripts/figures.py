"""Generate key figures for PPT."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ocl_python.data import load_dataset
from ocl_python.metrics import normalized_mutual_information

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["AC", "AP", "BC", "CS", "DS", "HR", "LG", "NS", "SB", "TT", "VT", "ZO"]
WOCL_MAIN_RESULT = Path("results/experiments/main_20260615T131021Z.md")
WOCL_DIAGNOSED_POSITIVE_DATASETS = ["NS", "TT", "VT"]
WOCL_EXTRA_POSITIVE_DATASETS = ["CS", "ZO"]
WOCL_NON_DEGRADATION_DATASETS = ["AC", "BC", "DS", "SB"]
WOCL_STABLE_MOTIVATED_DATASETS = (
    WOCL_DIAGNOSED_POSITIVE_DATASETS
    + WOCL_EXTRA_POSITIVE_DATASETS
    + WOCL_NON_DEGRADATION_DATASETS
)

# Read-only diagnostics used only for motivation/explanation. They are not used
# by WOCL during training.
LEAVE_ONE_ATTRIBUTE_CA_GAIN = {
    "HR": 0.1263,
    "NS": 0.0307,
    "TT": 0.0275,
    "VT": 0.0069,
    "AP": 0.0053,
}


def _parse_experiment_table(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if " | " not in line or line.startswith("Dataset") or line.startswith("---"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 6:
            continue
        dataset, method = parts[0], parts[1]
        try:
            values = [float(cell.split("+/-")[0].strip()) for cell in parts[2:]]
        except ValueError:
            continue
        rows[(dataset, method)] = {
            "CA": values[0],
            "ARI": values[1],
            "NMI": values[2],
            "CMP": values[3],
        }
    return rows


def _attribute_label_nmi_summary() -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        loaded = load_dataset(dataset)
        values = np.array(
            [
                normalized_mutual_information(loaded.features[:, attr], loaded.labels)
                for attr in range(loaded.features.shape[1])
            ],
            dtype=np.float64,
        )
        summary[dataset] = {
            "weak_ratio": float(np.mean(values < 0.02)),
            "cv": float(np.std(values) / (np.mean(values) + 1e-12)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return summary


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


def plot_wocl_motivation():
    """Motivation: equal attribute weights can dilute or amplify weak attributes."""
    summary = _attribute_label_nmi_summary()

    weak_ratio = [summary[dataset]["weak_ratio"] for dataset in DATASETS]
    cv = [summary[dataset]["cv"] for dataset in DATASETS]
    colors = [
        "#2ecc71" if dataset in WOCL_DIAGNOSED_POSITIVE_DATASETS
        else "#f39c12" if dataset in LEAVE_ONE_ATTRIBUTE_CA_GAIN
        else "#95a5a6"
        for dataset in DATASETS
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    x = np.arange(len(DATASETS))
    ax1.bar(x, weak_ratio, color=colors, alpha=0.88, label="Weak-attribute ratio")
    ax1.plot(x, cv, color="#2c3e50", marker="o", linewidth=2.0, label="NMI dispersion (CV)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(DATASETS)
    ax1.set_ylabel("Diagnostic value")
    ax1.set_title("Attribute-label NMI is uneven across attributes")
    ax1.set_ylim(0.0, max(max(cv), max(weak_ratio)) + 0.25)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(
        handles=[
            Line2D([0], [0], color="#2c3e50", marker="o", linewidth=2.0, label="NMI dispersion (CV)"),
            Patch(facecolor="#95a5a6", alpha=0.88, label="Weak-attribute ratio"),
            Patch(facecolor="#2ecc71", alpha=0.88, label="Diagnosed + WOCL improves"),
            Patch(facecolor="#f39c12", alpha=0.88, label="Diagnosed / open issue"),
        ],
        fontsize=8,
        loc="upper right",
    )

    sorted_gain_items = sorted(
        LEAVE_ONE_ATTRIBUTE_CA_GAIN.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    gain_datasets = [item[0] for item in sorted_gain_items]
    gain_values = [item[1] for item in sorted_gain_items]
    gain_colors = [
        "#2ecc71" if dataset in WOCL_DIAGNOSED_POSITIVE_DATASETS else "#f39c12"
        for dataset in gain_datasets
    ]
    y = np.arange(len(gain_datasets))
    ax2.barh(y, gain_values, color=gain_colors, alpha=0.88)
    ax2.set_yticks(y)
    ax2.set_yticklabels(gain_datasets)
    ax2.invert_yaxis()
    ax2.set_xlabel("Best CA gain after removing one attribute")
    ax2.set_title("Some equal-weighted attributes hurt OCL")
    ax2.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(gain_values):
        ax2.text(value + 0.002, idx, f"+{value:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Motivation for WOCL: equal attribute weighting is not always reliable",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wocl_motivation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wocl_vs_ocl():
    """WOCL extension result on the latest 30-run main-dataset experiment."""
    rows = _parse_experiment_table(WOCL_MAIN_RESULT)
    methods = ["KMD", "OCL", "WOCL"]
    ca_by_method = {
        method: [rows[(dataset, method)]["CA"] for dataset in DATASETS]
        for method in methods
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    x = np.arange(len(DATASETS))
    w = 0.25
    colors = {"KMD": "#e74c3c", "OCL": "#3498db", "WOCL": "#2ecc71"}
    offsets = {"KMD": -w, "OCL": 0.0, "WOCL": w}
    labels = {"KMD": "KMD", "OCL": "OCL", "WOCL": "WOCL"}
    for method in methods:
        ax1.bar(
            x + offsets[method],
            ca_by_method[method],
            w,
            label=labels[method],
            color=colors[method],
            alpha=0.85,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(DATASETS)
    ax1.set_ylabel("CA")
    ax1.set_title("WOCL vs OCL — CA on 12 Main Datasets")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    metric_names = ["CA", "ARI", "NMI", "CMP"]
    avg = {
        method: [
            float(np.mean([rows[(dataset, method)][metric] for dataset in DATASETS]))
            for metric in metric_names
        ]
        for method in methods
    }
    x2 = np.arange(len(metric_names))
    for idx, method in enumerate(methods):
        ax2.bar(
            x2 + (idx - 1) * w,
            avg[method],
            w,
            label=labels[method],
            color=colors[method],
            alpha=0.85,
        )
        for j, value in enumerate(avg[method]):
            ax2.text(
                x2[j] + (idx - 1) * w,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["CA↑", "ARI↑", "NMI↑", "CMP↓"])
    ax2.set_ylim(0.0, max(max(values) for values in avg.values()) + 0.08)
    ax2.set_title("30-run Average on 12 Main Datasets")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Attribute-Weighted OCL (WOCL): delay=1, mix=0.5, objective guard",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wocl_vs_ocl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wocl_motivated_results():
    """WOCL result on datasets where equal-weight motivation is visible."""
    rows = _parse_experiment_table(WOCL_MAIN_RESULT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    metrics = ["CA", "ARI", "NMI"]
    metric_colors = {"CA": "#3498db", "ARI": "#2ecc71", "NMI": "#9b59b6"}
    x = np.arange(len(WOCL_DIAGNOSED_POSITIVE_DATASETS))
    w = 0.24
    for idx, metric in enumerate(metrics):
        values = [
            rows[(dataset, "WOCL")][metric] - rows[(dataset, "OCL")][metric]
            for dataset in WOCL_DIAGNOSED_POSITIVE_DATASETS
        ]
        ax1.bar(
            x + (idx - 1) * w,
            values,
            w,
            label=f"Δ{metric}",
            color=metric_colors[metric],
            alpha=0.88,
        )
        for j, value in enumerate(values):
            ax1.text(
                x[j] + (idx - 1) * w,
                value + 0.0009,
                f"{value:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(WOCL_DIAGNOSED_POSITIVE_DATASETS)
    ax1.set_ylabel("WOCL - OCL")
    ax1.set_title("Diagnosed equal-weight issue: WOCL improves")
    ax1.axhline(0.0, color="#2c3e50", linewidth=0.8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0.0, 0.034)

    nmi_delta = [
        rows[(dataset, "WOCL")]["NMI"] - rows[(dataset, "OCL")]["NMI"]
        for dataset in DATASETS
    ]
    delta_colors = [
        "#2ecc71" if value > 0.0005
        else "#e74c3c" if value < -0.0005
        else "#95a5a6"
        for value in nmi_delta
    ]
    x2 = np.arange(len(DATASETS))
    ax2.bar(x2, nmi_delta, color=delta_colors, alpha=0.88)
    ax2.axhline(0.0, color="#2c3e50", linewidth=0.8)
    for j, value in enumerate(nmi_delta):
        if abs(value) >= 0.003 or DATASETS[j] in WOCL_DIAGNOSED_POSITIVE_DATASETS:
            va = "bottom" if value >= 0 else "top"
            offset = 0.0012 if value >= 0 else -0.0012
            ax2.text(
                x2[j],
                value + offset,
                f"{value:+.3f}",
                ha="center",
                va=va,
                fontsize=8,
                rotation=90,
            )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(DATASETS)
    ax2.set_ylabel("ΔNMI (WOCL - OCL)")
    ax2.set_title("All main datasets: positives, near-zero cases, and limitations")
    ax2.set_ylim(min(nmi_delta) - 0.006, max(nmi_delta) + 0.006)
    ax2.legend(
        handles=[
            Patch(facecolor="#2ecc71", alpha=0.88, label="NMI improves"),
            Patch(facecolor="#95a5a6", alpha=0.88, label="Near zero"),
            Patch(facecolor="#e74c3c", alpha=0.88, label="Current limitation"),
        ],
        fontsize=8,
        loc="upper left",
    )
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "WOCL improves where equal attribute weighting is a visible issue",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wocl_motivated_results.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "wocl_focused_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10})
    plot_ocl_vs_kmd()
    plot_ablation()
    plot_lnro_rnro()
    plot_paper_vs_ours()
    plot_wocl_motivation()
    plot_wocl_vs_ocl()
    plot_wocl_motivated_results()
    print(
        "Done: ocl_vs_kmd.png, ablation.png, lnro_rnro.png, "
        "paper_vs_ours.png, wocl_motivation.png, wocl_vs_ocl.png, "
        "wocl_motivated_results.png"
    )
