"""Plot the basis comparison from results/*.json into figures/.   python plot_results.py

fig_results.png  every seed's score (dots) and the mean (line), best epoch vs final epoch, per task
fig_curves.png   validation accuracy and training loss per epoch, mean over seeds, per task
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"

# categorical slots 1-3 of the reference palette, validated all-pairs for colour-vision deficiency;
# a basis keeps its colour in every chart
BASES = ["frame", "random", "identity"]
COLOR = {"frame": "#2a78d6", "random": "#eb6834", "identity": "#1baf7a"}
TASKS = {"rte": "RTE", "mrpc": "MRPC"}
PAPER = {"rte": 79.8}  # paper's frame result; its MRPC 92.3 matches our F1, not accuracy, so it isn't drawn
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def load_runs():
    runs = defaultdict(list)
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text())
        if r["status"] == "done":
            runs[(r["task"], r["basis"])].append(r)
    return runs


def style_title(fig, title, subtitle):
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)


def plot_results(runs):
    """Dot plot: one dot per seed, a short line at the mean. Rows: best epoch, final epoch."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), sharey="col")
    seed_offset = {42: -0.09, 43: 0.0, 44: 0.09}
    for col, (task, task_name) in enumerate(TASKS.items()):
        for row, (key, row_name) in enumerate([("best", "Best epoch"), ("final", "Final epoch")]):
            ax = axes[row, col]
            for i, basis in enumerate(BASES):
                rs = runs.get((task, basis), [])
                values = [100 * r[key] for r in rs]
                if not values:
                    continue
                ax.scatter([i + seed_offset[r["seed"]] for r in rs], values, s=46, color=COLOR[basis],
                           edgecolors=SURFACE, linewidths=1.5, zorder=3)
                mean = statistics.mean(values)
                ax.plot([i - 0.2, i + 0.2], [mean, mean], color=COLOR[basis], linewidth=2, zorder=2)
                ax.text(i + 0.24, mean, f"{mean:.1f}", va="center", ha="left", fontsize=9.5, color=INK)
            if row == 0 and task in PAPER:
                ax.axhline(PAPER[task], color=MUTED, linewidth=0.8, zorder=1)
                ax.text(2.55, PAPER[task] - 0.15, f"paper (frame) {PAPER[task]}", va="top", ha="right",
                        fontsize=8.5, color=MUTED)
            counts = [len(runs.get((task, b), [])) for b in BASES]
            ax.set_xticks(range(3), [f"{b}\nn={n}" for b, n in zip(BASES, counts)])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, 2.6)
            ax.set_title(f"{task_name} · {row_name.lower()}", loc="left", fontsize=10.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel("Validation accuracy (%)")

    legend = [Line2D([], [], marker="o", linestyle="", markersize=7, color=MUTED, markeredgecolor=SURFACE,
                     label="one seed"),
              Line2D([], [], color=MUTED, linewidth=2, label="mean over seeds")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_2)
    style_title(fig, "The identity basis trails clearly; frame and random are close",
                "RoBERTa-base, 1,000 coefficients per matrix, paper hyperparameters. Best epoch = peak of all "
                "epochs; final = last epoch.")
    fig.text(0.02, 0.012, "The paper's MRPC 92.3 matches our frame runs' best-epoch F1 (92.4, 92.5), "
             "so it is not drawn on the accuracy axis.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.87), h_pad=2.5, w_pad=3)
    fig.savefig(FIGURES / "fig_results.png", dpi=200)
    plt.close(fig)


def rolling(values, window):
    return [statistics.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def plot_curves(runs):
    """Mean over seeds per epoch; band = min to max across seeds. Top: validation accuracy, bottom: train loss."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex="col")
    for col, (task, task_name) in enumerate(TASKS.items()):
        for basis in BASES:
            rs = runs.get((task, basis), [])
            if not rs:
                continue
            epochs = range(1, len(rs[0]["per_epoch"]) + 1)
            # both are noisy epoch to epoch (train_loss is only the last batch), so smooth over 5 epochs
            acc = [rolling([100 * e["metric"] for e in r["per_epoch"]], 5) for r in rs]
            loss = [rolling([e["train_loss"] for e in r["per_epoch"]], 5) for r in rs]
            for row, series in enumerate([acc, loss]):
                ax = axes[row, col]
                per_epoch = list(zip(*series))
                ax.plot(epochs, [statistics.mean(v) for v in per_epoch], color=COLOR[basis], linewidth=2,
                        label=f"{basis} (n={len(rs)})")
                if len(rs) > 1:
                    ax.fill_between(epochs, [min(v) for v in per_epoch], [max(v) for v in per_epoch],
                                    color=COLOR[basis], alpha=0.10, linewidth=0)
        axes[0, col].set_title(f"{task_name} · validation accuracy (5-epoch average)", loc="left", fontsize=10.5,
                               color=INK, pad=8)
        axes[1, col].set_title(f"{task_name} · training loss (5-epoch average)", loc="left", fontsize=10.5,
                               color=INK, pad=8)
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylim(bottom=0)
        for ax in axes[:, col]:
            ax.margins(x=0.01)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Loss")
    axes[0, 0].set_ylim(45, 85)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    mrpc_counts = {b: len(runs.get(("mrpc", b), [])) for b in BASES}
    labels = [lab.split(" (")[0] for lab in labels]
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK_2, handlelength=1.6)
    style_title(fig, "All three bases fit the training data; identity generalizes worse",
                "Lines = mean over seeds, bands = min to max across seeds. Seeds per basis: RTE 3 each; MRPC "
                + ", ".join(f"{b} {n}" for b, n in mrpc_counts.items()) + ".")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.5, w_pad=3)
    fig.savefig(FIGURES / "fig_curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    runs = load_runs()
    plot_results(runs)
    plot_curves(runs)
    print(f"wrote {FIGURES / 'fig_results.png'} and {FIGURES / 'fig_curves.png'}")
