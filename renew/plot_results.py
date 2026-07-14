"""
renew/plot_results.py
=====================
Read cached per-seed results and produce publication-quality figures:
  - Accuracy curves (naive vs RENEW) with CI shading
  - Accuracy bar chart with individual seed scatter
  - Per-seed heatmaps
  - Summary CSV

Usage:
  python renew/plot_results.py --env-name maze10
  python renew/plot_results.py --env-name maze10 --seeds 0 3
  python renew/plot_results.py --env-name maze10 --heatmap-seed 3
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tyro


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "out")

# ── Publication style ────────────────────────────────────────────────────────
# Times-like serif to match the paper body font (rlj.sty loads `times`).
# "Nimbus Roman" / "Liberation Serif" are metric-compatible Times clones that
# ship on most Linux boxes; fall back to whatever serif is available.
_SERIF_STACK = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                "Tinos", "DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": _SERIF_STACK,
    "mathtext.fontset": "stix",   # Times-like math glyphs (bundled with mpl)
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Paper geometry (RLC / rlj.sty) ───────────────────────────────────────────
# The camera-ready figure is authored at its *exact printed size* so that fonts
# specified in points map 1:1 to points in the PDF. rlj.sty sets \textwidth to
# 5.5in and the paper includes this figure at width=0.70\linewidth, so:
PAPER_TEXTWIDTH_IN = 5.5          # \textwidth in rlj.sty
FIG_WIDTH_FRAC     = 0.70         # \includegraphics[width=0.70\linewidth]{...}
FIG_WIDTH_IN       = PAPER_TEXTWIDTH_IN * FIG_WIDTH_FRAC   # = 3.85 in
# Body text is 10pt Times; in-figure text at ~9pt reads as the same font at the
# printed scale (matching caption/body). Because we save at natural size (no
# tight bbox), these point sizes are exactly what appears in the PDF.
PAPER_FS = {
    "col_title":  9.5,  # method name (Pretrained / Naive / RENEW)
    "row_label":  9.0,  # "Transition Errors" / "Decoder Uncertainty"
    "cell_err":   5.5,  # per-cell error count
    "cell_unc":   5.0,  # per-cell uncertainty value
    "cbar_tick":  6.5,  # colorbar end labels (Low / High)
    "cbar_label": 7.5,  # colorbar caption
}

# Colour palette
C_PRETRAIN = "#636e72"   # grey
C_NAIVE    = "#d63031"   # red
C_RENEW    = "#00b894"   # green/teal


@dataclass
class PlotArgs:
    env_name: str = "maze10"
    """Results directory name under out/."""
    seeds: Optional[List[int]] = None
    """Specific seeds to include. Default: all available."""
    maze_size: int = 10
    dpi: int = 200
    heatmap_seed: Optional[int] = None
    """If set, only regenerate heatmaps for this seed."""


# =============================================================================
# Loading helpers
# =============================================================================

def _seed_dir(env_name: str, seed: int) -> str:
    return os.path.join(RESULTS_DIR, env_name, f"seed_{seed}")


def _find_seeds(env_name: str) -> List[int]:
    base = os.path.join(RESULTS_DIR, env_name)
    if not os.path.exists(base):
        return []
    seeds = []
    for name in os.listdir(base):
        if name.startswith("seed_"):
            try:
                seeds.append(int(name.split("_")[1]))
            except ValueError:
                pass
    return sorted(seeds)


def _has_method(env_name: str, seed: int, method: str) -> bool:
    return os.path.exists(os.path.join(_seed_dir(env_name, seed), method, "metrics.json"))


def _load_metrics(env_name: str, seed: int, method: str) -> Dict[str, Any]:
    d = os.path.join(_seed_dir(env_name, seed), method)
    with open(os.path.join(d, "metrics.json")) as f:
        metrics = json.load(f)
    for name in ["error_grid", "uncertainty_grid", "walls"]:
        path = os.path.join(d, f"{name}.npy")
        if os.path.exists(path):
            metrics[name] = np.load(path)
    if "target_rc" in metrics:
        metrics["target_rc"] = tuple(metrics["target_rc"])
    return metrics


def _load_curves(env_name: str, seed: int, method: str):
    """Returns (losses, accuracies, acc_steps)."""
    d = os.path.join(_seed_dir(env_name, seed), method)
    losses, acc_steps, accuracies = [], [], []

    loss_path = os.path.join(d, "losses.csv")
    if os.path.exists(loss_path):
        with open(loss_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                losses.append(float(row["loss"]))

    acc_path = os.path.join(d, "accuracies.csv")
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                acc_steps.append(int(row["step"]))
                accuracies.append(float(row["accuracy"]))

    return losses, accuracies, acc_steps


# =============================================================================
# Heatmap helpers
# =============================================================================

def _add_walls_and_star(ax, walls, target_rc, maze_size):
    for r in range(maze_size):
        for c in range(maze_size):
            if walls[r, c]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            fc="#2d3436", ec="none", zorder=2))
    if target_rc is not None:
        tr, tc = target_rc
        ax.plot(tc, tr, marker="*", markersize=9, color="#ffeaa7",
                markeredgecolor="#2d3436", markeredgewidth=0.5, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _make_error_heatmap(ax, metrics, maze_size):
    error_grid = metrics["error_grid"]
    walls = metrics["walls"]
    target_rc = metrics.get("target_rc")
    n_wrong = error_grid.sum(axis=-1).astype(float)
    n_wrong[walls] = np.nan
    im = ax.imshow(n_wrong, cmap="RdYlGn_r", vmin=0, vmax=4,
                   interpolation="nearest", origin="upper")
    for r in range(maze_size):
        for c in range(maze_size):
            if not walls[r, c] and error_grid[r, c].any():
                ax.text(c, r, str(int(n_wrong[r, c])), ha="center", va="center",
                        fontsize=PAPER_FS["cell_err"], color="white",
                        fontweight="bold", zorder=3)
    _add_walls_and_star(ax, walls, target_rc, maze_size)
    return im


def _make_unc_heatmap(ax, metrics, maze_size, vmax=None):
    unc = metrics["uncertainty_grid"].mean(axis=-1).copy()
    walls = metrics["walls"]
    target_rc = metrics.get("target_rc")
    unc[walls] = np.nan
    # vmax shared across columns so the panels are directly comparable.
    im = ax.imshow(unc, cmap="inferno", vmin=0.0, vmax=vmax,
                   interpolation="nearest", origin="upper")
    for r in range(maze_size):
        for c in range(maze_size):
            if not walls[r, c] and not np.isnan(unc[r, c]) and unc[r, c] > 0.01:
                ax.text(c, r, f"{unc[r,c]:.2f}", ha="center", va="center",
                        fontsize=PAPER_FS["cell_unc"], color="white",
                        fontweight="bold", zorder=3)
    _add_walls_and_star(ax, walls, target_rc, maze_size)
    return im


# =============================================================================
# Per-seed heatmaps
# =============================================================================

def plot_seed_heatmaps(env_name, seed, plot_dir, maze_size, dpi):
    """Camera-ready 2x3 triptych (Figure 3): transition error (top) and
    epistemic uncertainty (bottom) for Pretrained / Naive / RENEW.

    The figure is authored at exactly FIG_WIDTH_IN wide and saved at natural
    size (no tight bbox), so `\\includegraphics[width=0.70\\linewidth]` renders
    it at scale 1.0 and the in-figure fonts match the paper's point sizes.
    """
    methods = ["pretrained", "naive", "renew"]
    titles = ["Pretrained", "Naive", "RENEW"]

    available = [m for m in methods if _has_method(env_name, seed, m)]
    if not available:
        print(f"  Seed {seed}: no data, skipping.")
        return

    # Pre-load metrics and set a shared uncertainty scale so the three columns
    # are directly comparable (per-panel normalisation would hide RENEW's win).
    mets = {m: _load_metrics(env_name, seed, m) for m in available}
    unc_vmax = max(np.nanmax(mets[m]["uncertainty_grid"].mean(axis=-1))
                   for m in available)

    # Height: two square panel rows + a little headroom for the one-line
    # titles. Panels are forced square via set_box_aspect below.
    panel_w = FIG_WIDTH_IN / 3.0
    fig_h = 2 * panel_w + 0.34
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=max(dpi, 300))
    gs = fig.add_gridspec(2, 3, wspace=0.06, hspace=0.06,
                          left=0.070, right=0.995, top=0.90, bottom=0.015)
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
            axes[r, c].set_box_aspect(1)   # force each panel square

    im_unc = None
    for col, (method, title) in enumerate(zip(methods, titles)):
        if method not in mets:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "No data",
                                     transform=axes[row, col].transAxes, ha="center", va="center",
                                     fontsize=PAPER_FS["col_title"], color="#636e72")
                axes[row, col].set_title(title, fontsize=PAPER_FS["col_title"])
                for spine in axes[row, col].spines.values():
                    spine.set_visible(False)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        met = mets[method]
        ax = axes[0, col]
        ax.grid(False)
        _make_error_heatmap(ax, met, maze_size)
        ax.set_title(title, fontsize=PAPER_FS["col_title"], fontweight="bold")

        ax = axes[1, col]
        ax.grid(False)
        im_unc = _make_unc_heatmap(ax, met, maze_size, vmax=unc_vmax)

    axes[0, 0].set_ylabel("Transition Errors",
                          fontsize=PAPER_FS["row_label"], labelpad=4)
    axes[1, 0].set_ylabel("Decoder Uncertainty",
                          fontsize=PAPER_FS["row_label"], labelpad=4)

    # Note: the uncertainty panels share a common color scale (unc_vmax) so the
    # columns are directly comparable; the low->high scale is explained in the
    # figure caption rather than shown as a colorbar.

    path = os.path.join(plot_dir, f"heatmaps_seed_{seed}.png")
    # Save at natural size ("standard", not "tight") so the saved width equals
    # FIG_WIDTH_IN exactly and \includegraphics[width=0.70\linewidth] is scale 1.
    with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
        fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# Accuracy bar chart
# =============================================================================

def plot_accuracy_bars(env_name, seeds, plot_dir, dpi):
    data = {}
    for method in ["pretrained", "naive", "renew"]:
        vals = []
        for s in seeds:
            if _has_method(env_name, s, method):
                m = _load_metrics(env_name, s, method)
                vals.append(m["transition_acc"])
        data[method] = np.array(vals)

    n_seeds = len(seeds)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

    labels = ["Pretrained", "Naive", "RENEW"]
    colors = [C_PRETRAIN, C_NAIVE, C_RENEW]
    methods = ["pretrained", "naive", "renew"]

    means = [data[m].mean() * 100 if len(data[m]) else 0 for m in methods]
    cis = [1.96 * data[m].std() * 100 / np.sqrt(len(data[m]))
           if len(data[m]) > 1 else 0 for m in methods]

    bars = ax.bar(labels, means, yerr=cis, capsize=6, width=0.55,
                  color=colors, edgecolor="white", linewidth=1.5,
                  error_kw=dict(lw=1.8, capthick=1.8, color="#2d3436"))

    # Scatter individual seeds
    rng = np.random.default_rng(42)
    for i, m in enumerate(methods):
        if len(data[m]):
            jitter = rng.uniform(-0.12, 0.12, len(data[m]))
            ax.scatter(i + jitter, data[m] * 100, color="#2d3436",
                       s=18, alpha=0.45, zorder=5, linewidths=0)

    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.8,
                f"{mean:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Transition Accuracy (%)")
    ax.set_title(f"Final Accuracy ({n_seeds} seeds, 95% CI)", fontweight="bold")
    ax.set_ylim(80, 102)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()

    path = os.path.join(plot_dir, "accuracy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# Loss curves
# =============================================================================

def plot_loss_curves(env_name, seeds, plot_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)

    for ax, method, label, color in [
        (axes[0], "naive", "Naive (random)", C_NAIVE),
        (axes[1], "renew", "RENEW (ours)", C_RENEW),
    ]:
        all_losses = []
        for s in seeds:
            if _has_method(env_name, s, method):
                losses, _, _ = _load_curves(env_name, s, method)
                if losses:
                    all_losses.append(losses)

        if not all_losses:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        max_len = max(len(l) for l in all_losses)
        padded = np.full((len(all_losses), max_len), np.nan)
        for i, l in enumerate(all_losses):
            padded[i, :len(l)] = l

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        n_valid = np.sum(~np.isnan(padded), axis=0).clip(1)
        ci = 1.96 * std / np.sqrt(n_valid)
        steps = np.arange(max_len)

        ax.plot(steps, mean, color=color, linewidth=2.0)
        ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.18)
        ax.set_xlabel("Gradient Step")
        ax.set_ylabel("BT Loss")
        ax.set_title(label)

    fig.suptitle(f"Preference Loss ({len(seeds)} seeds, 95% CI)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(plot_dir, "loss_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# Accuracy curves — the hero figure
# =============================================================================

def plot_accuracy_curves(env_name, seeds, plot_dir, dpi):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    # Pretrained baseline as horizontal band
    pretrain_accs = []
    for s in seeds:
        if _has_method(env_name, s, "pretrained"):
            m = _load_metrics(env_name, s, "pretrained")
            pretrain_accs.append(m["transition_acc"])

    if pretrain_accs:
        pt_mean = np.mean(pretrain_accs) * 100
        pt_ci = 1.96 * np.std(pretrain_accs) * 100 / np.sqrt(len(pretrain_accs))
        ax.axhline(pt_mean, color=C_PRETRAIN, linestyle="--", linewidth=1.5,
                    label=f"Pretrained ({pt_mean:.1f}%)", zorder=1)
        ax.axhspan(pt_mean - pt_ci, pt_mean + pt_ci, color=C_PRETRAIN, alpha=0.10, zorder=0)

    for method, label, color in [
        ("naive", "Naive (random)", C_NAIVE),
        ("renew", "RENEW (ours)", C_RENEW),
    ]:
        all_steps, all_accs = [], []
        for s in seeds:
            if _has_method(env_name, s, method):
                _, accs, steps = _load_curves(env_name, s, method)
                if accs:
                    all_steps.append(steps)
                    all_accs.append(accs)

        if not all_accs:
            continue

        max_step = max(max(s) for s in all_steps)
        common = np.arange(0, max_step + 1)
        interp = np.full((len(all_accs), len(common)), np.nan)
        for i, (steps, accs) in enumerate(zip(all_steps, all_accs)):
            interp[i] = np.interp(common, steps, accs)

        mean = np.nanmean(interp, axis=0) * 100
        std = np.nanstd(interp, axis=0) * 100
        n_valid = np.sum(~np.isnan(interp), axis=0).clip(1)
        ci = 1.96 * std / np.sqrt(n_valid)

        final_mean = mean[-1]

        ax.plot(common, mean, color=color, linewidth=2.2,
                label=f"{label} ({final_mean:.1f}%)", zorder=3)
        ax.fill_between(common, mean - ci, mean + ci, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Transition Accuracy (%)")
    ax.set_title(f"Finetuning Accuracy ({len(seeds)} seeds, 95% CI)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    ax.set_ylim(82, 101)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()

    path = os.path.join(plot_dir, "accuracy_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# Summary CSV
# =============================================================================

def save_summary_csv(env_name, seeds, plot_dir):
    path = os.path.join(plot_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "pretrained_acc", "pretrained_n_wrong",
                         "naive_acc", "naive_n_wrong",
                         "renew_acc", "renew_n_wrong"])
        for s in seeds:
            row = [s]
            for method in ["pretrained", "naive", "renew"]:
                if _has_method(env_name, s, method):
                    m = _load_metrics(env_name, s, method)
                    row.extend([f"{m['transition_acc']:.4f}", m["n_wrong"]])
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    print(f"\n  Summary ({len(seeds)} seeds):")
    for name, method in [("Pretrained", "pretrained"), ("Naive", "naive"), ("RENEW", "renew")]:
        vals = []
        for s in seeds:
            if _has_method(env_name, s, method):
                m = _load_metrics(env_name, s, method)
                vals.append(m["transition_acc"])
        if vals:
            arr = np.array(vals)
            ci = 1.96 * arr.std() / np.sqrt(len(arr))
            print(f"    {name:12s}: {arr.mean()*100:.1f} ± {ci*100:.1f}% (95% CI)  "
                  f"[{arr.min()*100:.1f}–{arr.max()*100:.1f}]")

    print(f"  → {path}")


# =============================================================================
# Main
# =============================================================================

def main(args: PlotArgs) -> None:
    seeds = args.seeds if args.seeds else _find_seeds(args.env_name)

    if not seeds:
        print(f"No results found for '{args.env_name}'.")
        print(f"Run: bash renew/run_pretrain.sh && bash renew/run_finetune.sh")
        return

    plot_dir = os.path.join(RESULTS_DIR, args.env_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if args.heatmap_seed is not None:
        print(f"Regenerating heatmaps for seed {args.heatmap_seed}...")
        plot_seed_heatmaps(args.env_name, args.heatmap_seed, plot_dir,
                           args.maze_size, args.dpi)
        return

    print(f"Plotting {args.env_name} — seeds: {seeds}")
    print(f"Output: {plot_dir}/\n")

    plot_accuracy_bars(args.env_name, seeds, plot_dir, args.dpi)
    plot_loss_curves(args.env_name, seeds, plot_dir, args.dpi)
    plot_accuracy_curves(args.env_name, seeds, plot_dir, args.dpi)
    save_summary_csv(args.env_name, seeds, plot_dir)

    print(f"\n  Per-seed heatmaps:")
    for s in seeds:
        plot_seed_heatmaps(args.env_name, s, plot_dir, args.maze_size, args.dpi)

    print(f"\nDone! All plots in: {plot_dir}/")


if __name__ == "__main__":
    main(tyro.cli(PlotArgs))