"""
plot_gymnax_comparison.py
=========================
Paper-quality 2×2 figure: Naive vs RENEW on classic control.

    Rows:    MountainCar, Acrobot (shared y-axis per row)
    Columns: K=2, K=8

Usage:
    python plot_gymnax_comparison.py
    python plot_gymnax_comparison.py --k2-dir out/compare_gymnax_pretrained_k2 \
                                     --k8-dir out/compare_gymnax_pretrained_k8
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


@dataclass
class PlotArgs:
    k2_dir: str = "out/compare_gymnax_pretrained_k2"
    """Directory containing K=2 results."""
    k8_dir: str = "out/compare_gymnax_pretrained_k8"
    """Directory containing K=8 results."""
    envs: tuple = ("MountainCar-v0", "Acrobot-v1")
    seeds: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    output: str = "out/gymnax_comparison.png"


def _load_results(base_dir, env, seeds):
    results = {"naive": [], "renew": []}
    for seed in seeds:
        for method in ["naive", "renew"]:
            npz_path = os.path.join(
                base_dir, env, f"seed_{seed}", method, "result.npz")
            if not os.path.exists(npz_path):
                print(f"  Warning: missing {npz_path}")
                continue
            d = dict(np.load(npz_path, allow_pickle=True))
            results[method].append(dict(
                labels=d["labels"],
                mses=d["mses"],
                final_mse=float(d["final_mse"]),
            ))
    return results


def _plot_method(ax, results_list, color, label):
    if not results_list:
        return
    ref_labels = results_list[0]["labels"]
    aligned = [r["mses"] for r in results_list
               if len(r["mses"]) == len(ref_labels)]
    if aligned:
        arr = np.array(aligned)
        n = len(aligned)
        mean = arr.mean(0)
        ax.plot(ref_labels, mean, linewidth=1.5, color=color, label=label)
        if n > 1:
            ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(n)
            ax.fill_between(ref_labels, mean - ci, mean + ci,
                            alpha=0.18, color=color, linewidth=0)


# ── Style ────────────────────────────────────────────────────────────
ENV_NAMES = {
    "MountainCar-v0": "MountainCar",
    "Acrobot-v1": "Acrobot",
}

COLORS = {
    "naive": "#4C72B0",   # muted blue
    "renew": "#C44E52",   # muted red
}

LABELS = {
    "naive": "Naive",
    "renew": "RENEW",
}


def main(args: PlotArgs):
    # ── Global rc overrides (kept local to avoid side-effects) ───────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    k_configs = [
        (args.k2_dir, 2),
        (args.k8_dir, 8),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(5.5, 3.6), dpi=300,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )

    # ── Plot each panel ──────────────────────────────────────────────
    for row, env in enumerate(args.envs):
        for col, (base_dir, K) in enumerate(k_configs):
            ax = axes[row][col]
            results = _load_results(base_dir, env, args.seeds)

            for method in ["naive", "renew"]:
                _plot_method(ax, results[method],
                             COLORS[method], LABELS[method])

            # ── Tick formatting ──────────────────────────────────────
            ax.tick_params(labelsize=7, pad=2)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"
                                      if x > 0 else "0"))

            # Column titles (top row only)
            if row == 0:
                ax.set_title(f"$K = {K}$", fontsize=9, pad=4)

            # Hide right-column y tick labels (shared y-axis)
            if col > 0:
                ax.set_yticklabels([])

            # Hide x tick labels on top row
            if row == 0:
                ax.set_xticklabels([])

            # ── Spine styling ────────────────────────────────────────
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Print summary
            for method in ["naive", "renew"]:
                if results[method]:
                    finals = [r["final_mse"] for r in results[method]]
                    m = np.mean(finals)
                    n = len(finals)
                    ci = (1.96 * np.std(finals, ddof=1) / np.sqrt(n)
                          if n > 1 else 0.0)
                    print(f"  {env} K={K} {method}: "
                          f"{m:.6f} +/- {ci:.6f} ({n} seeds)")

        # ── Share y-axis limits across K values for this env ─────────
        ylims = [axes[row][c].get_ylim() for c in range(2)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for c in range(2):
            axes[row][c].set_ylim(ymin, ymax)

    # ── Row labels as right-side text annotations ────────────────────
    for row, env in enumerate(args.envs):
        ax = axes[row][1]
        ax.annotate(
            ENV_NAMES.get(env, env),
            xy=(1.0, 0.5), xycoords="axes fraction",
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, fontstyle="italic",
            ha="left", va="center", rotation=-90,
        )

    # ── Shared axis labels ───────────────────────────────────────────
    fig.text(0.02, 0.5, "Val MSE", va="center", rotation="vertical",
             fontsize=9)
    fig.text(0.5, 0.04, "Preference Labels", ha="center", fontsize=9)

    # ── Shared legend ────────────────────────────────────────────────
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=8,
        frameon=False, columnspacing=1.5,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.subplots_adjust(left=0.10, right=0.93, top=0.92, bottom=0.14)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main(tyro.cli(PlotArgs))