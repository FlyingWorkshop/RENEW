"""
plot_jumanji_comparison.py
==========================
Paper-quality 1x3 figure: Naive vs RENEW on Jumanji environments.

    Columns: Sliding3, Sliding5, Sokoban

Reads cached result.npz files from compare_jumanji output directory.

Usage:
    python plot_jumanji_comparison.py
    python plot_jumanji_comparison.py --results-dir out/compare_dlhf_renew_pretrained
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


@dataclass
class PlotArgs:
    results_dir: str = "out/compare_dlhf_renew_pretrained"
    """Directory containing compare_jumanji results."""
    envs: List[str] = field(default_factory=lambda: [
        "sliding3", "sliding5", "sokoban",
    ])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output: str = "out/jumanji_comparison.png"


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
                val_l1s=d["val_l1s"],
                final_val_l1=float(d["final_val_l1"]),
            ))
    return results


def _plot_method(ax, results_list, color, label):
    if not results_list:
        return
    ref_labels = results_list[0]["labels"]
    aligned = [r["val_l1s"] for r in results_list
               if len(r["val_l1s"]) == len(ref_labels)]
    if aligned:
        arr = np.array(aligned)
        n = len(aligned)
        mean = arr.mean(0)
        ax.plot(ref_labels, mean, linewidth=0.75, color=color, label=label)
        if n > 1:
            ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(n)
            ax.fill_between(ref_labels, mean - ci, mean + ci,
                            alpha=0.18, color=color, linewidth=0)


# ── Style ────────────────────────────────────────────────────────────
ENV_NAMES = {
    "sliding3": "Sliding $3{\\times}3$",
    "sliding5": "Sliding $5{\\times}5$",
    "sokoban": "Sokoban",
}

COLORS = {
    "naive": "#4C72B0",   # muted blue
    "renew": "#C44E52",   # muted red
}

LABELS = {
    "naive": "Naive",
    "renew": "RENEW",
}

# ── Paper geometry (RLC / rlj.sty) ───────────────────────────────────
# Author the figure at its exact printed width so in-figure point sizes map
# 1:1 to points in the PDF. rlj.sty sets \textwidth = 5.5in and the paper
# includes this figure at width=0.75\textwidth, so the print width is:
FIG_WIDTH_IN = 0.75 * 5.5          # = 4.125 in
# Times-like serif to match the paper body (rlj.sty loads `times`).
_SERIF_STACK = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                "Tinos", "DejaVu Serif"]
# Point sizes are exactly what appears in the PDF (scale 1.0), so ~7-9pt reads
# as the same font as the 10pt body, slightly smaller.
FS = {"title": 9.0, "axlabel": 8.5, "tick": 7.0, "legend": 8.0}


def main(args: PlotArgs):
    # ── Global rc overrides ──────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF_STACK,
        "mathtext.fontset": "stix",   # Times-like math (bundled with mpl)
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    n_envs = len(args.envs)
    panel_w = FIG_WIDTH_IN / n_envs
    fig_h = panel_w * 0.95 + 0.62      # panel + title + x-label/legend strip
    fig, axes = plt.subplots(
        1, n_envs, figsize=(FIG_WIDTH_IN, fig_h), dpi=400,
        gridspec_kw={"wspace": 0.30},
    )
    if n_envs == 1:
        axes = [axes]

    # ── Plot each panel ──────────────────────────────────────────────
    for col, env in enumerate(args.envs):
        ax = axes[col]
        results = _load_results(args.results_dir, env, args.seeds)

        for method in ["naive", "renew"]:
            _plot_method(ax, results[method],
                         COLORS[method], LABELS[method])

        # ── Tick formatting ──────────────────────────────────────────
        ax.tick_params(labelsize=FS["tick"], pad=2)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"
                                  if x > 0 else "0"))

        # Column titles
        ax.set_title(ENV_NAMES.get(env, env), fontsize=FS["title"], pad=4)
        if col == 0:
            ax.set_ylabel("Val $\\ell_1$", fontsize=FS["axlabel"])

        # ── Spine styling ────────────────────────────────────────────
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Print summary
        for method in ["naive", "renew"]:
            if results[method]:
                finals = [r["final_val_l1"] for r in results[method]]
                m = np.mean(finals)
                n = len(finals)
                ci = (1.96 * np.std(finals, ddof=1) / np.sqrt(n)
                      if n > 1 else 0.0)
                print(f"  {env} {method}: "
                      f"{m:.4f} +/- {ci:.4f} ({n} seeds)")

    # ── Shared x-axis label + legend (stacked in the bottom strip) ───
    fig.text(0.5, 0.135, "Preference Labels", ha="center",
             fontsize=FS["axlabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=FS["legend"],
        frameon=False, columnspacing=1.5, handlelength=1.6,
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.30)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Save at natural size ("standard", not "tight") so the saved width equals
    # FIG_WIDTH_IN exactly and \includegraphics[width=0.75\textwidth] is scale 1.
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(args.output)
    plt.close(fig)
    print(f"\n  Saved -> {args.output}")


if __name__ == "__main__":
    main(tyro.cli(PlotArgs))