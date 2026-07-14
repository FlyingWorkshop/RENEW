"""
world_models/ablation_k.py
===========================
Ablation on K (number of candidate rollouts per start state in RENEW).

For each value of K we run RENEW with the same total preference-label
budget.  The naive baseline is always batch-matched: for a given K,
naive uses batch size B*(K-1) so it consumes the same labels-per-step
as RENEW.  This isolates active selection as the only variable.

Output structure:
    out/ablation_k/{env}/
        K{k}/seed_{s}/result.npz
        naive_K{k}/seed_{s}/result.npz
        ablation_k.png
        ablation_k_summary.txt

Usage:
    # Quick
    python world_models/ablation_k.py --envs maze10 --seeds 0 --k-values 2 4

    # Full
    python world_models/ablation_k.py --envs maze10 maze15 sokoban \
        --seeds 0 1 2 3 4 --k-values 2 3 4 6 8
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import collect_val_set, make_preferences_batch_fn
from network import LatentCNNNetwork, build_network, init_params
from dlhf import (
    DLHFArgs,
    make_meta,
    train_dlhf,
    train_dlhf_renew,
    save_rollout,
    save_curves,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


# =============================================================================
# Args
# =============================================================================

@dataclass
class AblationKArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    envs: List[str] = field(default_factory=lambda: [
        "maze10", 
        # "maze15", 
        "sliding3", 
        "sokoban",
    ])
    k_values: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    """Values of K (num_candidates) to sweep over."""

    overwrite: bool = False

    # --- Budget ---
    preference_budget: int = 10_000

    # --- Architecture (shared) ---
    embed_dim:       int = 128
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 4
    dec_layers:      int = 3

    # --- Training (shared) ---
    lr:             float = 3e-4
    batch_size:     int   = 64
    horizon:        int   = 1
    beta_bt:        float = 1.0
    scramble_steps: int   = 0
    context_len:    int   = 10

    # --- Ensemble (shared) ---
    ensemble_size: int = 3

    # --- RENEW-specific (non-K) ---
    pool_multiplier:  int   = 4
    selection_temp:   float = 1.0

    # --- Validation ---
    val_size:     int = 500
    val_scramble: int = 30
    eval_every:   int = 10

    # --- Rollout visualisation ---
    rollout_steps: int = 30

    # --- Output ---
    results_dir: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================

def _make_dlhf_args(aargs: AblationKArgs, env: str, seed: int,
                    steps: int, results_subdir: str,
                    renew: bool = False, K: int = 2,
                    batch_size_override: Optional[int] = None) -> DLHFArgs:
    bs = batch_size_override if batch_size_override is not None else aargs.batch_size
    return DLHFArgs(
        seed=seed,
        env=env,
        embed_dim=aargs.embed_dim,
        latent_channels=aargs.latent_channels,
        enc_layers=aargs.enc_layers,
        dyn_layers=aargs.dyn_layers,
        dec_layers=aargs.dec_layers,
        lr=aargs.lr,
        steps=steps,
        batch_size=bs,
        horizon=aargs.horizon,
        beta_bt=aargs.beta_bt,
        scramble_steps=aargs.scramble_steps,
        context_len=aargs.context_len,
        ensemble_size=aargs.ensemble_size,
        renew=renew,
        num_candidates=K,
        pool_multiplier=aargs.pool_multiplier,
        selection_temp=aargs.selection_temp,
        val_size=aargs.val_size,
        val_scramble=aargs.val_scramble,
        eval_every=aargs.eval_every,
        results_dir=results_subdir,
        rollout_steps=aargs.rollout_steps,
    )


def _init_ensemble(network, meta, aargs, seed, batch_size):
    """Initialise ensemble params + optimiser states."""
    M = aargs.ensemble_size
    opt = optax.adam(aargs.lr)
    all_params, all_opts = [], []
    for m in range(M):
        rng = jax.random.PRNGKey(seed + m * 1000)
        p = init_params(network, meta, batch_size, aargs.context_len, rng)
        all_params.append(p)
        all_opts.append(opt.init(p))
    stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opts)
    return stacked_params, stacked_opt_state


# =============================================================================
# Runners
# =============================================================================

def run_naive_matched(aargs: AblationKArgs, env: str, seed: int,
                      out_dir: str, K: int):
    """Naive DLHF with batch size matched to RENEW K's labels-per-step."""
    B = aargs.batch_size
    labels_per_step = B * (K - 1)
    naive_bs = labels_per_step  # one label per start state
    steps = aargs.preference_budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    tag = f"naive_K{K}"
    dlhf_args = _make_dlhf_args(
        aargs, env, seed, steps, out_dir, renew=False,
        batch_size_override=naive_bs)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = aargs.ensemble_size

    stacked_params, stacked_opt_state = _init_ensemble(
        network, meta, aargs, seed, naive_bs)

    val_set   = collect_val_set(meta, aargs.val_size, aargs.val_scramble, seed)
    gen_batch = make_preferences_batch_fn(
        meta, naive_bs, aargs.scramble_steps, aargs.horizon)
    train_rng = jax.random.PRNGKey(seed + 123)

    print(f"\n{'='*60}")
    print(f"  {tag}: {env} | seed {seed} | {steps} steps | "
          f"B={naive_bs} | {labels_per_step} labels/step | "
          f"{aargs.preference_budget:,} labels | E={M}")
    print(f"{'='*60}")

    t0 = time.time()
    stacked_params, _, losses, val_l1s, eval_steps = train_dlhf(
        network, stacked_params, stacked_opt_state, gen_batch,
        val_set, dlhf_args, train_rng, ensemble_size=M,
    )
    wall_time = time.time() - t0

    labels_at_eval = [s * labels_per_step for s in eval_steps]

    save_curves(losses, val_l1s, eval_steps,
                os.path.join(out_dir, "curves.png"))

    result = dict(
        env=env, seed=seed, method=tag, K=K,
        eval_steps=np.array(eval_steps),
        labels=np.array(labels_at_eval),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


def run_renew_k(aargs: AblationKArgs, env: str, seed: int, out_dir: str,
                K: int):
    """RENEW with a specific value of K."""
    B = aargs.batch_size
    labels_per_step = B * (K - 1)
    steps = aargs.preference_budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    dlhf_args = _make_dlhf_args(
        aargs, env, seed, steps, out_dir, renew=True, K=K)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = aargs.ensemble_size

    stacked_params, stacked_opt_state = _init_ensemble(
        network, meta, aargs, seed, B)

    val_set   = collect_val_set(meta, aargs.val_size, aargs.val_scramble, seed)
    train_rng = jax.random.PRNGKey(seed + 123)

    tag = f"renew_K{K}"
    print(f"\n{'='*60}")
    print(f"  {tag}: {env} | seed {seed} | {steps} steps | "
          f"B={B} | {labels_per_step} labels/step | "
          f"{aargs.preference_budget:,} labels | E={M}")
    print(f"{'='*60}")

    t0 = time.time()
    stacked_params, _, losses, val_l1s, eval_steps, extra = train_dlhf_renew(
        network, stacked_params, stacked_opt_state,
        meta, val_set, dlhf_args, train_rng, ensemble_size=M,
    )
    wall_time = time.time() - t0

    labels_at_eval = [s * labels_per_step for s in eval_steps]

    save_curves(losses, val_l1s, eval_steps,
                os.path.join(out_dir, "curves.png"),
                extra_metrics=extra)

    result = dict(
        env=env, seed=seed, method=tag, K=K,
        eval_steps=np.array(eval_steps),
        labels=np.array(labels_at_eval),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# Plotting
# =============================================================================

def _get_colormap(methods):
    """Assign distinct colours to each method string."""
    cmap = plt.cm.tab10
    return {m: cmap(i / max(len(methods), 1)) for i, m in enumerate(methods)}


def _label(m):
    if m.startswith("naive_K"):
        k = m.split("K")[-1]
        return f"Naive (matched, K={k})"
    if m.startswith("renew_K"):
        k = m.split("K")[-1]
        return f"RENEW K={k}"
    return m


def save_ablation_plot(all_results, envs, aargs, base_dir):
    n_envs = len(envs)
    cols = min(n_envs, 4)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), dpi=130,
                             squeeze=False)

    all_methods = sorted(set(r["method"] for r in all_results))
    colors = _get_colormap(all_methods)

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]

        for method in all_methods:
            method_results = [
                r for r in all_results
                if r["env"] == env and r["method"] == method
            ]
            if not method_results:
                continue

            color = colors[method]
            linestyle = "--" if method.startswith("naive") else "-"

            # Individual seed traces
            for r in method_results:
                ax.plot(r["labels"], r["val_l1s"],
                        linewidth=0.5, alpha=0.25, color=color,
                        linestyle=linestyle)

            # Mean + CI
            ref_labels = method_results[0]["labels"]
            aligned = [r["val_l1s"] for r in method_results
                       if len(r["val_l1s"]) == len(ref_labels)]
            if aligned:
                arr = np.array(aligned)
                n = len(aligned)
                mean = arr.mean(0)
                ax.plot(ref_labels, mean, linewidth=2, color=color,
                        linestyle=linestyle,
                        label=f"{_label(method)} (n={n})")
                if n > 1:
                    ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(n)
                    ax.fill_between(ref_labels, mean - ci, mean + ci,
                                    alpha=0.12, color=color)

        ax.set_xlabel("Preference Labels")
        ax.set_ylabel("Val L1")
        ax.set_title(env, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")

    for idx in range(n_envs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"K-Ablation — {aargs.preference_budget:,} labels | "
        f"E={aargs.ensemble_size} | {len(aargs.seeds)} seeds\n"
        f"K ∈ {{{', '.join(str(k) for k in aargs.k_values)}}} | "
        f"Naive always batch-matched",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    path = os.path.join(base_dir, "ablation_k.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Ablation plot -> {path}")


def save_ablation_table(all_results, envs, aargs, base_dir):
    all_methods = sorted(set(r["method"] for r in all_results))

    print(f"\n{'='*80}")
    print(f"  K-ABLATION RESULTS (naive always batch-matched)")
    print(f"  Budget: {aargs.preference_budget:,} labels | E={aargs.ensemble_size}")
    print(f"{'='*80}")

    # Header
    method_width = 22
    header_parts = [f"{'Env':<12s}"]
    for m in all_methods:
        header_parts.append(f"{_label(m):>{method_width}s}")
    header = "  ".join(header_parts)
    sep = "  ".join(["-" * 12] + ["-" * method_width] * len(all_methods))
    print(f"  {header}")
    print(f"  {sep}")

    lines = [header, sep]
    for env in envs:
        parts = [f"{env:<12s}"]
        for method in all_methods:
            results = [r for r in all_results
                       if r["env"] == env and r["method"] == method]
            if not results:
                parts.append(f"{'---':>{method_width}s}")
                continue
            finals = [r["final_val_l1"] for r in results]
            m_val = np.mean(finals)
            n = len(finals)
            ci = 1.96 * np.std(finals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            parts.append(f"{m_val:.4f} ± {ci:.4f}".rjust(method_width))
        line = "  ".join(parts)
        print(f"  {line}")
        lines.append(line)

    path = os.path.join(base_dir, "ablation_k_summary.txt")
    with open(path, "w") as f:
        f.write("K-Ablation Results (naive always batch-matched)\n")
        f.write(f"Budget: {aargs.preference_budget:,} labels\n")
        f.write(f"E={aargs.ensemble_size}\n")
        f.write(f"K values: {aargs.k_values}\n\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Main
# =============================================================================

def main(aargs: AblationKArgs):
    base_dir = aargs.results_dir or os.path.join(OUT_DIR, "ablation_k")
    os.makedirs(base_dir, exist_ok=True)

    B = aargs.batch_size

    print(f"\n{'='*60}")
    print(f"  K-ABLATION (naive always batch-matched)")
    print(f"{'='*60}")
    print(f"  Envs:           {aargs.envs}")
    print(f"  Seeds:          {aargs.seeds}")
    print(f"  K values:       {aargs.k_values}")
    print(f"  Label budget:   {aargs.preference_budget:,}")
    print(f"  Base batch:     B={B}")
    print(f"  Ensemble:       E={aargs.ensemble_size}")
    print(f"  Overwrite:      {aargs.overwrite}")
    print(f"  Output:         {base_dir}")

    for K in aargs.k_values:
        labels_per_step = B * (K - 1)
        naive_bs = labels_per_step
        steps = aargs.preference_budget // labels_per_step
        print(f"  K={K}: {steps} steps, {labels_per_step} labels/step, "
              f"naive B={naive_bs}, RENEW B={B}")

    print(f"{'='*60}")

    all_results = []

    for env in aargs.envs:
        for seed in aargs.seeds:
            for K in aargs.k_values:
                # RENEW K
                renew_tag = f"renew_K{K}"
                renew_dir = os.path.join(
                    base_dir, env, f"seed_{seed}", f"K{K}")
                npz_path = os.path.join(renew_dir, "result.npz")

                if os.path.exists(npz_path) and not aargs.overwrite:
                    print(f"\n  {renew_tag} {env} seed {seed}: "
                          f"loading {npz_path}")
                    d = dict(np.load(npz_path, allow_pickle=True))
                    result = dict(
                        env=str(d.get("env", env)),
                        seed=int(d.get("seed", seed)),
                        method=str(d.get("method", renew_tag)),
                        eval_steps=d["eval_steps"],
                        labels=d["labels"],
                        val_l1s=d["val_l1s"],
                        final_val_l1=float(d["final_val_l1"]),
                    )
                else:
                    try:
                        result = run_renew_k(aargs, env, seed, renew_dir, K)
                    except Exception as e:
                        print(f"\n  ERROR: {renew_tag} {env} seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue
                all_results.append(result)

                # Naive matched to this K
                naive_tag = f"naive_K{K}"
                naive_dir = os.path.join(
                    base_dir, env, f"seed_{seed}", naive_tag)
                npz_path = os.path.join(naive_dir, "result.npz")

                if os.path.exists(npz_path) and not aargs.overwrite:
                    print(f"\n  {naive_tag} {env} seed {seed}: "
                          f"loading {npz_path}")
                    d = dict(np.load(npz_path, allow_pickle=True))
                    result = dict(
                        env=str(d.get("env", env)),
                        seed=int(d.get("seed", seed)),
                        method=str(d.get("method", naive_tag)),
                        eval_steps=d["eval_steps"],
                        labels=d["labels"],
                        val_l1s=d["val_l1s"],
                        final_val_l1=float(d["final_val_l1"]),
                    )
                else:
                    try:
                        result = run_naive_matched(
                            aargs, env, seed, naive_dir, K)
                    except Exception as e:
                        print(f"\n  ERROR: {naive_tag} {env} seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue
                all_results.append(result)

    if not all_results:
        print("No successful runs. Exiting.")
        return

    # ---- Aggregate outputs ----
    np.savez(
        os.path.join(base_dir, "all_results.npz"),
        envs=aargs.envs, seeds=aargs.seeds, k_values=aargs.k_values,
        preference_budget=aargs.preference_budget,
        run_envs=np.array([r["env"] for r in all_results]),
        run_seeds=np.array([r["seed"] for r in all_results]),
        run_methods=np.array([r["method"] for r in all_results]),
        run_final_val_l1=np.array([r["final_val_l1"] for r in all_results]),
    )

    save_ablation_plot(all_results, aargs.envs, aargs, base_dir)
    save_ablation_table(all_results, aargs.envs, aargs, base_dir)
    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(AblationKArgs))