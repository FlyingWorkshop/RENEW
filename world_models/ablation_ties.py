"""
world_models/ablation_ties.py
==============================
Ablation: does excluding ties from the Bradley-Terry loss help?

Two conditions per (env, seed):
    - exclude_ties=True  (default): ties masked out of BT loss (valid = 1 - ties)
    - exclude_ties=False:           ties included, preference label = 0.5

Both use RENEW with the same budget, batch size, K, and ensemble size.
The naive (batch-matched) baseline is included for each condition.

Requires a small patch to network.py and dlhf.py — see REQUIRED CHANGES
below.

Output structure:
    out/ablation_ties/{env}/
        renew_exclude/seed_{s}/result.npz
        renew_include/seed_{s}/result.npz
        naive_exclude/seed_{s}/result.npz
        naive_include/seed_{s}/result.npz
        ablation_ties.png
        ablation_ties_summary.txt

Usage:
    # Quick
    python world_models/ablation_ties.py --envs maze10 --seeds 0

    # Full
    python world_models/ablation_ties.py --envs maze10 sliding3 sokoban \
        --seeds 0 1 2 3 4

REQUIRED CHANGES
================
1. DLHFArgs (dlhf.py): add field
       exclude_ties: bool = True

2. build_network (network.py): pass it through
       exclude_ties=getattr(args, "exclude_ties", True),

3. LatentCNNNetwork (network.py): add field
       exclude_ties: bool = True

4. preferences_loss (network.py): replace the loss line
       OLD:  l_bt = jnp.mean(bce * (1.0 - ties))
       NEW:  if self.exclude_ties:
                 l_bt = jnp.mean(bce * (1.0 - ties))
             else:
                 # ties get preference = 0.5, already handled by BCE
                 l_bt = jnp.mean(bce)

5. k_candidate_preferences_loss (network.py): replace the loss computation
       OLD:  valid = 1.0 - ties
             ...
             l_bt  = jnp.sum(bce * valid) / n_valid
       NEW:  if self.exclude_ties:
                 valid = 1.0 - ties
                 n_valid = jnp.sum(valid) + 1e-8
                 l_bt = jnp.sum(bce * valid) / n_valid
             else:
                 # Include ties: preference = 0.5 for tied pairs
                 # Self-pair (winner vs itself) still excluded
                 k_idx = jnp.arange(K)[:, None]
                 self_mask = (k_idx == best_idx[None, :]).astype(jnp.float32)
                 valid = 1.0 - self_mask
                 # For tied (non-self) pairs, preference = 0.5
                 preferences = jnp.where(
                     ties * (1.0 - self_mask), 0.5, preferences)
                 bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
                 n_valid = jnp.sum(valid) + 1e-8
                 l_bt = jnp.sum(bce * valid) / n_valid

   NOTE: Since Flax modules can't use Python if/else at call time for
   traced values, and exclude_ties is a static bool set at construction,
   the if/else on self.exclude_ties is fine (it's resolved at trace time).
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
class AblationTiesArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    envs: List[str] = field(default_factory=lambda: [
        "maze10",
        # "maze15",
        "sliding3",
        "sokoban",
    ])

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

    # --- RENEW-specific ---
    num_candidates:   int   = 2
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

def _make_dlhf_args(aargs: AblationTiesArgs, env: str, seed: int,
                    steps: int, results_subdir: str,
                    renew: bool = False,
                    exclude_ties: bool = True,
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
        num_candidates=aargs.num_candidates,
        pool_multiplier=aargs.pool_multiplier,
        selection_temp=aargs.selection_temp,
        val_size=aargs.val_size,
        val_scramble=aargs.val_scramble,
        eval_every=aargs.eval_every,
        results_dir=results_subdir,
        rollout_steps=aargs.rollout_steps,
        exclude_ties=exclude_ties,
    )


def _init_ensemble(network, meta, aargs, seed, batch_size):
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

def run_naive_matched(aargs: AblationTiesArgs, env: str, seed: int,
                      out_dir: str, exclude_ties: bool):
    """Naive DLHF, batch-matched to RENEW's labels-per-step."""
    B = aargs.batch_size
    K = aargs.num_candidates
    labels_per_step = B * (K - 1)
    naive_bs = labels_per_step
    steps = aargs.preference_budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    tie_str = "exclude" if exclude_ties else "include"
    tag = f"naive_{tie_str}"

    dlhf_args = _make_dlhf_args(
        aargs, env, seed, steps, out_dir, renew=False,
        exclude_ties=exclude_ties, batch_size_override=naive_bs)
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
          f"B={naive_bs} | ties={'excluded' if exclude_ties else 'included'} | "
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
        env=env, seed=seed, method=tag,
        exclude_ties=exclude_ties,
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


def run_renew(aargs: AblationTiesArgs, env: str, seed: int,
              out_dir: str, exclude_ties: bool):
    """RENEW with ties either excluded or included."""
    B = aargs.batch_size
    K = aargs.num_candidates
    labels_per_step = B * (K - 1)
    steps = aargs.preference_budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    tie_str = "exclude" if exclude_ties else "include"
    tag = f"renew_{tie_str}"

    dlhf_args = _make_dlhf_args(
        aargs, env, seed, steps, out_dir, renew=True,
        exclude_ties=exclude_ties)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = aargs.ensemble_size

    stacked_params, stacked_opt_state = _init_ensemble(
        network, meta, aargs, seed, B)

    val_set   = collect_val_set(meta, aargs.val_size, aargs.val_scramble, seed)
    train_rng = jax.random.PRNGKey(seed + 123)

    print(f"\n{'='*60}")
    print(f"  {tag}: {env} | seed {seed} | {steps} steps | "
          f"B={B} | K={K} | ties={'excluded' if exclude_ties else 'included'} | "
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
        env=env, seed=seed, method=tag,
        exclude_ties=exclude_ties,
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

METHODS_ORDER = ["naive_exclude", "naive_include", "renew_exclude", "renew_include"]

DISPLAY_NAMES = {
    "naive_exclude":  "Naive (exclude ties)",
    "naive_include":  "Naive (include ties)",
    "renew_exclude":  "RENEW (exclude ties)",
    "renew_include":  "RENEW (include ties)",
}

METHOD_COLORS = {
    "naive_exclude":  "tab:blue",
    "naive_include":  "tab:cyan",
    "renew_exclude":  "tab:red",
    "renew_include":  "tab:orange",
}

METHOD_LINESTYLES = {
    "naive_exclude":  "--",
    "naive_include":  ":",
    "renew_exclude":  "-",
    "renew_include":  "-.",
}


def save_ablation_plot(all_results, envs, aargs, base_dir):
    n_envs = len(envs)
    cols = min(n_envs, 4)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), dpi=130,
                             squeeze=False)

    present_methods = [m for m in METHODS_ORDER
                       if any(r["method"] == m for r in all_results)]

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]

        for method in present_methods:
            method_results = [
                r for r in all_results
                if r["env"] == env and r["method"] == method
            ]
            if not method_results:
                continue

            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]

            for r in method_results:
                ax.plot(r["labels"], r["val_l1s"],
                        linewidth=0.5, alpha=0.25, color=color, linestyle=ls)

            ref_labels = method_results[0]["labels"]
            aligned = [r["val_l1s"] for r in method_results
                       if len(r["val_l1s"]) == len(ref_labels)]
            if aligned:
                arr = np.array(aligned)
                n = len(aligned)
                mean = arr.mean(0)
                ax.plot(ref_labels, mean, linewidth=2, color=color,
                        linestyle=ls,
                        label=f"{DISPLAY_NAMES[method]} (n={n})")
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
        f"Tie Exclusion Ablation — {aargs.preference_budget:,} labels | "
        f"E={aargs.ensemble_size} K={aargs.num_candidates} | "
        f"{len(aargs.seeds)} seeds",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    path = os.path.join(base_dir, "ablation_ties.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Ablation plot -> {path}")


def save_ablation_table(all_results, envs, aargs, base_dir):
    present_methods = [m for m in METHODS_ORDER
                       if any(r["method"] == m for r in all_results)]

    print(f"\n{'='*80}")
    print(f"  TIE EXCLUSION ABLATION")
    print(f"  Budget: {aargs.preference_budget:,} labels | "
          f"E={aargs.ensemble_size} K={aargs.num_candidates}")
    print(f"{'='*80}")

    method_width = 24
    header_parts = [f"{'Env':<12s}"]
    for m in present_methods:
        header_parts.append(f"{DISPLAY_NAMES[m]:>{method_width}s}")
    header = "  ".join(header_parts)
    sep = "  ".join(["-" * 12] + ["-" * method_width] * len(present_methods))
    print(f"  {header}")
    print(f"  {sep}")

    lines = [header, sep]
    for env in envs:
        parts = [f"{env:<12s}"]
        for method in present_methods:
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

    path = os.path.join(base_dir, "ablation_ties_summary.txt")
    with open(path, "w") as f:
        f.write("Tie Exclusion Ablation Results\n")
        f.write(f"Budget: {aargs.preference_budget:,} labels\n")
        f.write(f"E={aargs.ensemble_size}, K={aargs.num_candidates}\n\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Main
# =============================================================================

def main(aargs: AblationTiesArgs):
    base_dir = aargs.results_dir or os.path.join(OUT_DIR, "ablation_ties")
    os.makedirs(base_dir, exist_ok=True)

    B = aargs.batch_size
    K = aargs.num_candidates
    labels_per_step = B * (K - 1)
    steps = aargs.preference_budget // labels_per_step

    print(f"\n{'='*60}")
    print(f"  TIE EXCLUSION ABLATION")
    print(f"{'='*60}")
    print(f"  Envs:           {aargs.envs}")
    print(f"  Seeds:          {aargs.seeds}")
    print(f"  Label budget:   {aargs.preference_budget:,}")
    print(f"  Labels/step:    {labels_per_step} (matched)")
    print(f"  Steps:          {steps}")
    print(f"  K:              {K}")
    print(f"  Ensemble:       E={aargs.ensemble_size}")
    print(f"  Conditions:     exclude_ties x {{True, False}} "
          f"x {{naive, renew}}")
    print(f"  Overwrite:      {aargs.overwrite}")
    print(f"  Output:         {base_dir}")
    print(f"{'='*60}")

    all_results = []

    for env in aargs.envs:
        for seed in aargs.seeds:
            for exclude_ties in [True, False]:
                tie_str = "exclude" if exclude_ties else "include"

                # --- RENEW ---
                renew_tag = f"renew_{tie_str}"
                renew_dir = os.path.join(
                    base_dir, env, f"seed_{seed}", renew_tag)
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
                        result = run_renew(
                            aargs, env, seed, renew_dir, exclude_ties)
                    except Exception as e:
                        print(f"\n  ERROR: {renew_tag} {env} seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue
                all_results.append(result)

                # --- Naive (batch-matched) ---
                naive_tag = f"naive_{tie_str}"
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
                            aargs, env, seed, naive_dir, exclude_ties)
                    except Exception as e:
                        print(f"\n  ERROR: {naive_tag} {env} seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue
                all_results.append(result)

    if not all_results:
        print("No successful runs. Exiting.")
        return

    np.savez(
        os.path.join(base_dir, "all_results.npz"),
        envs=aargs.envs, seeds=aargs.seeds,
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
    main(tyro.cli(AblationTiesArgs))