"""
world_models/from_scratch.py
============================
Section 4.1: Learning world models from preferences (from scratch).

Compares two training regimes:
    - Supervised:  1000 random transitions (reconstruction loss)
    - DLHF:        1,000,000 preference labels (Bradley-Terry loss)

The asymmetry is the point: demonstrations are expensive and limited,
preferences are cheap and scalable. The question is whether a large
budget of cheap binary comparisons can match a small budget of direct
transition supervision.

Outputs per environment/seed:
    out/from_scratch/{env}/seed_{seed}/
        dlhf/curves.png         -- loss + val_l1 curves
        dlhf/rollout.gif        -- GT vs Model side-by-side
        supervised/curves.png   -- loss + val_l1 curves
        supervised/rollout.gif  -- GT vs Model side-by-side

Aggregated outputs:
    out/from_scratch/
        summary.png         -- multi-env val_l1 curves (DLHF vs supervised)
        summary.txt         -- results table
        summary.npz         -- raw data

Usage:
    # Quick sanity check
    python world_models/from_scratch.py --envs maze10 --seeds 0 \
        --supervised-transitions 100 --preference-budget 10000

    # Full run (3 seeds, default envs)
    python world_models/from_scratch.py --seeds 0 1 2

    # Custom budgets
    python world_models/from_scratch.py --supervised-transitions 500 \
        --preference-budget 500000
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

from env import collect_val_set, collect_offline_dataset
from network import LatentCNNNetwork, build_network, init_params, save_params
from dlhf import (
    DLHFArgs,
    make_meta,
    train_dlhf,
    make_preferences_batch_fn,
    save_rollout,
    save_curves,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


# =============================================================================
# Args
# =============================================================================

@dataclass
class FromScratchArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    """Seeds to run. 3 is enough for a sanity check."""

    envs: List[str] = field(default_factory=lambda: [
        # "maze5", 
        "maze10", 
        # "maze15", 
        # "maze20",
        "sliding5",
        "sokoban",
        "2048",
        # "connector8_3",
        # "pacman",
    ])
    """Environments to evaluate."""

    # --- Budgets ---
    supervised_transitions: int = 1_000
    """Number of random transitions for the supervised baseline.
    Represents the 'expensive demonstrations' regime."""

    preference_budget: int = 1_000_000
    """Number of preference labels for DLHF.
    Represents the 'cheap comparisons' regime."""

    supervised_steps: int = 2_500
    """Training steps for supervised baseline. Trains on the fixed dataset
    with random minibatch sampling (with replacement)."""

    # --- Architecture ---
    embed_dim:       int = 128
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 4
    dec_layers:      int = 3

    # --- Training ---
    lr:             float = 3e-4
    batch_size:     int   = 64
    horizon:        int   = 1
    beta_bt:        float = 1.0
    scramble_steps: int   = 0
    context_len:    int   = 10

    # --- Ensemble (single model for from-scratch) ---
    ensemble_size: int = 1

    # --- Validation ---
    val_size:     int = 500
    val_scramble: int = 30
    eval_every:   int = 100

    # --- Rollout visualisation ---
    rollout_steps: int = 30

    # --- Output ---
    results_dir: Optional[str] = None

    overwrite: bool = False
    """If False (default), skip (env, seed, method) combos whose
    result.npz already exists. Pass --overwrite to force re-run."""


# =============================================================================
# Helpers
# =============================================================================

def _result_exists(results_dir: str) -> Optional[dict]:
    """If result.npz exists in results_dir, load and return it; else None."""
    path = os.path.join(results_dir, "result.npz")
    if os.path.exists(path):
        data = dict(np.load(path, allow_pickle=True))
        # Convert numpy scalars back to python types
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 0:
                data[k] = v.item()
        return data
    return None


def _make_dlhf_args(fargs: FromScratchArgs, env: str, seed: int,
                    steps: int, results_subdir: str) -> DLHFArgs:
    """Build a DLHFArgs for a single (env, seed) run."""
    return DLHFArgs(
        seed=seed,
        env=env,
        embed_dim=fargs.embed_dim,
        latent_channels=fargs.latent_channels,
        enc_layers=fargs.enc_layers,
        dyn_layers=fargs.dyn_layers,
        dec_layers=fargs.dec_layers,
        lr=fargs.lr,
        steps=steps,
        batch_size=fargs.batch_size,
        horizon=fargs.horizon,
        beta_bt=fargs.beta_bt,
        scramble_steps=fargs.scramble_steps,
        context_len=fargs.context_len,
        ensemble_size=fargs.ensemble_size,
        renew=False,
        val_size=fargs.val_size,
        val_scramble=fargs.val_scramble,
        eval_every=fargs.eval_every,
        results_dir=results_subdir,
        rollout_steps=fargs.rollout_steps,
    )


# =============================================================================
# DLHF from-scratch run
# =============================================================================

def run_dlhf(fargs: FromScratchArgs, env: str, seed: int, base_dir: str):
    """Run from-scratch DLHF for a single (env, seed)."""
    B = fargs.batch_size
    steps = fargs.preference_budget // B
    rd = os.path.join(base_dir, env, f"seed_{seed}", "dlhf")

    # --- Skip if already done ---
    if not fargs.overwrite:
        cached = _result_exists(rd)
        if cached is not None:
            print(f"  SKIP dlhf {env} seed {seed} "
                  f"(result.npz exists, final_val_l1="
                  f"{cached['final_val_l1']:.4f})")
            return cached

    os.makedirs(rd, exist_ok=True)

    dlhf_args = _make_dlhf_args(fargs, env, seed, steps, rd)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = fargs.ensemble_size

    # Fresh random init
    opt = optax.adam(fargs.lr)
    all_params, all_opts = [], []
    for m in range(M):
        rng = jax.random.PRNGKey(seed + m * 1000)
        p = init_params(network, meta, fargs.batch_size, fargs.context_len, rng)
        all_params.append(p)
        all_opts.append(opt.init(p))

    stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opts)

    val_set   = collect_val_set(meta, fargs.val_size, fargs.val_scramble, seed)
    gen_batch = make_preferences_batch_fn(meta, fargs.batch_size,
                                          fargs.scramble_steps, fargs.horizon)
    train_rng = jax.random.PRNGKey(seed + 123)

    print(f"\n{'='*60}")
    print(f"  DLHF: {env} | seed {seed} | {steps} steps | "
          f"{fargs.preference_budget} pref labels")
    print(f"{'='*60}")

    t0 = time.time()
    stacked_params, _, losses, val_l1s, eval_steps = train_dlhf(
        network, stacked_params, stacked_opt_state, gen_batch,
        val_set, dlhf_args, train_rng, ensemble_size=M,
    )
    wall_time = time.time() - t0

    # Save
    for i in range(M):
        p = jax.tree.map(lambda x: x[i], stacked_params)
        save_params(p, os.path.join(rd, f"dlhf_{i}.pkl"))
    save_curves(losses, val_l1s, eval_steps, os.path.join(rd, "curves.png"))

    print("  Generating rollout visualisation...")
    p0 = jax.tree.map(lambda x: x[0], stacked_params)
    try:
        save_rollout(network, p0, meta, dlhf_args)
    except Exception as e:
        print(f"  Warning: rollout vis failed: {e}")

    result = dict(
        env=env, seed=seed, method="dlhf",
        eval_steps=np.array(eval_steps),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
        total_steps=steps,
        preference_budget=fargs.preference_budget,
    )
    np.savez(os.path.join(rd, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# Supervised baseline training
# =============================================================================

def _train_supervised(network, params, opt_state, dataset, val_set,
                      steps, batch_size, lr, eval_every, rng):
    """Train with reconstruction loss on a fixed offline dataset."""
    opt = optax.adam(lr)
    B = batch_size
    ctx = dataset["context_boards"]       # (N, T, obs_dim)
    act = dataset["context_actions"]      # (N, T)
    N = ctx.shape[0]

    @jax.jit
    def _val_l1(params):
        return network.apply(
            params, val_set["obs"], val_set["action"], val_set["gt_next"],
            method=LatentCNNNetwork.val_l1)

    @jax.jit
    def _step(params, opt_state, batch_ctx, batch_act, sk):
        def loss_fn(p):
            return network.apply(
                p, batch_ctx, batch_act, sk,
                method=LatentCNNNetwork.offline_loss)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    @jax.jit
    def _sample_batch(rng):
        idx = jax.random.randint(rng, (B,), 0, N)
        return ctx[idx], act[idx]

    @jax.jit
    def _scan_chunk(carry, _):
        params, opt_state, rng = carry
        rng, bk, sk = jax.random.split(rng, 3)
        batch_ctx, batch_act = _sample_batch(bk)
        params, opt_state, loss = _step(params, opt_state, batch_ctx, batch_act, sk)
        return (params, opt_state, rng), loss

    @jax.jit
    def _run_chunk(params, opt_state, rng):
        (params, opt_state, rng), losses = jax.lax.scan(
            _scan_chunk, (params, opt_state, rng), None, length=eval_every)
        return params, opt_state, rng, losses

    print(f"\n  Supervised -- {steps} steps, {N} transitions, "
          f"eval every {eval_every}")

    all_losses = []
    val_l1s    = []
    eval_steps = []

    val_l1 = float(_val_l1(params))
    val_l1s.append(val_l1)
    eval_steps.append(0)
    print(f"  init   | val_l1={val_l1:.4f}")

    t0 = time.time()
    steps_done = 0
    n_full = steps // eval_every
    remainder = steps % eval_every

    for _ in range(n_full):
        params, opt_state, rng, chunk_losses = _run_chunk(
            params, opt_state, rng)
        jax.block_until_ready(params)
        steps_done += eval_every
        all_losses.extend(np.array(chunk_losses).tolist())

        val_l1 = float(_val_l1(params))
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        loss = float(chunk_losses[-1])
        print(f"  step {steps_done:5d} | recon={loss:.4f} | "
              f"val_l1={val_l1:.4f} | {time.time()-t0:.1f}s")

    if remainder > 0:
        for _ in range(remainder):
            rng, bk, sk = jax.random.split(rng, 3)
            batch_ctx, batch_act = _sample_batch(bk)
            params, opt_state, loss = _step(
                params, opt_state, batch_ctx, batch_act, sk)
            all_losses.append(float(loss))
        steps_done += remainder
        jax.block_until_ready(params)
        val_l1 = float(_val_l1(params))
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        print(f"  step {steps_done:5d} | val_l1={val_l1:.4f} | "
              f"{time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  Done -- {elapsed:.1f}s ({elapsed/steps*1000:.1f} ms/step)")
    return params, all_losses, val_l1s, eval_steps


def run_supervised(fargs: FromScratchArgs, env: str, seed: int,
                   base_dir: str):
    """Run supervised baseline for a single (env, seed)."""
    B = fargs.batch_size
    steps = fargs.supervised_steps
    rd = os.path.join(base_dir, env, f"seed_{seed}", "supervised")

    # --- Skip if already done ---
    if not fargs.overwrite:
        cached = _result_exists(rd)
        if cached is not None:
            print(f"  SKIP supervised {env} seed {seed} "
                  f"(result.npz exists, final_val_l1="
                  f"{cached['final_val_l1']:.4f})")
            return cached

    os.makedirs(rd, exist_ok=True)

    # context_len=2: each sequence is exactly 1 transition
    sup_context_len = 2
    dlhf_args = _make_dlhf_args(fargs, env, seed, steps, rd)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)

    n_transitions = fargs.supervised_transitions

    print(f"\n{'='*60}")
    print(f"  SUPERVISED: {env} | seed {seed} | {steps} steps | "
          f"{n_transitions} transitions")
    print(f"{'='*60}")

    # Collect fixed dataset
    dataset = collect_offline_dataset(
        meta, n_transitions, sup_context_len,
        scramble_steps=fargs.scramble_steps, seed=seed,
    )

    # Same random init seed as DLHF
    rng = jax.random.PRNGKey(seed)
    params = init_params(
        network, meta, fargs.batch_size, sup_context_len, rng)
    opt = optax.adam(fargs.lr)
    opt_state = opt.init(params)

    val_set   = collect_val_set(
        meta, fargs.val_size, fargs.val_scramble, seed)
    train_rng = jax.random.PRNGKey(seed + 456)

    t0 = time.time()
    params, losses, val_l1s, eval_steps = _train_supervised(
        network, params, opt_state, dataset, val_set,
        steps, fargs.batch_size, fargs.lr, fargs.eval_every, train_rng,
    )
    wall_time = time.time() - t0

    save_params(params, os.path.join(rd, "supervised_0.pkl"))
    save_curves(losses, val_l1s, eval_steps,
                os.path.join(rd, "curves.png"))

    # Rollout vis
    dlhf_args_for_rollout = _make_dlhf_args(
        fargs, env, seed, steps, rd)
    print("  Generating supervised rollout visualisation...")
    try:
        save_rollout(network, params, meta, dlhf_args_for_rollout)
    except Exception as e:
        print(f"  Warning: rollout vis failed: {e}")

    result = dict(
        env=env, seed=seed, method="supervised",
        eval_steps=np.array(eval_steps),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
        total_steps=steps,
        supervised_transitions=n_transitions,
    )
    np.savez(os.path.join(rd, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# Summary plotting
# =============================================================================

def save_summary_plot(all_results, fargs: FromScratchArgs, base_dir: str):
    """Multi-env summary: DLHF vs supervised, one subplot per env.

    X-axis is training steps (not budget) since the two methods have
    fundamentally different data regimes.
    """
    envs = fargs.envs
    n_envs = len(envs)
    cols = min(n_envs, 4)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), dpi=130,
                             squeeze=False)

    colors = {"dlhf": "tab:red", "supervised": "tab:blue"}
    labels_map = {
        "dlhf": f"DLHF ({fargs.preference_budget:,} prefs)",
        "supervised": (f"Supervised "
                       f"({fargs.supervised_transitions:,} trans)"),
    }

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]

        for method in ["supervised", "dlhf"]:
            method_results = [
                r for r in all_results
                if r["env"] == env and r["method"] == method
            ]
            if not method_results:
                continue

            color = colors[method]

            # Individual seeds (thin lines)
            for r in method_results:
                ax.plot(r["eval_steps"], r["val_l1s"],
                        linewidth=0.5, alpha=0.3, color=color)

            # Mean +/- std
            ref_steps = method_results[0]["eval_steps"]
            aligned = [r["val_l1s"] for r in method_results
                       if len(r["val_l1s"]) == len(ref_steps)]
            if aligned:
                all_vals = np.array(aligned)
                mean_vals = all_vals.mean(axis=0)
                ax.plot(ref_steps, mean_vals, linewidth=2,
                        color=color,
                        label=(f"{labels_map[method]} "
                               f"(n={len(aligned)})"))
                if len(aligned) > 1:
                    std_vals = all_vals.std(axis=0)
                    ax.fill_between(
                        ref_steps,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        alpha=0.12, color=color)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Val L1")
        ax.set_title(env, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for idx in range(n_envs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"From-Scratch: DLHF ({fargs.preference_budget:,} prefs) vs "
        f"Supervised ({fargs.supervised_transitions:,} transitions)\n"
        f"({len(fargs.seeds)} seeds)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(base_dir, "summary.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary plot -> {path}")


def save_results_table(all_results, fargs: FromScratchArgs,
                       base_dir: str):
    """Print and save a summary table: DLHF vs supervised per env."""
    dlhf_steps = fargs.preference_budget // fargs.batch_size

    print(f"\n{'='*70}")
    print(f"  FROM-SCRATCH RESULTS")
    print(f"  Supervised: {fargs.supervised_transitions:,} transitions, "
          f"{fargs.supervised_steps:,} steps")
    print(f"  DLHF:       {fargs.preference_budget:,} preference labels, "
          f"{dlhf_steps:,} steps")
    print(f"{'='*70}")
    header = (f"  {'Env':<12s}  {'Supervised':>20s}  "
              f"{'DLHF (prefs)':>20s}  {'Gap':>10s}")
    sep = (f"  {'-'*12}  {'-'*20}  {'-'*20}  {'-'*10}")
    print(header)
    print(sep)

    lines = [header, sep]
    for env in fargs.envs:
        sup_results = [
            r for r in all_results
            if r["env"] == env and r["method"] == "supervised"
        ]
        dlhf_results = [
            r for r in all_results
            if r["env"] == env and r["method"] == "dlhf"
        ]

        sup_str, dlhf_str, gap_str = "---", "---", "---"
        sup_mean, dlhf_mean = None, None

        if sup_results:
            finals = [r["final_val_l1"] for r in sup_results]
            sup_mean = np.mean(finals)
            sup_std  = np.std(finals)
            sup_str = f"{sup_mean:.4f} +/- {sup_std:.4f}"

        if dlhf_results:
            finals = [r["final_val_l1"] for r in dlhf_results]
            dlhf_mean = np.mean(finals)
            dlhf_std  = np.std(finals)
            dlhf_str = f"{dlhf_mean:.4f} +/- {dlhf_std:.4f}"

        if sup_mean is not None and dlhf_mean is not None:
            gap = dlhf_mean - sup_mean
            gap_str = f"{gap:+.4f}"

        line = (f"  {env:<12s}  {sup_str:>20s}  "
                f"{dlhf_str:>20s}  {gap_str:>10s}")
        print(line)
        lines.append(line)

    # Save to text
    path = os.path.join(base_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("From-Scratch DLHF vs Supervised Results\n")
        f.write(f"Supervised: {fargs.supervised_transitions} "
                f"transitions, {fargs.supervised_steps} steps\n")
        f.write(f"DLHF: {fargs.preference_budget} preference labels, "
                f"{dlhf_steps} steps\n")
        f.write(f"Seeds: {fargs.seeds}\n\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Main
# =============================================================================

def main(fargs: FromScratchArgs) -> None:
    base_dir = (fargs.results_dir
                or os.path.join(OUT_DIR, "from_scratch"))
    os.makedirs(base_dir, exist_ok=True)

    B = fargs.batch_size
    dlhf_steps = fargs.preference_budget // B

    print(f"\n{'='*60}")
    print(f"  SECTION 4.1: FROM-SCRATCH DLHF vs SUPERVISED")
    print(f"{'='*60}")
    print(f"  Envs:              {fargs.envs}")
    print(f"  Seeds:             {fargs.seeds}")
    print(f"  Supervised:        {fargs.supervised_transitions:,} "
          f"transitions, {fargs.supervised_steps:,} steps")
    print(f"  DLHF:              {fargs.preference_budget:,} pref "
          f"labels, {dlhf_steps:,} steps")
    print(f"  Batch size:        {B}")
    print(f"  Ensemble:          {fargs.ensemble_size}")
    print(f"  Overwrite:         {fargs.overwrite}")

    all_results = []

    for env in fargs.envs:
        for seed in fargs.seeds:
            # --- Supervised baseline ---
            try:
                result = run_supervised(fargs, env, seed, base_dir)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR: supervised {env} seed {seed}: {e}")
                import traceback
                traceback.print_exc()

            # --- DLHF from scratch ---
            try:
                result = run_dlhf(fargs, env, seed, base_dir)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR: dlhf {env} seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("No successful runs. Exiting.")
        return

    # --- Aggregated outputs ---
    npz_path = os.path.join(base_dir, "summary.npz")
    np.savez(
        npz_path,
        envs=fargs.envs,
        seeds=fargs.seeds,
        preference_budget=fargs.preference_budget,
        supervised_transitions=fargs.supervised_transitions,
        supervised_steps=fargs.supervised_steps,
        run_envs=np.array([r["env"] for r in all_results]),
        run_seeds=np.array([r["seed"] for r in all_results]),
        run_methods=np.array([r["method"] for r in all_results]),
        run_final_val_l1=np.array(
            [r["final_val_l1"] for r in all_results]),
        run_wall_times=np.array(
            [r["wall_time"] for r in all_results]),
    )
    print(f"  Raw data -> {npz_path}")

    save_summary_plot(all_results, fargs, base_dir)
    save_results_table(all_results, fargs, base_dir)

    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(FromScratchArgs))