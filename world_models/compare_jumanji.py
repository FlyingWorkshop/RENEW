"""
world_models/compare_jumanji.py
======================================
Sample efficiency comparison: Naive DLHF vs RENEW on Jumanji envs.

Both methods train an ensemble of world models from scratch using only
preference labels. The ONLY difference:

    Naive DLHF:  random start states, 2-segment random preference pairs
    RENEW:       active start-state selection via ensemble disagreement,
                 K-candidate preference pairs

Fair comparison: both conditions use the SAME labels-per-step.
    - RENEW uses B start states and produces B*(K-1) labels per step.
    - Naive uses B*(K-1) as its batch size, also producing B*(K-1) labels
      per step.  This isolates active selection as the only variable.

Everything else is identical:
    - Same ensemble size (E members)
    - Same architecture, lr, beta, horizon
    - Same total preference label budget
    - Same number of training steps

Optionally, both methods can be warm-started with a short offline
(reconstruction) pretraining phase on random transitions. When enabled,
results are saved under a separate directory hierarchy so that
pretrained vs scratch comparisons are straightforward. Pretrained
checkpoints are saved to disk and can be reloaded with
--pretrain-checkpoint to re-run finetuning without re-pretraining.

Output structure (no pretrain):
    out/compare_dlhf_renew/{env}/seed_{s}/naive/curves.png, result.npz
    out/compare_dlhf_renew/{env}/seed_{s}/renew/curves.png, result.npz
    out/compare_dlhf_renew/comparison.png, summary.txt

Output structure (with pretrain):
    out/compare_dlhf_renew_pretrained/{env}/seed_{s}/pretrained_0.pkl, ...
    out/compare_dlhf_renew_pretrained/{env}/seed_{s}/naive/curves.png, result.npz
    out/compare_dlhf_renew_pretrained/{env}/seed_{s}/renew/curves.png, result.npz
    out/compare_dlhf_renew_pretrained/comparison.png, summary.txt

Usage:
    # Quick test (with default pretraining)
    python world_models/compare_jumanji.py --envs maze10 --seeds 0

    # No pretraining
    python world_models/compare_jumanji.py --no-pretrain --envs maze10 --seeds 0

    # Override budget for a specific env
    python world_models/compare_jumanji.py \
        --env-budgets.sliding5 3000 --env-budgets.maze10 12000

    # Re-run finetuning from saved pretrain checkpoint
    python world_models/compare_jumanji.py \
        --pretrain-checkpoint out/compare_dlhf_renew_pretrained/maze10/seed_0 \
        --overwrite --envs maze10 --seeds 0

    # Full run
    python world_models/compare_jumanji.py --envs maze10 sliding5 sokoban \
        --seeds 0 1 2 3 4

    # Skip completed seeds (default), force rerun:
    python world_models/compare_jumanji.py --overwrite
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    save_curves,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


# =============================================================================
# Args
# =============================================================================

@dataclass
class CompareArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    envs: List[str] = field(default_factory=lambda: [
        "sliding3",
        "sliding5", 
        "sokoban",
        # "pacman", 
        # "connector8_3",
        "maze5",
        "maze10",
        # "maze15",
        # "maze20",
    ])

    # --- Overwrite / rerun control ---
    overwrite: bool = False
    """Re-run finetuning even if result.npz exists. Also remakes plots."""
    overwrite_pretrain: bool = False
    """Re-run pretraining even if pretrained_*.pkl exist. Also remakes plots."""

    # --- Pretraining ---
    pretrain: bool = True
    """If True, warm-start with offline reconstruction pretraining."""
    pretrain_steps: int = 500
    """Number of offline pretraining gradient steps."""
    pretrain_batch: int = 32
    """Batch size for offline pretraining."""
    offline_dataset_size: int = 25
    """Number of context sequences to collect for pretraining."""
    pretrain_checkpoint: Optional[str] = None
    """Path to a directory containing pretrained_0.pkl, pretrained_1.pkl, ...
    If provided, skip pretraining and load from these files instead.
    When set, --pretrain is implied."""

    # --- Budget ---
    env_budgets: Dict[str, int] = field(default_factory=lambda: {
        "sliding3":     100_000,
        "sliding5":     100_000,
        "sokoban":     100_000,
        # "connector8_3": 50_000,
        "maze5":        100_000,
        "maze10":       100_000,
        "maze15":       100_000,
        "maze20":       100_000,
    })
    """Per-environment preference label budgets. Envs not in this dict
    fall back to default_budget."""
    default_budget: int = 5_000
    """Fallback budget for envs not listed in env_budgets."""

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
    """K — candidate rollouts per start state."""
    pool_multiplier:  int   = 16
    """Pool size = batch_size * pool_multiplier."""
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
# Budget resolver
# =============================================================================

def _get_budget(cargs: CompareArgs, env: str) -> int:
    """Return the preference budget for the given env name."""
    return cargs.env_budgets.get(env, cargs.default_budget)


def _budget_str(cargs: CompareArgs) -> str:
    """Human-readable budget summary for printing."""
    parts = []
    for env in cargs.envs:
        b = _get_budget(cargs, env)
        parts.append(f"{env}={b:,}")
    return ", ".join(parts)


# =============================================================================
# Build DLHFArgs for each method
# =============================================================================

def _make_dlhf_args(cargs: CompareArgs, env: str, seed: int,
                    steps: int, results_subdir: str,
                    renew: bool = False,
                    batch_size_override: Optional[int] = None) -> DLHFArgs:
    bs = batch_size_override if batch_size_override is not None else cargs.batch_size
    return DLHFArgs(
        seed=seed,
        env=env,
        embed_dim=cargs.embed_dim,
        latent_channels=cargs.latent_channels,
        enc_layers=cargs.enc_layers,
        dyn_layers=cargs.dyn_layers,
        dec_layers=cargs.dec_layers,
        lr=cargs.lr,
        steps=steps,
        batch_size=bs,
        horizon=cargs.horizon,
        beta_bt=cargs.beta_bt,
        scramble_steps=cargs.scramble_steps,
        context_len=cargs.context_len,
        ensemble_size=cargs.ensemble_size,
        renew=renew,
        num_candidates=cargs.num_candidates,
        pool_multiplier=cargs.pool_multiplier,
        selection_temp=cargs.selection_temp,
        val_size=cargs.val_size,
        val_scramble=cargs.val_scramble,
        eval_every=cargs.eval_every,
        results_dir=results_subdir,
        rollout_steps=cargs.rollout_steps,
    )


# =============================================================================
# Pretrain checkpoint I/O
# =============================================================================

def _save_pretrain_checkpoint(stacked_params, checkpoint_dir, ensemble_size):
    """Save pretrained ensemble members as pretrained_0.pkl, ..."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i in range(ensemble_size):
        p = jax.tree.map(lambda x: np.array(x[i]), stacked_params)
        path = os.path.join(checkpoint_dir, f"pretrained_{i}.pkl")
        with open(path, "wb") as f:
            pickle.dump(p, f)
    print(f"  Saved pretrain checkpoint ({ensemble_size} members) → {checkpoint_dir}")


def _load_pretrain_checkpoint(checkpoint_dir, ensemble_size):
    """Load pretrained ensemble members, return stacked params."""
    all_params = []
    for i in range(ensemble_size):
        path = os.path.join(checkpoint_dir, f"pretrained_{i}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pretrain checkpoint not found: {path}\n"
                f"Expected {ensemble_size} files: pretrained_0.pkl .. "
                f"pretrained_{ensemble_size-1}.pkl")
        with open(path, "rb") as f:
            all_params.append(pickle.load(f))
    stacked = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    print(f"  Loaded pretrain checkpoint ({ensemble_size} members) ← {checkpoint_dir}")
    return stacked


# =============================================================================
# Offline pretraining
# =============================================================================

def collect_offline_dataset(meta, size: int, context_len: int,
                            scramble_steps: int, seed: int):
    """Collect short context sequences via random policy for reconstruction."""
    env = meta.make_env()
    Tc  = context_len
    B   = size

    @jax.jit
    def _collect(key):
        k1, k2, k3 = jax.random.split(key, 3)
        states, _ = jax.vmap(env.reset)(jax.random.split(k1, B))
        def _scramble_step(i, carry):
            s, k = carry
            k = jax.random.fold_in(k, i)
            acts = jax.random.randint(k, (B,), 0, meta.num_actions)
            s, _ = jax.vmap(env.step)(s, acts)
            return s, k
        states, _ = jax.lax.fori_loop(
            0, scramble_steps, _scramble_step, (states, k2))
        ctx_boards, ctx_actions = [], []
        for t in range(Tc):
            acts = jax.random.randint(
                jax.random.fold_in(k3, t), (B,), 0, meta.num_actions)
            ctx_boards.append(meta.extract_obs(states))
            ctx_actions.append(acts)
            states, _ = jax.vmap(env.step)(states, acts)
        return dict(context_boards=jnp.stack(ctx_boards, 1),
                    context_actions=jnp.stack(ctx_actions, 1))

    print(f"  Collecting offline dataset ({size} seqs × {Tc} steps)...")
    dataset = jax.tree.map(
        lambda x: x.block_until_ready(),
        _collect(jax.random.PRNGKey(seed + 999_999)),
    )
    return dataset


def pretrain_ensemble(network, stacked_params, stacked_opt_state,
                      dataset, cargs: CompareArgs, seed: int):
    """Offline reconstruction pretraining for the full ensemble via vmap."""
    size = dataset["context_boards"].shape[0]
    opt  = optax.adam(cargs.lr)

    @jax.jit
    def _sample(key):
        idxs = jax.random.randint(key, (cargs.pretrain_batch,), 0, size)
        return jax.tree.map(lambda x: x[idxs], dataset)

    def _single_member_step(params, opt_state, batch, sk):
        def loss_fn(p):
            return network.apply(
                p, batch["context_boards"], batch["context_actions"], sk,
                method=LatentCNNNetwork.offline_loss)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    @jax.jit
    def _ensemble_step(stacked_params, stacked_opt_state, rng):
        rng, bk, sk = jax.random.split(rng, 3)
        batch = _sample(bk)
        new_params, new_opt_state, losses = jax.vmap(
            lambda p, o: _single_member_step(p, o, batch, sk)
        )(stacked_params, stacked_opt_state)
        return new_params, new_opt_state, rng, losses[0]

    chunk_size = min(50, cargs.pretrain_steps)

    @jax.jit
    def _scan_chunk(carry, _):
        sp, so, rng = carry
        sp, so, rng, loss = _ensemble_step(sp, so, rng)
        return (sp, so, rng), loss

    def _run_chunk(sp, so, rng, length):
        (sp, so, rng), losses = jax.lax.scan(
            _scan_chunk, (sp, so, rng), None, length=length)
        return sp, so, rng, losses

    rng = jax.random.PRNGKey(seed + 42)
    total = cargs.pretrain_steps
    n_full = total // chunk_size
    remainder = total % chunk_size

    print(f"  Pretraining ensemble ({cargs.ensemble_size} members, "
          f"{total} steps, batch={cargs.pretrain_batch})...")
    t0 = time.time()
    all_losses = []

    for _ in range(n_full):
        stacked_params, stacked_opt_state, rng, chunk_losses = _run_chunk(
            stacked_params, stacked_opt_state, rng, chunk_size)
        jax.block_until_ready(stacked_params)
        all_losses.extend(np.array(chunk_losses).tolist())

    if remainder > 0:
        stacked_params, stacked_opt_state, rng, chunk_losses = _run_chunk(
            stacked_params, stacked_opt_state, rng, remainder)
        jax.block_until_ready(stacked_params)
        all_losses.extend(np.array(chunk_losses).tolist())

    elapsed = time.time() - t0
    print(f"  Pretrain done: {total} steps in {elapsed:.1f}s | "
          f"final loss={all_losses[-1]:.4f}")

    return stacked_params, stacked_opt_state


# =============================================================================
# Run one (env, seed, method) — delegates to dlhf.py functions
# =============================================================================

def run_naive(cargs: CompareArgs, env: str, seed: int, out_dir: str,
              budget: int, pretrained_params=None):
    """Naive DLHF (batch-matched): same labels-per-step as RENEW."""
    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)
    naive_bs = labels_per_step
    steps = budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    dlhf_args = _make_dlhf_args(cargs, env, seed, steps, out_dir,
                                renew=False, batch_size_override=naive_bs)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = cargs.ensemble_size

    if pretrained_params is not None:
        stacked_params = pretrained_params
        opt = optax.adam(cargs.lr)
        stacked_opt_state = jax.vmap(opt.init)(stacked_params)
    else:
        opt = optax.adam(cargs.lr)
        all_params, all_opts = [], []
        for m in range(M):
            rng = jax.random.PRNGKey(seed + m * 1000)
            p = init_params(network, meta, naive_bs, cargs.context_len, rng)
            all_params.append(p)
            all_opts.append(opt.init(p))
        stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
        stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opts)

    val_set   = collect_val_set(meta, cargs.val_size, cargs.val_scramble, seed)
    gen_batch = make_preferences_batch_fn(
        meta, naive_bs, cargs.scramble_steps, cargs.horizon)
    train_rng = jax.random.PRNGKey(seed + 123)

    pt_tag = " (pretrained)" if pretrained_params is not None else ""
    print(f"\n{'='*60}")
    print(f"  NAIVE{pt_tag}: {env} | seed {seed} | {steps} steps | "
          f"B={naive_bs} | {labels_per_step} labels/step | "
          f"{budget:,} labels | E={M}")
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
        env=env, seed=seed, method="naive",
        eval_steps=np.array(eval_steps),
        labels=np.array(labels_at_eval),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
        budget=budget,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


def run_renew(cargs: CompareArgs, env: str, seed: int, out_dir: str,
              budget: int, pretrained_params=None):
    """RENEW: active selection + K-candidate preference pairs."""
    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)
    steps = budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    dlhf_args = _make_dlhf_args(cargs, env, seed, steps, out_dir, renew=True)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)
    M       = cargs.ensemble_size

    if pretrained_params is not None:
        stacked_params = pretrained_params
        opt = optax.adam(cargs.lr)
        stacked_opt_state = jax.vmap(opt.init)(stacked_params)
    else:
        opt = optax.adam(cargs.lr)
        all_params, all_opts = [], []
        for m in range(M):
            rng = jax.random.PRNGKey(seed + m * 1000)
            p = init_params(network, meta, B, cargs.context_len, rng)
            all_params.append(p)
            all_opts.append(opt.init(p))
        stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
        stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opts)

    val_set   = collect_val_set(meta, cargs.val_size, cargs.val_scramble, seed)
    train_rng = jax.random.PRNGKey(seed + 123)

    pt_tag = " (pretrained)" if pretrained_params is not None else ""
    print(f"\n{'='*60}")
    print(f"  RENEW{pt_tag}: {env} | seed {seed} | {steps} steps | "
          f"B={B} | {labels_per_step} labels/step | "
          f"{budget:,} labels | E={M} K={K}")
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
        env=env, seed=seed, method="renew",
        eval_steps=np.array(eval_steps),
        labels=np.array(labels_at_eval),
        val_l1s=np.array(val_l1s),
        losses=np.array(losses),
        final_val_l1=val_l1s[-1],
        wall_time=wall_time,
        budget=budget,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final val_l1={val_l1s[-1]:.4f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# Comparison plot
# =============================================================================

def save_comparison(all_results, envs, cargs, base_dir):
    n_envs = len(envs)
    cols = min(n_envs, 4)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), dpi=130,
                             squeeze=False)

    colors = {"naive": "tab:blue", "renew": "tab:red"}
    labels_map = {"naive": "Naive", "renew": "RENEW"}

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]
        budget = _get_budget(cargs, env)

        for method in ["naive", "renew"]:
            method_results = [
                r for r in all_results
                if r["env"] == env and r["method"] == method
            ]
            if not method_results:
                continue

            color = colors[method]

            for r in method_results:
                ax.plot(r["labels"], r["val_l1s"],
                        linewidth=0.5, alpha=0.3, color=color)

            ref_labels = method_results[0]["labels"]
            aligned = [r["val_l1s"] for r in method_results
                       if len(r["val_l1s"]) == len(ref_labels)]
            if aligned:
                arr = np.array(aligned)
                n = len(aligned)
                mean = arr.mean(0)
                ax.plot(ref_labels, mean, linewidth=2, color=color,
                        label=f"{labels_map[method]} (n={n})")
                if n > 1:
                    ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(n)
                    ax.fill_between(ref_labels, mean - ci, mean + ci,
                                    alpha=0.15, color=color)

        ax.set_xlabel("Preference Labels")
        ax.set_ylabel("Val L1")
        if not cargs.pretrain:
            ax.set_yscale("log")
        ax.set_title(f"{env} ({budget:,} labels)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(n_envs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    K = cargs.num_candidates
    labels_per_step = cargs.batch_size * (K - 1)
    pt_tag = " (pretrained)" if cargs.pretrain else ""
    fig.suptitle(
        f"Naive vs RENEW{pt_tag}\n"
        f"E={cargs.ensemble_size}, K={K}, {labels_per_step} labels/step | "
        f"{len(cargs.seeds)} seeds",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    path = os.path.join(base_dir, "comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison plot -> {path}")


def save_results_table(all_results, envs, cargs, base_dir):
    K = cargs.num_candidates
    labels_per_step = cargs.batch_size * (K - 1)
    pt_tag = " (pretrained)" if cargs.pretrain else ""
    init_label = "Pretrain" if cargs.pretrain else "Init"

    print(f"\n{'='*80}")
    print(f"  NAIVE vs RENEW{pt_tag} — FINAL RESULTS")
    print(f"  E={cargs.ensemble_size} K={K} | {labels_per_step} labels/step")
    if cargs.pretrain:
        print(f"  Pretrain: {cargs.pretrain_steps} steps on "
              f"{cargs.offline_dataset_size} offline sequences")
    print(f"{'='*80}")
    header = (f"  {'Env':<12s}  {'Budget':>8s}  {init_label:>20s}  "
              f"{'Naive':>20s}  {'RENEW':>20s}  {'Gap':>10s}")
    sep = (f"  {'-'*12}  {'-'*8}  {'-'*20}  "
           f"{'-'*20}  {'-'*20}  {'-'*10}")
    print(header); print(sep)

    lines = [header, sep]
    for idx, env in enumerate(envs):
        budget = _get_budget(cargs, env)

        def _fmt(method):
            results = [r for r in all_results
                       if r["env"] == env and r["method"] == method]
            if not results:
                return "---", None
            finals = [r["final_val_l1"] for r in results]
            m = np.mean(finals)
            n = len(finals)
            ci = 1.96 * np.std(finals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return f"{m:.4f} ± {ci:.4f}", m

        def _fmt_init():
            # Use val_l1s[0] from any method — both start from same params
            results = [r for r in all_results
                       if r["env"] == env and "val_l1s" in r
                       and len(r["val_l1s"]) > 0]
            if not results:
                return "---"
            inits = [float(r["val_l1s"][0]) for r in results]
            m = np.mean(inits)
            n = len(inits)
            ci = 1.96 * np.std(inits, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return f"{m:.4f} ± {ci:.4f}"

        init_str = _fmt_init()
        naive_str, naive_m = _fmt("naive")
        renew_str, renew_m = _fmt("renew")
        gap = (f"{renew_m - naive_m:+.4f}"
               if naive_m is not None and renew_m is not None else "---")
        line = (f"  {env:<12s}  {budget:>8,}  {init_str:>20s}  "
                f"{naive_str:>20s}  {renew_str:>20s}  {gap:>10s}")
        print(line); lines.append(line)

    path = os.path.join(base_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(f"Naive vs RENEW{pt_tag} Results\n")
        f.write(f"E={cargs.ensemble_size}, K={cargs.num_candidates}\n")
        f.write(f"Labels/step: {labels_per_step}\n")
        if cargs.pretrain:
            f.write(f"Pretrain: {cargs.pretrain_steps} steps on "
                    f"{cargs.offline_dataset_size} offline sequences\n")
        f.write("\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Per-seed results printout
# =============================================================================

def _print_seed_result(env, seed, seed_results):
    """Print comparison for a single (env, seed) after both methods complete."""
    naive_r = next((r for r in seed_results if r["method"] == "naive"), None)
    renew_r = next((r for r in seed_results if r["method"] == "renew"), None)

    if naive_r is None or renew_r is None:
        return

    naive_l1 = naive_r["final_val_l1"]
    renew_l1 = renew_r["final_val_l1"]
    gap = naive_l1 - renew_l1

    if gap > 0:
        winner = "RENEW wins"
    elif gap < 0:
        winner = "Naive wins"
    else:
        winner = "Tie"

    print(f"\n  --- {env} seed {seed} ---")
    print(f"  Naive val_l1 = {naive_l1:.4f}")
    print(f"  RENEW val_l1 = {renew_l1:.4f}")
    print(f"  Gap (naive - renew) = {gap:+.4f}  ({winner})")


def _save_seed_comparison_plot(seed_dir, env, seed, seed_results,
                              force=False, pretrain=False):
    """Save a per-seed naive vs RENEW val_l1 comparison plot.

    Generates the plot from result data (including cached .npz).
    Only skips if the plot already exists and force is False.
    """
    path = os.path.join(seed_dir, "comparison.png")
    if os.path.exists(path) and not force:
        return

    naive_r = next((r for r in seed_results if r["method"] == "naive"), None)
    renew_r = next((r for r in seed_results if r["method"] == "renew"), None)

    if naive_r is None or renew_r is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)

    ax.plot(naive_r["labels"], naive_r["val_l1s"],
            linewidth=1.5, color="tab:blue",
            label="Naive")
    ax.plot(renew_r["labels"], renew_r["val_l1s"],
            linewidth=1.5, color="tab:red",
            label="RENEW")

    ax.set_xlabel("Preference Labels")
    ax.set_ylabel("Val L1")
    if not pretrain:
        ax.set_yscale("log")
    ax.set_title(f"{env} — seed {seed}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    naive_final = naive_r["final_val_l1"]
    renew_final = renew_r["final_val_l1"]
    gap = naive_final - renew_final
    winner = "RENEW" if gap > 0 else ("Naive" if gap < 0 else "Tie")
    ax.annotate(
        f"Naive={naive_final:.4f}  RENEW={renew_final:.4f}\n"
        f"Gap={gap:+.4f} ({winner})",
        xy=(0.98, 0.98), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Seed comparison plot → {path}")


# =============================================================================
# Main
# =============================================================================

def main(cargs: CompareArgs):
    # --pretrain-checkpoint implies --pretrain
    if cargs.pretrain_checkpoint is not None:
        cargs.pretrain = True

    # Any overwrite implies remaking plots
    remake_plots = cargs.overwrite or cargs.overwrite_pretrain

    suffix = "_pretrained" if cargs.pretrain else ""
    base_dir = cargs.results_dir or os.path.join(
        OUT_DIR, f"compare_dlhf_renew{suffix}")
    os.makedirs(base_dir, exist_ok=True)

    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)

    print(f"\n{'='*60}")
    print(f"  NAIVE vs RENEW COMPARISON")
    print(f"{'='*60}")
    print(f"  Envs:           {cargs.envs}")
    print(f"  Seeds:          {cargs.seeds}")
    print(f"  Label budgets:  {_budget_str(cargs)}")
    print(f"  Labels/step:    {labels_per_step} (both methods)")
    print(f"  RENEW batch:    B={B} (K={K})")
    print(f"  Naive batch:    B={labels_per_step}")
    print(f"  Ensemble:       E={cargs.ensemble_size}")
    if cargs.pretrain_checkpoint:
        print(f"  Pretrain:       from checkpoint {cargs.pretrain_checkpoint}")
    elif cargs.pretrain:
        print(f"  Pretrain:       {cargs.pretrain_steps} steps on "
              f"{cargs.offline_dataset_size} offline seqs "
              f"(batch={cargs.pretrain_batch})")
    else:
        print(f"  Pretrain:       off")
    print(f"  Overwrite:      {cargs.overwrite}")
    print(f"  Overwrite PT:   {cargs.overwrite_pretrain}")
    print(f"  Output:         {base_dir}")
    print(f"{'='*60}")

    all_results = []

    for env_idx, env in enumerate(cargs.envs):
        budget = _get_budget(cargs, env)
        steps = budget // labels_per_step

        for seed in cargs.seeds:
            seed_dir = os.path.join(base_dir, env, f"seed_{seed}")

            # --- Check if both methods already cached ---
            naive_npz = os.path.join(seed_dir, "naive", "result.npz")
            renew_npz = os.path.join(seed_dir, "renew", "result.npz")
            both_cached = (os.path.exists(naive_npz) and
                           os.path.exists(renew_npz) and
                           not cargs.overwrite)

            # --- Pretrained params (shared init for both methods) ---
            pretrained_params = None
            if cargs.pretrain and not both_cached:
                if cargs.pretrain_checkpoint:
                    pretrained_params = _load_pretrain_checkpoint(
                        cargs.pretrain_checkpoint, cargs.ensemble_size)
                else:
                    ckpt_dir = seed_dir
                    ckpt_path = os.path.join(ckpt_dir, "pretrained_0.pkl")

                    if os.path.exists(ckpt_path) and not cargs.overwrite_pretrain:
                        print(f"\n--- Loading cached pretrain for "
                              f"{env} seed {seed} ---")
                        pretrained_params = _load_pretrain_checkpoint(
                            ckpt_dir, cargs.ensemble_size)
                    else:
                        dummy_args = _make_dlhf_args(
                            cargs, env, seed, 1, base_dir, renew=False)
                        meta    = make_meta(dummy_args)
                        network = build_network(meta, dummy_args)
                        M       = cargs.ensemble_size

                        opt = optax.adam(cargs.lr)
                        all_params, all_opts = [], []
                        for m in range(M):
                            rng = jax.random.PRNGKey(seed + m * 1000)
                            p = init_params(network, meta,
                                            cargs.pretrain_batch,
                                            cargs.context_len, rng)
                            all_params.append(p)
                            all_opts.append(opt.init(p))

                        stacked_params = jax.tree.map(
                            lambda *ps: jnp.stack(ps), *all_params)
                        stacked_opt_state = jax.tree.map(
                            lambda *os: jnp.stack(os), *all_opts)

                        print(f"\n--- Pretraining for {env} seed {seed} ---")
                        dataset = collect_offline_dataset(
                            meta, cargs.offline_dataset_size,
                            cargs.context_len, cargs.scramble_steps, seed)
                        stacked_params, stacked_opt_state = pretrain_ensemble(
                            network, stacked_params, stacked_opt_state,
                            dataset, cargs, seed)
                        pretrained_params = stacked_params

                        _save_pretrain_checkpoint(
                            pretrained_params, ckpt_dir, cargs.ensemble_size)

            # --- Run both methods ---
            seed_results = []
            for method_name, run_fn in [("naive", run_naive),
                                        ("renew", run_renew)]:
                rd = os.path.join(seed_dir, method_name)
                npz_path = os.path.join(rd, "result.npz")

                if os.path.exists(npz_path) and not cargs.overwrite:
                    print(f"\n  {method_name} {env} seed {seed}: "
                          f"loading {npz_path}")
                    d = dict(np.load(npz_path, allow_pickle=True))
                    result = dict(
                        env=str(d.get("env", env)),
                        seed=int(d.get("seed", seed)),
                        method=str(d.get("method", method_name)),
                        eval_steps=d["eval_steps"],
                        labels=d["labels"],
                        val_l1s=d["val_l1s"],
                        final_val_l1=float(d["final_val_l1"]),
                    )
                else:
                    pp = (jax.tree.map(lambda x: x.copy(), pretrained_params)
                          if pretrained_params is not None else None)
                    try:
                        result = run_fn(cargs, env, seed, rd,
                                        budget=budget,
                                        pretrained_params=pp)
                    except Exception as e:
                        print(f"\n  ERROR: {method_name} {env} seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue

                seed_results.append(result)
                all_results.append(result)

            _print_seed_result(env, seed, seed_results)
            _save_seed_comparison_plot(seed_dir, env, seed, seed_results,
                                      force=remake_plots,
                                      pretrain=cargs.pretrain)

    if not all_results:
        print("No successful runs. Exiting.")
        return

    np.savez(
        os.path.join(base_dir, "summary.npz"),
        envs=cargs.envs, seeds=cargs.seeds,
        env_budgets=np.array([_get_budget(cargs, e) for e in cargs.envs]),
        pretrain=cargs.pretrain,
        pretrain_steps=cargs.pretrain_steps,
        run_envs=np.array([r["env"] for r in all_results]),
        run_seeds=np.array([r["seed"] for r in all_results]),
        run_methods=np.array([r["method"] for r in all_results]),
        run_final_val_l1=np.array([r["final_val_l1"] for r in all_results]),
    )

    save_comparison(all_results, cargs.envs, cargs, base_dir)
    save_results_table(all_results, cargs.envs, cargs, base_dir)
    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))