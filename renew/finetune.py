"""
renew/finetune.py
=================
The repair experiment: load a pretrained world-model ensemble and run two
finetuning methods on the same preference budget, then compare:

  1. Naive  — random preference pairs (baseline)
  2. RENEW  — uncertainty-targeted preference pairs (ours)

Each RENEW round recomputes epistemic uncertainty (ensemble disagreement)
from the current model, then samples preference start states weighted by
uncertainty. No value iteration or policy rollouts — purely
uncertainty-driven selection.

Writes per-method metrics/curves under --results-dir and a comparison
figure (Pretrained / Naive / RENEW × error / uncertainty heatmaps).

Usage (typically driven by renew/run_finetune.sh):
  python renew/finetune.py --checkpoint out/maze10/seed_0/pretrained.pkl \
      --seed 0 --maze-size 10 --pref-budget 1600 --num-rounds 3 \
      --ensemble-size 3 --results-dir out/maze10/seed_0
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pretrain import (
    LatentCNNNetwork, EnvMeta, make_maze_meta, build_network,
    evaluate_grid, get_maze_layout,
    _add_walls_and_star, make_error_heatmap, make_unc_heatmap,
    load_params, save_params, load_ensemble, save_ensemble,
    save_loss_accuracy_curves, save_results,
    NUM_ACTIONS, NUM_TILES, OUT_DIR,
)

FINETUNE_DIR = os.path.join(OUT_DIR, "finetune")
os.makedirs(FINETUNE_DIR, exist_ok=True)


# =============================================================================
# Args
# =============================================================================

@dataclass
class FinetuneArgs:
    checkpoint: str = "out/pretrained.pkl"
    """Path to pretrained WM checkpoint."""
    seed: int = 0

    # --- Environment ---
    maze_size: int = 10

    # --- Architecture (must match checkpoint) ---
    embed_dim:       int = 32
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 3
    dec_layers:      int = 2

    # --- Shared preference budget ---
    pref_budget: int = 1600
    """Total preference pairs queried from oracle. Same for both methods."""
    finetune_batch: int = 32
    finetune_lr:    float = 1e-4
    finetune_horizon: int = 1
    beta_bt: float = 1.0

    # --- RENEW active learning ---
    num_rounds: int = 3
    """Number of active learning rounds for RENEW (recompute uncertainty each round)."""
    ensemble_size: int = 1
    """Number of ensemble members. Must match pretrained checkpoint."""

    # --- Data gen ---
    scramble_steps: int = 0

    # --- Eval during finetuning ---
    eval_every: int = 10
    """Evaluate grid accuracy every N finetune steps."""

    # --- Results output ---
    results_dir: Optional[str] = None
    """If set, save metrics/curves data here for aggregation."""


# =============================================================================
# Uncertainty computation from WM decoder
# =============================================================================

def compute_cell_uncertainty(wm_network, wm_params, maze_size, seed=0,
                             ensemble_params=None):
    """
    Compute per-(cell, action) uncertainty from the WM.
    If ensemble_params is provided (list of param dicts), uses ensemble
    disagreement (mutual information). Otherwise falls back to single-model
    softmax entropy.
    Returns obs_pool (n_floor, obs_dim), unc_per_cell_action (n_floor, NUM_ACTIONS),
    and floor_cells.
    """
    walls, target_cell, _ = get_maze_layout(maze_size, seed)
    walls_flat = walls.ravel().astype(bool)
    floor_cells = np.where(~walls_flat)[0]
    n_floor = len(floor_cells)

    base_grid = np.where(walls_flat, 0, 1).astype(np.int32)
    base_grid[target_cell] = 3

    # Build obs for each floor cell
    all_obs = []
    for cell in floor_cells:
        obs = base_grid.copy()
        obs[cell] = 2
        all_obs.append(obs)
    all_obs = np.array(all_obs)  # (n_floor, obs_dim)

    # Expand over actions: (n_floor * NUM_ACTIONS,)
    obs_expanded = np.repeat(all_obs, NUM_ACTIONS, axis=0)
    acts_expanded = np.tile(np.arange(NUM_ACTIONS), n_floor)

    @jax.jit
    def _get_probs(params, obs, actions):
        logits = wm_network.apply(params, obs, actions, method=LatentCNNNetwork.decode)
        return jax.nn.softmax(logits, axis=-1)

    CHUNK = 2048

    if ensemble_params is not None and len(ensemble_params) > 1:
        # Ensemble disagreement: entropy of mean - mean of entropies
        all_member_probs = []
        for mp in ensemble_params:
            member_probs = []
            for i in range(0, obs_expanded.shape[0], CHUNK):
                o = jnp.array(obs_expanded[i:i+CHUNK])
                a = jnp.array(acts_expanded[i:i+CHUNK])
                member_probs.append(np.array(_get_probs(mp, o, a)))
            all_member_probs.append(np.concatenate(member_probs, axis=0))
        stacked = np.stack(all_member_probs, axis=0)  # (M, N, obs_dim, num_tiles)
        mean_probs = stacked.mean(axis=0)
        ent_mean = -(mean_probs * np.log(mean_probs + 1e-10)).sum(-1)  # (N, obs_dim)
        ent_each = -(stacked * np.log(stacked + 1e-10)).sum(-1)  # (M, N, obs_dim)
        mean_ent = ent_each.mean(axis=0)
        epistemic = (ent_mean - mean_ent).mean(-1)  # (N,)
        all_entropies = epistemic
    else:
        # Single model fallback: softmax entropy
        all_entropies = []
        for i in range(0, obs_expanded.shape[0], CHUNK):
            o = jnp.array(obs_expanded[i:i+CHUNK])
            a = jnp.array(acts_expanded[i:i+CHUNK])
            p = np.array(_get_probs(wm_params, o, a))
            cell_ent = -(p * np.log(p + 1e-10)).sum(-1)
            all_entropies.append(cell_ent.mean(-1))
        all_entropies = np.concatenate(all_entropies, axis=0)

    unc_per_cell_action = all_entropies.reshape(n_floor, NUM_ACTIONS)

    return all_obs, unc_per_cell_action, floor_cells


def build_uncertainty_grid(unc_per_cell_action, floor_cells, walls, maze_size):
    """Build a (maze_size, maze_size) grid of mean uncertainty for visualization."""
    unc_grid = np.full((maze_size, maze_size), np.nan)
    mean_unc = unc_per_cell_action.mean(axis=-1)  # (n_floor,)
    for i, cell in enumerate(floor_cells):
        r, c = divmod(cell, maze_size)
        unc_grid[r, c] = mean_unc[i]
    return unc_grid


# =============================================================================
# Naive finetuning (random preferences)
# =============================================================================

def make_random_pref_batch_fn(meta: EnvMeta, args: FinetuneArgs) -> Callable:
    env = meta.make_env()
    H = args.finetune_horizon

    def _oracle_rollout(start_obs, actions, key):
        obs = start_obs
        boards = []
        for t in range(H):
            obs = meta.oracle_step(jax.random.fold_in(key, t), obs, actions[t])
            boards.append(obs)
        return jnp.stack(boards, 0)

    @jax.jit
    def generate(key):
        keys = jax.random.split(key, 5)
        B = args.finetune_batch
        states, _ = jax.vmap(env.reset)(jax.random.split(keys[0], B))
        def _scramble(i, s):
            k = jax.random.fold_in(keys[1], i)
            acts = jax.random.randint(k, (B,), 0, meta.num_actions)
            s, _ = jax.vmap(env.step)(s, acts)
            return s
        states = jax.lax.fori_loop(0, args.scramble_steps, _scramble, states)
        start_obs = meta.extract_obs(states)
        acts1 = jax.random.randint(keys[2], (B, H), 0, meta.num_actions)
        acts2 = jax.random.randint(keys[3], (B, H), 0, meta.num_actions)
        okeys = jax.random.split(keys[4], B)
        gt1 = jax.vmap(_oracle_rollout)(start_obs, acts1, okeys)
        gt2 = jax.vmap(_oracle_rollout)(
            start_obs, acts2,
            jax.vmap(lambda k: jax.random.fold_in(k, 99))(okeys))
        return dict(start_obs=start_obs, acts1=acts1, acts2=acts2, gt1=gt1, gt2=gt2)

    return generate


def finetune_naive(network, params_list, meta, args, rng):
    """Finetune with random preference pairs. Finetunes all ensemble members
    on the same batches using vmap. Returns (params_list, rng, losses, accs, acc_steps)."""
    total_steps = args.pref_budget // args.finetune_batch
    gen_batch = make_random_pref_batch_fn(meta, args)

    # Stack into (M, ...) pytree for vmap, init opt per-member then stack
    stacked_params = jax.tree.map(lambda *ps: jnp.stack(ps), *params_list)
    opt = optax.adam(args.finetune_lr)
    all_opt_states = [opt.init(p) for p in params_list]
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opt_states)

    walls, target_cell, _ = get_maze_layout(args.maze_size, args.seed)

    def _single_step(params, opt_state, batch, sk):
        def loss_fn(p):
            return network.apply(
                p, batch["start_obs"], batch["acts1"], batch["acts2"],
                batch["gt1"], batch["gt2"], sk,
                method=LatentCNNNetwork.preferences_loss)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, info

    @jax.jit
    def _ensemble_step(stacked_params, stacked_opt_state, rng):
        rng, bk, sk = jax.random.split(rng, 3)
        batch = gen_batch(bk)
        new_params, new_opt_state, info = jax.vmap(
            lambda p, o: _single_step(p, o, batch, sk)
        )(stacked_params, stacked_opt_state)
        return new_params, new_opt_state, rng, info

    print(f"\n  Naive finetuning ({total_steps} steps × {args.finetune_batch} "
          f"= {total_steps * args.finetune_batch} prefs)...")

    all_losses = []
    acc_steps = []
    acc_values = []

    t0 = time.time()
    for step in range(total_steps):
        stacked_params, stacked_opt_state, rng, info = _ensemble_step(
            stacked_params, stacked_opt_state, rng)

        loss_val = float(info["bt_loss"][0])
        all_losses.append(loss_val)

        if step == 0 or (step + 1) % args.eval_every == 0 or step == total_steps - 1:
            jax.block_until_ready(stacked_params)
            p0 = jax.tree.map(lambda x: x[0], stacked_params)
            ens_list = [jax.tree.map(lambda x: x[i], stacked_params)
                        for i in range(len(params_list))] if len(params_list) > 1 else None
            metrics = evaluate_grid(network, p0, walls, target_cell,
                                    args.maze_size, ensemble_params=ens_list)
            acc = metrics["transition_acc"]
            acc_steps.append(step)
            acc_values.append(acc)
            print(f"    Step {step+1:4d}/{total_steps} | BT={loss_val:.4f} | "
                  f"acc={acc*100:.1f}%")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Unstack back to list
    result_list = [jax.tree.map(lambda x: x[i], stacked_params)
                   for i in range(len(params_list))]
    return result_list, rng, all_losses, acc_values, acc_steps


# =============================================================================
# RENEW: Targeted preference generation (uncertainty-weighted)
# =============================================================================

def make_targeted_pref_batch_fn(meta, args, obs_pool, unc_per_cell_action):
    """
    Build a preference batch generator that samples start states weighted
    by their mean decoder uncertainty across actions.
    """
    H = args.finetune_horizon

    # Weight by mean uncertainty across actions per cell
    mean_unc = unc_per_cell_action.mean(axis=-1)  # (n_floor,)
    weights = np.exp(mean_unc - mean_unc.max())
    weights = weights / (weights.sum() + 1e-10)
    weights_j = jnp.array(weights)
    obs_pool_j = jnp.array(obs_pool)

    def _oracle_rollout(start_obs, actions, key):
        obs = start_obs
        boards = []
        for t in range(H):
            obs = meta.oracle_step(jax.random.fold_in(key, t), obs, actions[t])
            boards.append(obs)
        return jnp.stack(boards, 0)

    @jax.jit
    def generate(key):
        keys = jax.random.split(key, 4)
        B = args.finetune_batch
        idxs = jax.random.choice(keys[0], obs_pool_j.shape[0], (B,), p=weights_j)
        start_obs = obs_pool_j[idxs]
        acts1 = jax.random.randint(keys[1], (B, H), 0, meta.num_actions)
        acts2 = jax.random.randint(keys[2], (B, H), 0, meta.num_actions)
        okeys = jax.random.split(keys[3], B)
        gt1 = jax.vmap(_oracle_rollout)(start_obs, acts1, okeys)
        gt2 = jax.vmap(_oracle_rollout)(
            start_obs, acts2,
            jax.vmap(lambda k: jax.random.fold_in(k, 99))(okeys))
        return dict(start_obs=start_obs, acts1=acts1, acts2=acts2, gt1=gt1, gt2=gt2)

    return generate


# =============================================================================
# RENEW: Active finetuning loop (uncertainty-based)
# =============================================================================

def finetune_renew(network, params_list, meta, args, rng):
    """
    Active finetuning with uncertainty-based selection.
    Each round: recompute WM uncertainty (ensemble disagreement if available),
    then sample preferences weighted by uncertainty. Uses vmap for ensemble.
    Returns (params_list, rng, losses, accs, acc_steps, unc_grids).
    """
    maze_size = args.maze_size
    M = len(params_list)
    walls, target_cell, target_rc = get_maze_layout(maze_size, args.seed)

    prefs_per_round = args.pref_budget // args.num_rounds
    steps_per_round = prefs_per_round // args.finetune_batch

    all_unc_grids = []
    all_losses = []
    acc_steps = []
    acc_values = []
    global_step = 0

    # Stack for vmap, init opt per-member then stack
    stacked_params = jax.tree.map(lambda *ps: jnp.stack(ps), *params_list)
    opt = optax.adam(args.finetune_lr)
    all_opt_states = [opt.init(p) for p in params_list]
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opt_states)

    def _single_step(params, opt_state, batch, sk):
        def loss_fn(p):
            return network.apply(
                p, batch["start_obs"], batch["acts1"], batch["acts2"],
                batch["gt1"], batch["gt2"], sk,
                method=LatentCNNNetwork.preferences_loss)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, info

    for rd in range(1, args.num_rounds + 1):
        print(f"\n    Round {rd}/{args.num_rounds}")

        # Unstack for uncertainty computation
        cur_list = [jax.tree.map(lambda x: x[i], stacked_params) for i in range(M)]

        # Compute uncertainty from current WM (ensemble or single)
        unc_label = "ensemble disagreement" if M > 1 else "decoder entropy"
        print(f"      Computing WM uncertainty ({unc_label})...")
        ens = cur_list if M > 1 else None
        obs_pool, unc_per_cell_action, floor_cells = compute_cell_uncertainty(
            network, cur_list[0], maze_size, seed=args.seed,
            ensemble_params=ens)
        mean_unc = unc_per_cell_action.mean()
        print(f"      mean_unc={mean_unc:.4f}, "
              f"max_unc={unc_per_cell_action.max():.4f}")

        unc_grid = build_uncertainty_grid(unc_per_cell_action, floor_cells, walls, maze_size)
        all_unc_grids.append(unc_grid)

        # Build targeted batch generator (new each round)
        gen_batch = make_targeted_pref_batch_fn(meta, args, obs_pool, unc_per_cell_action)

        @jax.jit
        def _ensemble_step(stacked_params, stacked_opt_state, rng):
            rng, bk, sk = jax.random.split(rng, 3)
            batch = gen_batch(bk)
            new_params, new_opt_state, info = jax.vmap(
                lambda p, o: _single_step(p, o, batch, sk)
            )(stacked_params, stacked_opt_state)
            return new_params, new_opt_state, rng, info

        rng, fk = jax.random.split(rng)
        for step in range(steps_per_round):
            stacked_params, stacked_opt_state, fk, info = _ensemble_step(
                stacked_params, stacked_opt_state, fk)

            loss_val = float(info["bt_loss"][0])
            all_losses.append(loss_val)

            if (step == 0 or (step + 1) % args.eval_every == 0
                    or step == steps_per_round - 1):
                jax.block_until_ready(stacked_params)
                p0 = jax.tree.map(lambda x: x[0], stacked_params)
                ens_eval = [jax.tree.map(lambda x: x[i], stacked_params)
                            for i in range(M)] if M > 1 else None
                metrics = evaluate_grid(network, p0, walls, target_cell,
                                        maze_size, ensemble_params=ens_eval)
                acc = metrics["transition_acc"]
                acc_steps.append(global_step + step)
                acc_values.append(acc)

            global_step += 1

        jax.block_until_ready(stacked_params)
        p0 = jax.tree.map(lambda x: x[0], stacked_params)
        ens_eval = [jax.tree.map(lambda x: x[i], stacked_params)
                    for i in range(M)] if M > 1 else None
        metrics = evaluate_grid(network, p0, walls, target_cell,
                                maze_size, ensemble_params=ens_eval)
        print(f"      Finetune: {steps_per_round} steps, "
              f"BT={float(info['bt_loss'][0]):.4f} | acc={metrics['transition_acc']*100:.1f}%")

    # Unstack back to list
    result_list = [jax.tree.map(lambda x: x[i], stacked_params) for i in range(M)]
    return result_list, rng, all_losses, acc_values, acc_steps, all_unc_grids


# =============================================================================
# Comparison figure
# =============================================================================

def save_comparison_figure(pretrain_metrics, naive_metrics, renew_metrics,
                           renew_unc_grids, maze_size, save_path):
    """
    3-column comparison: Pretrained / Naive / RENEW
    Row 1: Error heatmaps
    Row 2: Uncertainty heatmaps
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)

    make_error_heatmap(axes[0, 0], pretrain_metrics, maze_size)
    axes[0, 0].set_title(f"Pretrained\nacc={pretrain_metrics['transition_acc']*100:.1f}%",
                         fontsize=11)
    make_error_heatmap(axes[0, 1], naive_metrics, maze_size)
    axes[0, 1].set_title(f"Naive Finetune\nacc={naive_metrics['transition_acc']*100:.1f}%",
                         fontsize=11)
    make_error_heatmap(axes[0, 2], renew_metrics, maze_size)
    axes[0, 2].set_title(f"RENEW Finetune\nacc={renew_metrics['transition_acc']*100:.1f}%",
                         fontsize=11)

    make_unc_heatmap(axes[1, 0], pretrain_metrics, maze_size)
    make_unc_heatmap(axes[1, 1], naive_metrics, maze_size)
    make_unc_heatmap(axes[1, 2], renew_metrics, maze_size)

    fig.suptitle("Preference Finetuning Comparison (same budget)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure → {save_path}")

    # RENEW-specific uncertainty evolution panel
    if renew_unc_grids:
        n_rounds = len(renew_unc_grids)
        fig2, axes2 = plt.subplots(1, n_rounds + 1, figsize=(5 * (n_rounds + 1), 5), dpi=150)
        if n_rounds + 1 == 1:
            axes2 = [axes2]

        walls = renew_metrics["walls"]
        target_rc = renew_metrics["target_rc"]

        # Show error heatmap at the end
        ax = axes2[0]
        make_error_heatmap(ax, renew_metrics, maze_size)
        ax.set_title(f"RENEW Final Errors\nacc={renew_metrics['transition_acc']*100:.1f}%",
                     fontsize=10)

        # Show uncertainty grids per round
        for i, unc_grid in enumerate(renew_unc_grids):
            ax = axes2[i + 1]
            g = unc_grid.copy()
            g[walls] = np.nan
            im = ax.imshow(g, cmap="inferno", interpolation="nearest", origin="upper")
            for r in range(maze_size):
                for c in range(maze_size):
                    if not walls[r, c] and not np.isnan(g[r, c]) and g[r, c] > 0.01:
                        ax.text(c, r, f"{g[r,c]:.2f}", ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold", zorder=3)
            ax.set_title(f"Round {i+1} Uncertainty\nmean={np.nanmean(g):.4f}", fontsize=10)
            fig2.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            _add_walls_and_star(ax, walls, target_rc, maze_size)

        fig2.suptitle("RENEW — Uncertainty Evolution", fontsize=14, y=1.01)
        fig2.tight_layout()
        renew_path = save_path.replace("comparison.png", "renew_detail.png")
        fig2.savefig(renew_path, bbox_inches="tight")
        plt.close(fig2)
        print(f"  RENEW detail → {renew_path}")


# =============================================================================
# Main
# =============================================================================

def main(args: FinetuneArgs) -> None:
    maze_size = args.maze_size
    network, meta = build_network(maze_size, args)

    # Determine output directory
    if args.results_dir:
        out_dir = args.results_dir
        finetune_dir = args.results_dir
    else:
        out_dir = OUT_DIR
        finetune_dir = FINETUNE_DIR
    os.makedirs(finetune_dir, exist_ok=True)

    # Load pretrained ensemble (or single model)
    if args.ensemble_size > 1:
        ckpt_dir = os.path.dirname(args.checkpoint)
        pretrained_list = load_ensemble(ckpt_dir, prefix="pretrained")
        assert len(pretrained_list) == args.ensemble_size, \
            f"Expected {args.ensemble_size} members, found {len(pretrained_list)}"
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        pretrained_list = [load_params(args.checkpoint)]

    walls, target_cell, target_rc = get_maze_layout(maze_size, args.seed)

    # Evaluate pretrained
    print(f"\n{'='*60}")
    print(f"Pretrained baseline (ensemble_size={len(pretrained_list)})")
    print(f"{'='*60}")
    ens = pretrained_list if len(pretrained_list) > 1 else None
    pretrain_metrics = evaluate_grid(network, pretrained_list[0], walls, target_cell,
                                     maze_size, ensemble_params=ens)
    print(f"  Accuracy: {pretrain_metrics['transition_acc']*100:.1f}%")

    total_steps = args.pref_budget // args.finetune_batch
    print(f"\nPreference budget: {args.pref_budget} pairs "
          f"({total_steps} steps × {args.finetune_batch} batch)")

    # --- Naive finetuning ---
    print(f"\n{'='*60}")
    print(f"Method 1: Naive (random preferences)")
    print(f"{'='*60}")
    rng = jax.random.PRNGKey(args.seed + 100)
    # Deep copy pretrained params for naive
    naive_list = [jax.tree.map(jnp.array, p) for p in pretrained_list]
    naive_list, _, naive_losses, naive_accs, naive_acc_steps = finetune_naive(
        network, naive_list, meta, args, rng)
    ens = naive_list if len(naive_list) > 1 else None
    naive_metrics = evaluate_grid(network, naive_list[0], walls, target_cell,
                                  maze_size, ensemble_params=ens)
    print(f"  Accuracy: {naive_metrics['transition_acc']*100:.1f}%  "
          f"({naive_metrics['n_wrong']} wrong)")

    # --- RENEW finetuning ---
    print(f"\n{'='*60}")
    print(f"Method 2: RENEW (uncertainty-targeted preferences)")
    print(f"{'='*60}")
    rng = jax.random.PRNGKey(args.seed + 200)
    # Deep copy pretrained params for RENEW
    renew_list = [jax.tree.map(jnp.array, p) for p in pretrained_list]
    renew_list, _, renew_losses, renew_accs, renew_acc_steps, renew_unc_grids = \
        finetune_renew(network, renew_list, meta, args, rng)
    ens = renew_list if len(renew_list) > 1 else None
    renew_metrics = evaluate_grid(network, renew_list[0], walls, target_cell,
                                  maze_size, ensemble_params=ens)
    print(f"  Accuracy: {renew_metrics['transition_acc']*100:.1f}%  "
          f"({renew_metrics['n_wrong']} wrong)")

    # --- Save checkpoints ---
    if len(naive_list) > 1:
        save_ensemble(naive_list, finetune_dir, prefix="naive_finetuned")
        save_ensemble(renew_list, finetune_dir, prefix="renew_finetuned")
    else:
        save_params(naive_list[0], os.path.join(finetune_dir, "naive_finetuned.pkl"))
        save_params(renew_list[0], os.path.join(finetune_dir, "renew_finetuned.pkl"))

    # --- Save curves ---
    save_loss_accuracy_curves(
        naive_losses, naive_accs, naive_acc_steps,
        "Naive Finetuning (random preferences)",
        os.path.join(finetune_dir, "naive_curves.png"),
    )
    save_loss_accuracy_curves(
        renew_losses, renew_accs, renew_acc_steps,
        "RENEW Finetuning (uncertainty-targeted preferences)",
        os.path.join(finetune_dir, "renew_curves.png"),
    )

    # --- Save structured results for aggregation ---
    save_results(finetune_dir, "pretrained", pretrain_metrics)
    save_results(finetune_dir, "naive", naive_metrics,
                 losses=naive_losses, accuracies=naive_accs, acc_steps=naive_acc_steps)
    save_results(finetune_dir, "renew", renew_metrics,
                 losses=renew_losses, accuracies=renew_accs, acc_steps=renew_acc_steps)

    # --- Comparison figure ---
    save_comparison_figure(
        pretrain_metrics, naive_metrics, renew_metrics,
        renew_unc_grids, maze_size,
        os.path.join(finetune_dir, "comparison.png"))

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Results (same budget: {args.pref_budget} preference pairs)")
    print(f"{'='*60}")
    print(f"  Pretrained:  {pretrain_metrics['transition_acc']*100:.1f}%  "
          f"({pretrain_metrics['n_wrong']} errors)")
    print(f"  Naive:       {naive_metrics['transition_acc']*100:.1f}%  "
          f"({naive_metrics['n_wrong']} errors)")
    print(f"  RENEW:       {renew_metrics['transition_acc']*100:.1f}%  "
          f"({renew_metrics['n_wrong']} errors)")
    naive_delta = naive_metrics['transition_acc'] - pretrain_metrics['transition_acc']
    renew_delta = renew_metrics['transition_acc'] - pretrain_metrics['transition_acc']
    print(f"  Δ Naive:     {naive_delta*100:+.1f}%")
    print(f"  Δ RENEW:     {renew_delta*100:+.1f}%")
    if renew_delta > naive_delta:
        print(f"  RENEW wins by {(renew_delta - naive_delta)*100:.1f}%")
    elif naive_delta > renew_delta:
        print(f"  Naive wins by {(naive_delta - renew_delta)*100:.1f}%")
    else:
        print(f"  Tied!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(tyro.cli(FinetuneArgs))