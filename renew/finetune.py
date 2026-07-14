"""
renew/finetune.py
=================
Head-to-head comparison of RENEW vs Naive DLHF on a fixed pairwise label budget.
This is the repair experiment (Section 4.3 / Figure 3 / Table 3): finetune a
pretrained world model to repair model exploitation on Jumanji Maze.

Both methods start from the **same** random initialisation (controlled by --seed).
The label budget determines how many steps each method gets:

    naive_steps  = label_budget // batch_size
    renew_steps  = label_budget // (batch_size * (num_candidates - 1))

With K=16, RENEW gets 15x fewer training steps — its advantage must come
from active selection (picking better states to query), not more gradient signal.

Outputs:
    out/compare/{env}/seed_{seed}/
        curves.png          — 3-panel plot (pairwise labels / oracle queries / steps)
        val_curves.npz      — raw data for downstream multi-seed aggregation
        naive/dlhf_*.pkl    — naive checkpoints
        renew/dlhf_*.pkl    — renew checkpoints

Usage (typically driven by renew/run_finetune.sh):
    python renew/finetune.py --checkpoint out/maze10/seed_0/pretrained.pkl \
        --seed 0 --maze-size 10 --pref-budget 1600 --num-rounds 3 --ensemble-size 3
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The core world-model modules (env, network, dlhf) live in ../world_models.
# Add that directory to the import path so this script runs from anywhere.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "world_models"))

from env import collect_val_set, solve_maze_bfs, solve_sliding_tile
from network import LatentCNNNetwork, build_network, init_params, save_params, load_params
from dlhf import (
    DLHFArgs,
    make_meta,
    train_dlhf,
    train_dlhf_renew,
    make_preferences_batch_fn,
    ACTION_SYMS,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


# =============================================================================
# Args
# =============================================================================

@dataclass
class CompareArgs:
    seed:       int = 0
    env:        str = "maze10"
    """Environment: maze{N}, sliding{N}, or sokoban."""

    # --- Budget ---
    label_budget: int = 50_000
    """Total pairwise labels (human comparisons). Both methods share this budget.
    naive_steps = label_budget // batch_size
    renew_steps = label_budget // (batch_size * (num_candidates - 1))"""

    # --- Architecture (passed through to DLHFArgs) ---
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

    # --- Ensemble ---
    ensemble_size: int = 3

    # --- RENEW hyperparams ---
    num_candidates:  int = 4
    pool_multiplier: int = 4
    selection_temp:  float = 1.0
    """Temperature for uncertainty-weighted sampling. Lower = greedier
    (approaches hard top-k). Higher = more exploratory (approaches uniform)."""

    # --- Validation ---
    val_size:     int = 500
    val_scramble: int = 30
    eval_every:   int = 100

    # --- Rollout visualisation ---
    rollout_steps: int = 10

    # --- Pretrained checkpoint (optional, off by default) ---
    pretrain_checkpoint: Optional[str] = None
    """Path to a checkpoint dir. If provided, both methods start from this
    checkpoint instead of random init. Expects dlhf_0.pkl, dlhf_1.pkl, ..."""

    # --- Output ---
    results_dir: Optional[str] = None


# =============================================================================
# Build DLHFArgs from CompareArgs
# =============================================================================

def _make_dlhf_args(cargs: CompareArgs, steps: int, renew: bool,
                    results_subdir: str) -> DLHFArgs:
    return DLHFArgs(
        seed=cargs.seed,
        env=cargs.env,
        embed_dim=cargs.embed_dim,
        latent_channels=cargs.latent_channels,
        enc_layers=cargs.enc_layers,
        dyn_layers=cargs.dyn_layers,
        dec_layers=cargs.dec_layers,
        lr=cargs.lr,
        steps=steps,
        batch_size=cargs.batch_size,
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
    )


# =============================================================================
# Plotting
# =============================================================================

def _steps_to_oracle_queries(steps_arr, batch_size, is_renew):
    """Convert training steps to cumulative oracle queries."""
    queries_per_step = batch_size if is_renew else 2 * batch_size
    return np.array(steps_arr) * queries_per_step


def _steps_to_pairwise_labels(steps_arr, batch_size, is_renew, K):
    """Convert training steps to cumulative pairwise labels."""
    if is_renew:
        labels_per_step = batch_size * (K - 1)
    else:
        labels_per_step = batch_size
    return np.array(steps_arr) * labels_per_step


def save_comparison_plot(
    naive_eval_steps, naive_val_l1s,
    renew_eval_steps, renew_val_l1s,
    batch_size, K, path,
):
    # Auto y-limit: skip first eval point (init spike), use 1.5x max of the rest
    naive_rest = np.array(naive_val_l1s[1:]) if len(naive_val_l1s) > 1 else np.array(naive_val_l1s)
    renew_rest = np.array(renew_val_l1s[1:]) if len(renew_val_l1s) > 1 else np.array(renew_val_l1s)
    y_max = 1.5 * max(naive_rest.max(), renew_rest.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)

    # --- Panel 1: Pairwise labels (primary) ---
    ax = axes[0]
    naive_pl = _steps_to_pairwise_labels(naive_eval_steps, batch_size,
                                          is_renew=False, K=K)
    renew_pl = _steps_to_pairwise_labels(renew_eval_steps, batch_size,
                                          is_renew=True, K=K)
    ax.plot(naive_pl, naive_val_l1s, "o-", label="Naive DLHF", markersize=3,
            linewidth=1.2, color="tab:blue")
    ax.plot(renew_pl, renew_val_l1s, "s-", label="RENEW", markersize=3,
            linewidth=1.2, color="tab:red")
    ax.set_xlabel("Pairwise Labels")
    ax.set_ylabel("Val L1")
    ax.set_title("Pairwise Label Budget (primary)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Oracle queries ---
    ax = axes[1]
    naive_oq = _steps_to_oracle_queries(naive_eval_steps, batch_size, is_renew=False)
    renew_oq = _steps_to_oracle_queries(renew_eval_steps, batch_size, is_renew=True)
    ax.plot(naive_oq, naive_val_l1s, "o-", label="Naive DLHF", markersize=3,
            linewidth=1.2, color="tab:blue")
    ax.plot(renew_oq, renew_val_l1s, "s-", label="RENEW", markersize=3,
            linewidth=1.2, color="tab:red")
    ax.set_xlabel("Oracle Queries")
    ax.set_ylabel("Val L1")
    ax.set_title("Oracle Query Budget")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Raw training steps ---
    ax = axes[2]
    ax.plot(naive_eval_steps, naive_val_l1s, "o-", label="Naive DLHF", markersize=3,
            linewidth=1.2, color="tab:blue")
    ax.plot(renew_eval_steps, renew_val_l1s, "s-", label="RENEW", markersize=3,
            linewidth=1.2, color="tab:red")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Val L1")
    ax.set_title("Training Steps (unequal budget)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("RENEW vs Naive DLHF", fontsize=12, y=1.02)
    for ax in axes:
        ax.set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot → {path}")


# =============================================================================
# Rollout comparison visualisation: GT vs Naive vs RENEW
# =============================================================================

def _to_rgb(arr):
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def _model_rollout(network, params, meta, start_obs, actions, gt_states, env):
    """Roll out a trained model from start_obs using the given actions."""
    T = len(actions)

    encode_jit = jax.jit(lambda p, o: network.apply(
        p, o[None], method=LatentCNNNetwork.encode))
    step_lat_jit = jax.jit(lambda p, z, a: network.apply(
        p, z, jnp.array([a]), method=LatentCNNNetwork.step_latent))
    decode_jit = jax.jit(lambda p, z, a: network.apply(
        p, z, jnp.array([a]), method=LatentCNNNetwork.decode_latent)[0])

    wm_states = [gt_states[0]]
    z = encode_jit(params, start_obs)
    for t in range(T):
        z = step_lat_jit(params, z, actions[t])
        obs_pred = decode_jit(params, z, actions[t])
        fake_state = meta.obs_to_state(obs_pred, gt_states[t + 1])
        wm_states.append(fake_state)
        z = encode_jit(params, obs_pred)

    return wm_states


def save_comparison_rollout(network, naive_params, renew_params, meta, cargs, rd):
    """
    Generate GT vs Naive vs RENEW rollout visualisations:
      - rollout.gif      (3-way side-by-side GIF)
      - comparison.png   (3-row static comparison strip)
    """
    import imageio
    from PIL import Image

    T   = cargs.rollout_steps
    env = meta.make_env()

    # --- Generate start state and action sequence ---
    rng = jax.random.PRNGKey(cargs.seed + 777)
    rng, rk = jax.random.split(rng)
    state, _ = jax.jit(env.reset)(rk)
    start_obs = meta.extract_obs(jax.tree.map(lambda x: x[None], state))[0]

    cols = meta.grid_shape[1]
    if cargs.env.startswith("maze"):
        solver_actions = solve_maze_bfs(start_obs, cols)
    elif cargs.env.startswith("sliding"):
        solver_actions = solve_sliding_tile(start_obs, meta.grid_shape[0])
    else:
        solver_actions = list(np.random.default_rng(cargs.seed).integers(0, meta.num_actions, T))
    if len(solver_actions) < T:
        extra = list(np.random.default_rng(cargs.seed).integers(
            0, meta.num_actions, T - len(solver_actions)))
        solver_actions = solver_actions + extra
    actions = np.array(solver_actions[:T], dtype=np.int32)

    # --- Ground truth rollout ---
    step_jit = jax.jit(env.step)
    gt_states = [state]
    s = state
    for t in range(T):
        s, _ = step_jit(s, jnp.array(actions[t]))
        gt_states.append(s)

    # --- Model rollouts ---
    print("  Rolling out Naive model...")
    naive_states = _model_rollout(network, naive_params, meta, start_obs,
                                   actions, gt_states, env)
    print("  Rolling out RENEW model...")
    renew_states = _model_rollout(network, renew_params, meta, start_obs,
                                   actions, gt_states, env)

    # --- Render GIFs ---
    gt_gif_path    = os.path.join(rd, "gt.gif")
    naive_gif_path = os.path.join(rd, "naive.gif")
    renew_gif_path = os.path.join(rd, "renew.gif")

    print("  Rendering GT...")
    env.animate(gt_states, interval=300, save_path=gt_gif_path)
    print("  Rendering Naive...")
    env.animate(naive_states, interval=300, save_path=naive_gif_path)
    print("  Rendering RENEW...")
    env.animate(renew_states, interval=300, save_path=renew_gif_path)

    gt_frames    = [_to_rgb(f) for f in imageio.mimread(gt_gif_path, memtest=False)]
    naive_frames = [_to_rgb(f) for f in imageio.mimread(naive_gif_path, memtest=False)]
    renew_frames = [_to_rgb(f) for f in imageio.mimread(renew_gif_path, memtest=False)]

    # Pad to same length
    n = max(len(gt_frames), len(naive_frames), len(renew_frames))
    gt_frames    += [gt_frames[-1]]    * (n - len(gt_frames))
    naive_frames += [naive_frames[-1]] * (n - len(naive_frames))
    renew_frames += [renew_frames[-1]] * (n - len(renew_frames))

    # Resize to match GT dimensions
    h_target = gt_frames[0].shape[0]
    w_target = gt_frames[0].shape[1]

    def _resize(img):
        if img.shape[0] == h_target and img.shape[1] == w_target:
            return img
        return np.array(Image.fromarray(img).resize(
            (w_target, h_target), Image.LANCZOS))

    naive_frames = [_resize(f) for f in naive_frames]
    renew_frames = [_resize(f) for f in renew_frames]

    # --- Side-by-side GIF (GT | Naive | RENEW) ---
    PAD = 4
    combined_frames = []
    for gt_f, naive_f, renew_f in zip(gt_frames, naive_frames, renew_frames):
        sep = np.full((h_target, PAD, 3), 40, dtype=np.uint8)
        combined_frames.append(
            np.concatenate([gt_f, sep, naive_f, sep, renew_f], axis=1))

    rollout_gif_path = os.path.join(rd, "rollout.gif")
    imageio.mimsave(rollout_gif_path, combined_frames, duration=300, loop=0)
    print(f"  Rollout GIF → {rollout_gif_path}")

    # --- Static comparison strip (3 rows × N columns) ---
    n_show = min(len(gt_frames), 12)
    if n_show <= 1:
        return
    indices = np.linspace(0, len(gt_frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.5, 5), dpi=130)
    if n_show == 1:
        axes = axes.reshape(3, 1)

    row_data = [
        (gt_frames,    "GT"),
        (naive_frames, "Naive"),
        (renew_frames, "RENEW"),
    ]

    for col, t in enumerate(indices):
        for row, (frame_list, label) in enumerate(row_data):
            ax = axes[row, col]
            ax.imshow(frame_list[t])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=40,
                              va="center", fontweight="bold")
            if row == 0:
                if t == 0:
                    ax.set_title("t=0", fontsize=8, pad=3)
                elif t - 1 < len(actions):
                    sym = ACTION_SYMS.get(int(actions[t - 1]), str(actions[t - 1]))
                    ax.set_title(f"t={t} ({sym})", fontsize=8, pad=3)

        # Highlight errors: red border on Naive/RENEW where they differ from GT
        for row_idx, frame_list in [(1, naive_frames), (2, renew_frames)]:
            if not np.array_equal(gt_frames[t], frame_list[t]):
                for spine in axes[row_idx, col].spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)

    fig.suptitle(f"GT vs Naive vs RENEW — {cargs.env} — seed {cargs.seed}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    cmp_path = os.path.join(rd, "comparison.png")
    fig.savefig(cmp_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison → {cmp_path}")


# =============================================================================
# Main
# =============================================================================

def main(cargs: CompareArgs) -> None:
    B = cargs.batch_size
    K = cargs.num_candidates
    M = cargs.ensemble_size

    naive_steps = cargs.label_budget // B
    renew_steps = cargs.label_budget // (B * (K - 1))

    print(f"\n{'='*60}")
    print(f"  COMPARE: RENEW vs Naive DLHF")
    print(f"{'='*60}")
    print(f"  Env:            {cargs.env}")
    print(f"  Label budget:   {cargs.label_budget}")
    print(f"  Batch size:     {B}")
    print(f"  Ensemble:       {M}")
    print(f"  RENEW K:        {K}")
    print(f"  Naive steps:    {naive_steps}  ({naive_steps * B} labels)")
    print(f"  RENEW steps:    {renew_steps}  ({renew_steps * B * (K - 1)} labels)")
    print(f"  Naive oracle:   {naive_steps * 2 * B} queries")
    print(f"  RENEW oracle:   {renew_steps * B} queries")

    # --- Output directory ---
    if cargs.results_dir:
        rd = cargs.results_dir
    else:
        rd = os.path.join(OUT_DIR, "compare", cargs.env, f"seed_{cargs.seed}")
    os.makedirs(rd, exist_ok=True)
    naive_dir = os.path.join(rd, "naive")
    renew_dir = os.path.join(rd, "renew")
    os.makedirs(naive_dir, exist_ok=True)
    os.makedirs(renew_dir, exist_ok=True)

    # --- Environment + network ---
    meta    = make_meta(_make_dlhf_args(cargs, 0, False, naive_dir))
    network = build_network(meta, _make_dlhf_args(cargs, 0, True, renew_dir))

    # --- Shared init (same random params for both methods) ---
    opt = optax.adam(cargs.lr)
    if cargs.pretrain_checkpoint:
        print(f"\n  Loading pretrained checkpoint: {cargs.pretrain_checkpoint}")
        all_params = []
        for m in range(M):
            path = os.path.join(cargs.pretrain_checkpoint, f"dlhf_{m}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            all_params.append(load_params(path))
        print(f"  Loaded {M} member(s)")
    else:
        all_params = []
        for m in range(M):
            rng = jax.random.PRNGKey(cargs.seed + m * 1000)
            p = init_params(network, meta, cargs.batch_size, cargs.context_len, rng)
            all_params.append(p)
        print(f"\n  Initialised {M} LatentCNN member(s) (shared across methods)")

    # Deep copy so both methods start from identical params
    all_params_naive = all_params
    all_params_renew = jax.tree.map(lambda x: x.copy(), all_params)

    # --- Shared validation set ---
    val_set = collect_val_set(meta, cargs.val_size, cargs.val_scramble, cargs.seed)

    # =====================================================================
    # Run Naive DLHF
    # =====================================================================
    naive_args = _make_dlhf_args(cargs, naive_steps, renew=False,
                                  results_subdir=naive_dir)

    stacked_p = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params_naive)
    stacked_o = jax.tree.map(lambda *os: jnp.stack(os),
                              *[opt.init(p) for p in all_params_naive])

    gen_batch = make_preferences_batch_fn(
        meta, cargs.batch_size, cargs.scramble_steps, cargs.horizon)

    train_rng = jax.random.PRNGKey(cargs.seed + 123)
    t0 = time.time()
    stacked_p_naive, _, naive_losses, naive_val_l1s, naive_eval_steps = \
        train_dlhf(network, stacked_p, stacked_o, gen_batch,
                   val_set, naive_args, train_rng, ensemble_size=M)
    naive_time = time.time() - t0

    for i in range(M):
        p = jax.tree.map(lambda x: x[i], stacked_p_naive)
        save_params(p, os.path.join(naive_dir, f"dlhf_{i}.pkl"))

    # =====================================================================
    # Run RENEW
    # =====================================================================
    renew_args = _make_dlhf_args(cargs, renew_steps, renew=True,
                                  results_subdir=renew_dir)

    stacked_p = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params_renew)
    stacked_o = jax.tree.map(lambda *os: jnp.stack(os),
                              *[opt.init(p) for p in all_params_renew])

    train_rng = jax.random.PRNGKey(cargs.seed + 123)
    t0 = time.time()
    stacked_p_renew, _, renew_losses, renew_val_l1s, renew_eval_steps, renew_extra = \
        train_dlhf_renew(network, stacked_p, stacked_o,
                         meta, val_set, renew_args, train_rng, ensemble_size=M)
    renew_time = time.time() - t0

    for i in range(M):
        p = jax.tree.map(lambda x: x[i], stacked_p_renew)
        save_params(p, os.path.join(renew_dir, f"dlhf_{i}.pkl"))

    # =====================================================================
    # Results summary
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"  RESULTS  (label budget = {cargs.label_budget})")
    print(f"{'='*60}")
    print(f"  Naive DLHF:  final val_l1 = {naive_val_l1s[-1]:.4f}  "
          f"({naive_time:.1f}s, {naive_steps} steps)")
    print(f"  RENEW:       final val_l1 = {renew_val_l1s[-1]:.4f}  "
          f"({renew_time:.1f}s, {renew_steps} steps)")
    gap = naive_val_l1s[-1] - renew_val_l1s[-1]
    print(f"  Gap (naive - renew): {gap:.4f}  "
          f"({'RENEW wins' if gap > 0 else 'Naive wins'})")

    # =====================================================================
    # Save raw data for multi-seed aggregation
    # =====================================================================
    npz_path = os.path.join(rd, "val_curves.npz")
    np.savez(
        npz_path,
        naive_eval_steps=np.array(naive_eval_steps),
        naive_val_l1s=np.array(naive_val_l1s),
        naive_losses=np.array(naive_losses),
        renew_eval_steps=np.array(renew_eval_steps),
        renew_val_l1s=np.array(renew_val_l1s),
        renew_losses=np.array(renew_losses),
        renew_pref_accs=np.array(renew_extra.get("pref_accs", [])),
        oracle_budget=cargs.label_budget,  # legacy key name kept for compat
        label_budget=cargs.label_budget,
        batch_size=B,
        num_candidates=K,
        ensemble_size=M,
        seed=cargs.seed,
        env=cargs.env,
    )
    print(f"  Raw data → {npz_path}")

    # =====================================================================
    # Comparison plot
    # =====================================================================
    save_comparison_plot(
        naive_eval_steps, naive_val_l1s,
        renew_eval_steps, renew_val_l1s,
        B, K,
        os.path.join(rd, "curves.png"),
    )

    # =====================================================================
    # Rollout comparison: GT vs Naive vs RENEW
    # =====================================================================
    print("\nGenerating rollout comparison...")
    naive_p0 = jax.tree.map(lambda x: x[0], stacked_p_naive)
    renew_p0 = jax.tree.map(lambda x: x[0], stacked_p_renew)
    save_comparison_rollout(network, naive_p0, renew_p0, meta, cargs, rd)

    print("Done!")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))