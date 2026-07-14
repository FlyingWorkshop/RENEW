"""
world_models/dlhf.py
====================
Dynamics Learning from Human Feedback.

Two training modes:
  1. Standard DLHF  (--use-renew False, default)
     Random preference pairs with Bradley-Terry loss.
  2. RENEW Active    (--use-renew True)
     Ensemble disagreement → active (s0, actions) selection →
     K-candidate sampling → K-1 BT pairs per oracle query.

Usage:
  # Standard DLHF
  python world_models/dlhf.py --env sokoban --steps 50000

  # RENEW with ensemble + K-candidate
  python world_models/dlhf.py --env sokoban --steps 50000 \
      --use-renew --ensemble-size 5 --num-candidates 4 --pool-multiplier 4
"""

from __future__ import annotations

import os
import re
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

from env import (
    make_maze_meta, make_sliding_tile_meta, make_sokoban_meta,
    make_2048_meta, make_connector_meta, make_pacman_meta,
    collect_val_set, make_preferences_batch_fn,
    make_active_pool_fn, make_oracle_rollout_fn,
    solve_maze_bfs, solve_sliding_tile, solve_pacman_step,
)
from network import LatentCNNNetwork, build_network, init_params, save_params, load_params

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")

ACTION_SYMS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


@dataclass
class DLHFArgs:
    seed:       int = 0
    env:        str = "maze10"
    """Environment: maze{N}, sliding{N}, sokoban, 2048, connector{N}[_{agents}], or pacman."""

    # --- Architecture ---
    embed_dim:       int = 128
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 4
    dec_layers:      int = 3

    # --- Training ---
    lr:             float = 3e-4
    steps:          int   = 25_000
    batch_size:     int   = 64
    horizon:        int   = 1
    """Rollout horizon for preference comparisons."""
    beta_bt:        float = 1.0
    exclude_ties:   bool = True
    scramble_steps: int   = 0
    context_len:    int   = 10
    """Context length (only used for param init shape)."""

    # --- Ensemble ---
    ensemble_size: int = 3

    # --- RENEW active learning ---
    renew:       bool = True
    """Enable RENEW: active selection + K-candidate preference pairs."""
    num_candidates:  int  = 4
    """K: number of candidate rollouts per (s0, actions). Forms K-1 BT pairs."""
    pool_multiplier: int  = 4
    """Pool size = batch_size * pool_multiplier. Larger pools ≈ better selection."""
    selection_temp:  float = 1.0
    """Temperature for uncertainty-weighted sampling. Lower = greedier
    (approaches hard top-k). Higher = more exploratory (approaches uniform)."""
    recompute_every: int  = 0
    """Recompute uncertainty every N steps (0 = every step, which is the default
    since scoring is inside the jitted step). Set >0 to amortise with a cached
    weighting — but for the jitted loop, 0 is fastest."""

    # --- Validation ---
    val_size:     int = 500
    val_scramble: int = 30
    eval_every:   int = 100

    # --- Rollout visualisation ---
    rollout_steps: int = 10

    # --- Output ---
    results_dir: Optional[str] = None
    """Override output directory. Default: out/dlhf/{env}/seed_{seed}/"""

    # --- Resume ---
    resume: Optional[str] = None
    """Path to checkpoint dir to resume from (loads dlhf_0.pkl, dlhf_1.pkl, ...)."""


# =============================================================================
# Output directory
# =============================================================================

def run_dir(args: DLHFArgs) -> str:
    if args.results_dir:
        d = args.results_dir
    else:
        mode = "renew" if args.renew else "dlhf"
        d = os.path.join(OUT_DIR, mode, args.env, f"seed_{args.seed}")
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# Training curves
# =============================================================================

def save_curves(losses, val_l1s, eval_steps, path, extra_metrics=None):
    n_plots = 2 + (1 if extra_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), dpi=120)

    axes[0].plot(losses, linewidth=0.6, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("BT Loss")
    axes[0].set_title(f"Loss — final: {losses[-1]:.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_steps, val_l1s, "o-", linewidth=1, markersize=3, color="tab:red")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Val L1")
    axes[1].set_title(f"Val L1 — final: {val_l1s[-1]:.4f}")
    axes[1].grid(True, alpha=0.3)

    if extra_metrics and "pref_accs" in extra_metrics:
        pa = extra_metrics["pref_accs"]
        axes[2].plot(pa, linewidth=0.6, color="tab:green", alpha=0.8)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Pref Accuracy")
        axes[2].set_title(f"Pref Acc — final: {pa[-1]:.3f}")
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.5, ls="--", color="gray", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves → {path}")


# =============================================================================
# Rollout visualisation (unchanged from original)
# =============================================================================

def _to_rgb(arr):
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def save_rollout(network, params, meta, args):
    import imageio
    from PIL import Image

    rd  = run_dir(args)
    T   = args.rollout_steps
    env = meta.make_env()

    rng = jax.random.PRNGKey(args.seed + 777)
    rng, rk = jax.random.split(rng)
    state, _ = jax.jit(env.reset)(rk)
    start_obs = meta.extract_obs(jax.tree.map(lambda x: x[None], state))[0]

    cols = meta.grid_shape[1]
    if args.env == "pacman":
        # Reactive policy: compute action from current obs each step
        step_jit  = jax.jit(env.step)
        extract_single = lambda s: meta.extract_obs(jax.tree.map(lambda x: x[None], s))[0]
        gt_states = [state]
        actions_list = []
        s = state
        for t in range(T):
            obs_t = np.array(extract_single(s))
            act = solve_pacman_step(obs_t, meta.grid_shape[0], meta.grid_shape[1])
            actions_list.append(act)
            s, _ = step_jit(s, jnp.array(act))
            gt_states.append(s)
        actions = np.array(actions_list, dtype=np.int32)
    else:
        if args.env.startswith("maze"):
            solver_actions = solve_maze_bfs(start_obs, cols)
        elif args.env.startswith("sliding"):
            solver_actions = solve_sliding_tile(start_obs, meta.grid_shape[0])
        else:
            solver_actions = list(np.random.default_rng(args.seed).integers(0, meta.num_actions, T))
        if len(solver_actions) < T:
            extra = list(np.random.default_rng(args.seed).integers(0, meta.num_actions, T - len(solver_actions)))
            solver_actions = solver_actions + extra
        actions = np.array(solver_actions[:T], dtype=np.int32)

        step_jit  = jax.jit(env.step)
        gt_states = [state]
        s = state
        for t in range(T):
            s, _ = step_jit(s, jnp.array(actions[t]))
            gt_states.append(s)

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
        # Re-encode decoded obs to avoid pure latent-space drift
        z = encode_jit(params, obs_pred)

    print("  Rendering GT...")
    gt_gif = os.path.join(rd, "gt.gif")
    env.animate(gt_states, interval=300, save_path=gt_gif)

    print("  Rendering WM...")
    wm_gif = os.path.join(rd, "wm.gif")
    env.animate(wm_states, interval=300, save_path=wm_gif)

    gt_frames = [_to_rgb(f) for f in imageio.mimread(gt_gif, memtest=False)]
    wm_frames = [_to_rgb(f) for f in imageio.mimread(wm_gif, memtest=False)]

    n = max(len(gt_frames), len(wm_frames))
    gt_frames += [gt_frames[-1]] * (n - len(gt_frames))
    wm_frames += [wm_frames[-1]] * (n - len(wm_frames))

    h_target = gt_frames[0].shape[0]
    w_target = gt_frames[0].shape[1]
    def _resize(img):
        if img.shape[0] == h_target and img.shape[1] == w_target:
            return img
        return np.array(Image.fromarray(img).resize((w_target, h_target), Image.LANCZOS))
    wm_frames = [_resize(f) for f in wm_frames]

    PAD = 4
    side_frames = []
    for gt_f, wm_f in zip(gt_frames, wm_frames):
        sep = np.full((h_target, PAD, 3), 40, dtype=np.uint8)
        side_frames.append(np.concatenate([gt_f, sep, wm_f], axis=1))

    side_path = os.path.join(rd, "rollout.gif")
    imageio.mimsave(side_path, side_frames, duration=300, loop=0)
    print(f"  Rollout GIF → {side_path}")

    n_show = min(len(gt_frames), 12)
    if n_show <= 1:
        return
    indices = np.linspace(0, len(gt_frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3.4), dpi=130)
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col, t in enumerate(indices):
        for row, (frame_list, label) in enumerate(
                [(gt_frames, "True"), (wm_frames, "Model")]):
            ax = axes[row, col]
            ax.imshow(frame_list[t])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=35,
                              va="center", fontweight="bold")
            if row == 0:
                if t == 0:
                    ax.set_title("t=0", fontsize=8, pad=3)
                elif t - 1 < len(actions):
                    sym = ACTION_SYMS.get(int(actions[t - 1]), str(actions[t - 1]))
                    ax.set_title(f"t={t} ({sym})", fontsize=8, pad=3)

        if not np.array_equal(gt_frames[t], wm_frames[t]):
            for spine in axes[1, col].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

    fig.suptitle(f"DLHF — {args.env} — seed {args.seed}", fontsize=11, y=1.02)
    fig.tight_layout()
    cmp_path = os.path.join(rd, "comparison.png")
    fig.savefig(cmp_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison → {cmp_path}")


# =============================================================================
# Ensemble uncertainty scoring for active selection
# =============================================================================

def _build_score_pool_fn(network, ensemble_size):
    """
    Returns a function that scores a pool of (s0, actions) candidates by
    model uncertainty.

    M > 1: mutual information  (ensemble disagreement)
    M = 1: predictive entropy  (single model softmax entropy)

    The returned function has signature:
        score_pool(stacked_params, start_obs, actions) -> scores  (P,)
    and is designed to be called inside a @jax.jit context.
    """
    M = ensemble_size

    def score_pool(stacked_params, start_obs, actions):
        """
        Args:
            stacked_params: (M, ...) pytree
            start_obs:      (P, obs_dim)
            actions:        (P, H)
        Returns:
            uncertainty:    (P,) higher = more uncertain
        """
        def _member_logits(params):
            return network.apply(params, start_obs, actions,
                                 method=LatentCNNNetwork.rollout_logits)
            # returns (P, H, obs_dim, num_tiles)

        # Logits from all ensemble members: (M, P, H, obs_dim, tiles)
        all_logits = jax.vmap(_member_logits)(stacked_params)
        probs = jax.nn.softmax(all_logits, axis=-1)
        mean_probs = probs.mean(axis=0)           # (P, H, obs_dim, tiles)

        # Entropy of the mean prediction
        ent_mean = -(mean_probs * jnp.log(mean_probs + 1e-10)).sum(-1)
        # (P, H, obs_dim)

        if M > 1:
            # Mutual information = H[E[p]] - E[H[p]]
            ent_each = -(probs * jnp.log(probs + 1e-10)).sum(-1)
            # (M, P, H, obs_dim)
            mean_ent = ent_each.mean(axis=0)       # (P, H, obs_dim)
            uncertainty = (ent_mean - mean_ent).mean((-1, -2))   # (P,)
        else:
            # Single model: use total predictive entropy
            uncertainty = ent_mean.mean((-1, -2))   # (P,)

        return uncertainty

    return score_pool


# =============================================================================
# Standard DLHF training loop (unchanged)
# =============================================================================

def train_dlhf(network, stacked_params, stacked_opt_state, gen_batch,
               val_set, args, rng, ensemble_size=1):
    eval_every = args.eval_every
    opt = optax.adam(args.lr)

    @jax.jit
    def _val_l1(params):
        return network.apply(
            params, val_set["obs"], val_set["action"], val_set["gt_next"],
            method=LatentCNNNetwork.val_l1)

    def _eval(stacked_params):
        p0 = jax.tree.map(lambda x: x[0], stacked_params)
        return float(_val_l1(p0))

    def _member_step(params, opt_state, batch, sk):
        def loss_fn(p):
            return network.apply(
                p, batch["start_obs"], batch["acts1"], batch["acts2"],
                batch["gt1"], batch["gt2"], sk,
                method=LatentCNNNetwork.preferences_loss)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info

    @jax.jit
    def _ensemble_step(stacked_params, stacked_opt_state, rng):
        rng, bk, sk = jax.random.split(rng, 3)
        batch = gen_batch(bk)
        new_p, new_o, info = jax.vmap(
            lambda p, o: _member_step(p, o, batch, sk)
        )(stacked_params, stacked_opt_state)
        return new_p, new_o, rng, info

    @jax.jit
    def _scan_chunk(carry, _):
        sp, so, rng = carry
        sp, so, rng, info = _ensemble_step(sp, so, rng)
        return (sp, so, rng), info["bt_loss"][0]

    @jax.jit
    def _run_chunk(sp, so, rng):
        (sp, so, rng), losses = jax.lax.scan(
            _scan_chunk, (sp, so, rng), None, length=eval_every)
        return sp, so, rng, losses

    label = f"DLHF ({ensemble_size} members)" if ensemble_size > 1 else "DLHF"
    total = args.steps
    print(f"\n{'='*60}")
    print(f"{label} — {total} steps, H={args.horizon}, eval every {eval_every}")
    print(f"{'='*60}")

    all_losses = []
    val_l1s    = []
    eval_steps = []

    val_l1 = _eval(stacked_params)
    val_l1s.append(val_l1)
    eval_steps.append(0)
    print(f"  init   | val_l1={val_l1:.4f}")

    t0 = time.time()
    steps_done = 0
    n_full = total // eval_every
    remainder = total % eval_every

    for _ in range(n_full):
        stacked_params, stacked_opt_state, rng, chunk_losses = _run_chunk(
            stacked_params, stacked_opt_state, rng)
        jax.block_until_ready(stacked_params)
        steps_done += eval_every
        all_losses.extend(np.array(chunk_losses).tolist())

        val_l1 = _eval(stacked_params)
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        loss = float(chunk_losses[-1])
        print(f"  step {steps_done:5d} | bt_loss={loss:.4f} | val_l1={val_l1:.4f} | "
              f"{time.time()-t0:.1f}s")

    if remainder > 0:
        for _ in range(remainder):
            stacked_params, stacked_opt_state, rng, info = _ensemble_step(
                stacked_params, stacked_opt_state, rng)
            all_losses.append(float(info["bt_loss"][0]))
        steps_done += remainder
        jax.block_until_ready(stacked_params)
        val_l1 = _eval(stacked_params)
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        print(f"  step {steps_done:5d} | val_l1={val_l1:.4f} | {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  Done — {elapsed:.1f}s ({elapsed/total*1000:.1f} ms/step)")
    return stacked_params, rng, all_losses, val_l1s, eval_steps


# =============================================================================
# RENEW training loop: active selection + K-candidate preferences
# =============================================================================

def train_dlhf_renew(network, stacked_params, stacked_opt_state,
                     meta, val_set, args, rng, ensemble_size=1):
    """
    RENEW training loop.

    Each step:
      1. Generate a pool of P = B * pool_multiplier candidate (s0, actions) pairs.
      2. Score every candidate by ensemble disagreement (MI) or predictive
         entropy (single model).
      3. Sample the top-B highest-uncertainty candidates using softmax-weighted
         selection (controlled by selection_temp).
      4. Query oracle for GT trajectories on the selected (s0, actions).
      5. Generate K candidate rollouts from the model for each selected pair.
      6. Pick the best candidate (closest to GT), form K-1 BT preference pairs.
      7. Update all ensemble members on the resulting loss (via vmap).

    The entire step (pool gen → scoring → selection → oracle → K-candidate
    loss → param update) is compiled into a single XLA program via @jax.jit.
    """
    eval_every = args.eval_every
    opt = optax.adam(args.lr)
    B = args.batch_size
    P = B * args.pool_multiplier
    M = ensemble_size
    K = args.num_candidates
    H = args.horizon
    selection_temp = args.selection_temp

    # --- Build sub-components ---
    gen_pool      = make_active_pool_fn(meta, P, H, args.scramble_steps)
    oracle_rollout = make_oracle_rollout_fn(meta, B, H)
    score_pool_fn = _build_score_pool_fn(network, M)

    @jax.jit
    def _val_l1(params):
        return network.apply(
            params, val_set["obs"], val_set["action"], val_set["gt_next"],
            method=LatentCNNNetwork.val_l1)

    def _eval(stacked_params):
        p0 = jax.tree.map(lambda x: x[0], stacked_params)
        return float(_val_l1(p0))

    # --- Single RENEW step (fully jitted) ---
    def _member_step_renew(params, opt_state, sel_obs, sel_acts, gt, sk):
        def loss_fn(p):
            return network.apply(
                p, sel_obs, sel_acts, gt, sk,
                method=LatentCNNNetwork.k_candidate_preferences_loss)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, info

    @jax.jit
    def _renew_step(stacked_params, stacked_opt_state, rng):
        rng, pool_key, score_key, oracle_key, loss_key = jax.random.split(rng, 5)

        # 1. Generate candidate pool
        pool_obs, pool_acts = gen_pool(pool_key)         # (P, obs_dim), (P, H)

        # 2. Score by ensemble uncertainty
        scores = score_pool_fn(stacked_params, pool_obs, pool_acts)  # (P,)

        # 3. Select top-B most uncertain
        _, top_idx = jax.lax.top_k(scores, B)
        sel_obs  = pool_obs[top_idx]                     # (B, obs_dim)
        sel_acts = pool_acts[top_idx]                     # (B, H)

        # 4. Oracle ground truth
        gt = oracle_rollout(sel_obs, sel_acts, oracle_key)  # (B, H, obs_dim)

        # 5-6. K-candidate loss + update (vmapped over ensemble members)
        new_p, new_o, info = jax.vmap(
            lambda p, o: _member_step_renew(p, o, sel_obs, sel_acts, gt, loss_key)
        )(stacked_params, stacked_opt_state)

        return new_p, new_o, rng, info

    # --- Scan-compatible wrapper for chunked evaluation ---
    @jax.jit
    def _scan_chunk_renew(carry, _):
        sp, so, rng = carry
        sp, so, rng, info = _renew_step(sp, so, rng)
        return (sp, so, rng), (info["bt_loss"][0], info["pref_acc"][0])

    @jax.jit
    def _run_chunk_renew(sp, so, rng):
        (sp, so, rng), (losses, accs) = jax.lax.scan(
            _scan_chunk_renew, (sp, so, rng), None, length=eval_every)
        return sp, so, rng, losses, accs

    # --- Training ---
    label = f"RENEW (M={M}, K={K}, pool={P}, temp={selection_temp})"
    total = args.steps
    print(f"\n{'='*60}")
    print(f"{label} — {total} steps, H={H}, eval every {eval_every}")
    print(f"{'='*60}")

    all_losses   = []
    all_pref_acc = []
    val_l1s      = []
    eval_steps   = []

    val_l1 = _eval(stacked_params)
    val_l1s.append(val_l1)
    eval_steps.append(0)
    print(f"  init   | val_l1={val_l1:.4f}")

    t0 = time.time()
    steps_done = 0
    n_full = total // eval_every
    remainder = total % eval_every

    for _ in range(n_full):
        stacked_params, stacked_opt_state, rng, chunk_losses, chunk_accs = \
            _run_chunk_renew(stacked_params, stacked_opt_state, rng)
        jax.block_until_ready(stacked_params)
        steps_done += eval_every
        all_losses.extend(np.array(chunk_losses).tolist())
        all_pref_acc.extend(np.array(chunk_accs).tolist())

        val_l1 = _eval(stacked_params)
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        loss = float(chunk_losses[-1])
        acc  = float(chunk_accs[-1])
        print(f"  step {steps_done:5d} | bt_loss={loss:.4f} | pref_acc={acc:.3f} | "
              f"val_l1={val_l1:.4f} | {time.time()-t0:.1f}s")

    if remainder > 0:
        for _ in range(remainder):
            stacked_params, stacked_opt_state, rng, info = _renew_step(
                stacked_params, stacked_opt_state, rng)
            all_losses.append(float(info["bt_loss"][0]))
            all_pref_acc.append(float(info["pref_acc"][0]))
        steps_done += remainder
        jax.block_until_ready(stacked_params)
        val_l1 = _eval(stacked_params)
        val_l1s.append(val_l1)
        eval_steps.append(steps_done)
        print(f"  step {steps_done:5d} | val_l1={val_l1:.4f} | {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  Done — {elapsed:.1f}s ({elapsed/total*1000:.1f} ms/step)")

    extra = dict(pref_accs=all_pref_acc)
    return stacked_params, rng, all_losses, val_l1s, eval_steps, extra


# =============================================================================
# Entry point
# =============================================================================

def make_meta(args):
    """Create the right EnvMeta based on args.env."""
    if args.env == "sokoban":
        return make_sokoban_meta()
    if args.env == "2048":
        return make_2048_meta()
    if args.env == "pacman":
        return make_pacman_meta()

    # Connector: connector{grid_size} or connector{grid_size}_{num_agents}
    m_conn = re.match(r"connector(\d+)(?:_(\d+))?$", args.env)
    if m_conn:
        grid_size = int(m_conn.group(1))
        num_agents = int(m_conn.group(2)) if m_conn.group(2) else 3
        return make_connector_meta(grid_size, num_agents)

    # Maze / Sliding tile
    m = re.match(r"(maze|sliding)(\d+)$", args.env)
    if not m:
        raise ValueError(
            f"Unknown env: {args.env}. "
            f"Use maze{{N}}, sliding{{N}}, sokoban, 2048, connector{{N}}[_{{agents}}], or pacman.")
    kind, size = m.group(1), int(m.group(2))
    if kind == "maze":
        return make_maze_meta(size)
    else:
        return make_sliding_tile_meta(size)


def main(args: DLHFArgs) -> None:
    meta    = make_meta(args)
    network = build_network(meta, args)
    M       = args.ensemble_size
    rd      = run_dir(args)

    # Init or resume ensemble
    opt = optax.adam(args.lr)
    if args.resume:
        all_params = []
        for m in range(M):
            path = os.path.join(args.resume, f"dlhf_{m}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            all_params.append(load_params(path))
        all_opts = [opt.init(p) for p in all_params]
        print(f"Resumed {M} member(s) from {args.resume}")
    else:
        all_params, all_opts = [], []
        for m in range(M):
            rng = jax.random.PRNGKey(args.seed + m * 1000)
            p = init_params(network, meta, args.batch_size, args.context_len, rng)
            all_params.append(p)
            all_opts.append(opt.init(p))
        print(f"Initialised {M} LatentCNN member(s)")

    stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opts)

    val_set = collect_val_set(meta, args.val_size, args.val_scramble, args.seed)
    train_rng = jax.random.PRNGKey(args.seed + 123)

    # ---- Dispatch to standard DLHF or RENEW ----
    if args.renew:
        print(f"\n  Mode: RENEW (K={args.num_candidates}, pool×{args.pool_multiplier}, "
              f"M={M}, temp={args.selection_temp})")
        stacked_params, _, all_losses, val_l1s, eval_steps, extra = \
            train_dlhf_renew(
                network, stacked_params, stacked_opt_state,
                meta, val_set, args, train_rng, ensemble_size=M,
            )
    else:
        print(f"\n  Mode: Standard DLHF")
        gen_batch = make_preferences_batch_fn(
            meta, args.batch_size, args.scramble_steps, args.horizon,
        )
        stacked_params, _, all_losses, val_l1s, eval_steps = train_dlhf(
            network, stacked_params, stacked_opt_state, gen_batch,
            val_set, args, train_rng, ensemble_size=M,
        )
        extra = None

    # Save params
    for i in range(M):
        p = jax.tree.map(lambda x: x[i], stacked_params)
        save_params(p, os.path.join(rd, f"dlhf_{i}.pkl"))

    # Save curves
    save_curves(all_losses, val_l1s, eval_steps,
                os.path.join(rd, "curves.png"),
                extra_metrics=extra)

    # Save rollout visualisation
    print("Generating rollout visualisation...")
    p0 = jax.tree.map(lambda x: x[0], stacked_params)
    save_rollout(network, p0, meta, args)

    print("Done!")


if __name__ == "__main__":
    main(tyro.cli(DLHFArgs))