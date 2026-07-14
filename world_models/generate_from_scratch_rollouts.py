"""
generate_from_scratch_rollouts.py
=================================
Train + rollout world models for the Section 4.1 figure.

For each (env, budget, rollout_seed):
  1. Train a DLHF world model at that budget (or load checkpoint).
  2. Roll out the real env (GT) with a heuristic policy.
  3. Run autoregressive model predictions.
  4. Save raw flat obs arrays to .npz.
  5. Save a GT preview GIF per (env, rollout_seed) for visual selection.

Workflow:
  1. python generate_from_scratch_rollouts.py --rollout-seeds 0 1 2
  2. Browse out/from_scratch_rollouts/previews/ to pick good seeds
  3. python plot_from_scratch_rollouts.py --rollout-seeds 2 0 1

.npz keys:
    envs, budgets, seed, rollout_seeds, horizon
    actions_{env}_rs{rseed}                    (H,) int
    gt_obs_{env}_rs{rseed}                     (H+1, obs_dim) int
    model_obs_{env}_{budget}_rs{rseed}         (H+1, obs_dim) int

Preview GIFs:
    {preview_dir}/{env}_rs{rseed}.gif

Usage:
    python generate_from_scratch_rollouts.py --rollout-seeds 0 1 2 3 4
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import EnvMeta, collect_val_set, make_preferences_batch_fn
from network import (
    LatentCNNNetwork, build_network, init_params,
    save_params, load_params,
)
from dlhf import make_meta, train_dlhf, DLHFArgs
from heuristic_solvers import heuristic_step

ACTION_GLYPHS = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}


# =====================================================================
# Config
# =====================================================================

@dataclass
class Args:
    envs: List[str] = field(default_factory=lambda: [
        "sokoban", "maze10", "sliding5",
    ])
    budgets: List[int] = field(default_factory=lambda: [
        1_000, 5_000, 1_000_000,
    ])
    seed: int = 0
    """Training seed (determines checkpoint path)."""
    rollout_seeds: List[int] = field(default_factory=lambda: [18, 5, 10])
    """Rollout seeds to generate for ALL envs. Browse GIFs to pick."""
    horizon: int = 4

    ckpt_dir:    str = "out/from_scratch_rollouts"
    output:      str = "out/from_scratch_rollouts/rollout_data.npz"
    preview_dir: str = "out/from_scratch_rollouts/previews"

    # Architecture
    embed_dim:       int = 128
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 4
    dec_layers:      int = 3

    # Training
    lr:             float = 3e-4
    batch_size:     int   = 64
    bt_horizon:     int   = 1
    beta_bt:        float = 1.0
    scramble_steps: int   = 0
    context_len:    int   = 10
    ensemble_size:  int   = 1

    val_size:     int = 500
    val_scramble: int = 30
    eval_every:   int = 100

    rollout_scramble: int = 30

    override_actions: Optional[List[str]] = None
    """Per-env manual action overrides (positional, matching --envs order).
    Each entry is a comma-separated list of action indices, or 'auto' for heuristic.
    E.g. --override-actions '0,1,0' auto auto
    means sokoban gets [up,right,up], maze/sliding use heuristic.
    Actions: 0=up, 1=right, 2=down, 3=left."""

    force: bool = False
    """Re-train all models even if checkpoints exist."""


# =====================================================================
# Training
# =====================================================================

def _make_dlhf_args(args: Args, env: str, seed: int,
                    steps: int, results_dir: str) -> DLHFArgs:
    return DLHFArgs(
        seed=seed, env=env,
        embed_dim=args.embed_dim,
        latent_channels=args.latent_channels,
        enc_layers=args.enc_layers,
        dyn_layers=args.dyn_layers,
        dec_layers=args.dec_layers,
        lr=args.lr, steps=steps, batch_size=args.batch_size,
        horizon=args.bt_horizon, beta_bt=args.beta_bt,
        scramble_steps=args.scramble_steps,
        context_len=args.context_len, ensemble_size=args.ensemble_size,
        renew=False, val_size=args.val_size,
        val_scramble=args.val_scramble, eval_every=args.eval_every,
        results_dir=results_dir, rollout_steps=args.horizon,
    )


def train_model(args: Args, env: str, budget: int, seed: int) -> str:
    ckpt_dir = os.path.join(
        args.ckpt_dir, env, f"budget_{budget}", f"seed_{seed}")
    ckpt_path = os.path.join(ckpt_dir, "dlhf_0.pkl")

    if os.path.exists(ckpt_path) and not args.force:
        print(f"    EXISTS  {ckpt_path}")
        return ckpt_path

    os.makedirs(ckpt_dir, exist_ok=True)
    B = args.batch_size
    steps = budget // B
    M = args.ensemble_size

    dlhf_args = _make_dlhf_args(args, env, seed, steps, ckpt_dir)
    meta    = make_meta(dlhf_args)
    network = build_network(meta, dlhf_args)

    opt = optax.adam(args.lr)
    all_params, all_opts = [], []
    for m in range(M):
        rng = jax.random.PRNGKey(seed + m * 1000)
        p = init_params(network, meta, B, args.context_len, rng)
        all_params.append(p)
        all_opts.append(opt.init(p))

    stacked_params    = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    stacked_opt_state = jax.tree.map(lambda *os_: jnp.stack(os_), *all_opts)

    val_set   = collect_val_set(meta, args.val_size, args.val_scramble, seed)
    gen_batch = make_preferences_batch_fn(
        meta, B, args.scramble_steps, args.bt_horizon)
    train_rng = jax.random.PRNGKey(seed + 123)

    print(f"\n    TRAIN  {env}  budget={budget:,}  ({steps} steps)")
    t0 = time.time()
    stacked_params, _, losses, val_l1s, eval_steps = train_dlhf(
        network, stacked_params, stacked_opt_state, gen_batch,
        val_set, dlhf_args, train_rng, ensemble_size=M,
    )
    elapsed = time.time() - t0

    p0 = jax.tree.map(lambda x: x[0], stacked_params)
    save_params(p0, ckpt_path)
    print(f"    DONE   val_l1={val_l1s[-1]:.4f}  ({elapsed:.1f}s)")
    return ckpt_path


# =====================================================================
# Helpers
# =====================================================================

def scramble_single(env, meta, state, n_steps: int, key):
    for i in range(n_steps):
        k = jax.random.fold_in(key, i)
        a = jax.random.randint(k, (), 0, meta.num_actions)
        state, _ = env.step(state, a)
    return state


def extract_single(meta, state):
    batched = jax.tree.map(lambda x: x[None], state)
    return meta.extract_obs(batched)[0]


def infer_env_name(meta: EnvMeta) -> str:
    R, C = meta.grid_shape
    if meta.num_tiles == 4:
        return f"maze{R}"
    elif meta.num_tiles == 7:
        return "sokoban"
    else:
        return f"sliding{R}"


# =====================================================================
# Rollouts
# =====================================================================

def generate_gt_rollout(
    env, meta: EnvMeta, start_state, horizon: int,
    manual_actions: Optional[List[int]] = None,
) -> Tuple[List[int], List[np.ndarray], list]:
    """GT rollout with reactive heuristic or manual actions.

    Returns (actions, obs_list, state_list).
    """
    env_name = infer_env_name(meta)
    state    = start_state
    obs_flat = np.asarray(extract_single(meta, state))

    obs_list   = [obs_flat]
    state_list = [start_state]
    actions    = []

    for t in range(horizon):
        if manual_actions is not None:
            a = manual_actions[t]
        else:
            a = heuristic_step(env_name, obs_flat, meta.grid_shape)
        actions.append(a)
        state, _ = env.step(state, jnp.int32(a))
        obs_flat = np.asarray(extract_single(meta, state))
        obs_list.append(obs_flat)
        state_list.append(state)

    return actions, obs_list, state_list


def generate_model_rollout(
    network, params, start_obs: np.ndarray, actions: List[int],
) -> List[np.ndarray]:
    cur = jnp.array(start_obs)
    obs_list = [np.asarray(cur)]

    for a in actions:
        pred = network.apply(
            params, cur[None], jnp.array([a]),
            method=LatentCNNNetwork.predict,
        )
        cur = pred[0]
        obs_list.append(np.asarray(cur))

    return obs_list


# =====================================================================
# GT preview GIF
# =====================================================================

def save_gt_preview_gif(env, state_list, actions, path: str,
                        duration_ms: int = 600):
    """Save a GIF of the GT rollout using Jumanji's renderer."""
    from PIL import Image as PILImage
    from io import BytesIO

    pil_frames = []
    for state in state_list:
        env.render(state)
        fig = plt.gcf()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", edgecolor="none")
        plt.close("all")
        buf.seek(0)
        pil_frames.append(PILImage.open(buf).convert("RGB").copy())

    if pil_frames:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0)


# =====================================================================
# Main
# =====================================================================

def main(args: Args) -> None:
    data: Dict[str, np.ndarray] = {
        "envs":          np.array(args.envs),
        "budgets":       np.array(args.budgets),
        "seed":          np.array(args.seed),
        "rollout_seeds": np.array(args.rollout_seeds),
        "horizon":       np.array(args.horizon),
    }

    for env_str in args.envs:
        print(f"\n{'='*60}")
        print(f"  {env_str}")
        print(f"{'='*60}")

        dummy_args = _make_dlhf_args(args, env_str, args.seed, 100, "/tmp")
        meta    = make_meta(dummy_args)
        env     = meta.make_env()
        network = build_network(meta, dummy_args)

        data[f"grid_shape_{env_str}"]  = np.array(meta.grid_shape)
        data[f"num_tiles_{env_str}"]   = np.array(meta.num_tiles)
        data[f"num_actions_{env_str}"] = np.array(meta.num_actions)

        # Parse manual action override for this env
        manual_actions = None
        if args.override_actions is not None:
            env_idx = args.envs.index(env_str)
            if env_idx < len(args.override_actions):
                spec = args.override_actions[env_idx]
                if spec.lower() != "auto":
                    manual_actions = [int(a) for a in spec.split(",")]
                    assert len(manual_actions) == args.horizon, \
                        (f"--override-actions for {env_str}: got "
                         f"{len(manual_actions)} actions, need {args.horizon}")

        # Train models (once per budget, shared across rollout seeds)
        ckpt_paths = {}
        for budget in args.budgets:
            ckpt_paths[budget] = train_model(
                args, env_str, budget, args.seed)

        # Rollouts at each seed
        for rseed in args.rollout_seeds:
            rng = jax.random.PRNGKey(rseed)
            rng, reset_key, scramble_key = jax.random.split(rng, 3)
            start_state, _ = env.reset(reset_key)
            start_state = scramble_single(
                env, meta, start_state,
                args.rollout_scramble, scramble_key)

            # GT rollout
            actions, gt_obs_list, gt_state_list = generate_gt_rollout(
                env, meta, start_state, args.horizon, manual_actions)

            rs_tag = f"rs{rseed}"
            data[f"actions_{env_str}_{rs_tag}"] = np.array(actions)
            data[f"gt_obs_{env_str}_{rs_tag}"]  = np.stack(gt_obs_list)

            act_str = [ACTION_GLYPHS.get(a, str(a)) for a in actions]
            print(f"  rseed={rseed}  actions={act_str}")

            # Save GT preview GIF
            gif_path = os.path.join(
                args.preview_dir, f"{env_str}_rs{rseed}.gif")
            if not os.path.exists(gif_path):
                save_gt_preview_gif(
                    env, gt_state_list, actions, gif_path)
                print(f"    GIF -> {gif_path}")
            else:
                print(f"    GIF exists: {gif_path}")

            # Model rollouts at each budget
            for budget in args.budgets:
                params = load_params(ckpt_paths[budget])
                model_obs = generate_model_rollout(
                    network, params, gt_obs_list[0], actions)

                key = f"model_obs_{env_str}_{budget}_{rs_tag}"
                data[key] = np.stack(model_obs)

            print(f"    all budgets done")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"\n  Saved -> {args.output}  "
          f"({os.path.getsize(args.output) / 1024:.0f} KB)")
    print(f"  Preview GIFs in: {args.preview_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))