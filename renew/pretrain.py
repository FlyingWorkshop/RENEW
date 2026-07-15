"""
renew/pretrain.py
=================
Pretrain a LatentCNN world model on Maze with offline (reconstruction) data.
Also exports all shared infrastructure (network, env meta, eval, heatmaps)
for use by finetune.py.

Usage:
  python renew/pretrain.py
  python renew/pretrain.py --pretrain-steps 5000 --maze-size 10
  python renew/pretrain.py --results-dir out/maze10/seed_0
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import tyro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_ACTIONS = 4   # maze: up, right, down, left
NUM_TILES   = 4   # maze: wall=0, floor=1, agent=2, target=3
MOVES_DR    = jnp.array([-1, 0, 1, 0])
MOVES_DC    = jnp.array([0, 1, 0, -1])


# =============================================================================
# Args
# =============================================================================

@dataclass
class PretrainArgs:
    seed: int = 0

    # --- Environment ---
    maze_size: int = 10

    # --- Pretraining ---
    pretrain_lr:         float = 3e-4
    pretrain_steps:      int   = 500
    pretrain_batch:      int   = 32
    offline_dataset_size: int  = 50
    scramble_steps:      int   = 0
    context_len:         int   = 10

    # --- Architecture ---
    embed_dim:       int = 32
    channels:        int = 64
    num_layers:      int = 4
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 3
    dec_layers:      int = 2

    # --- Ensemble ---
    ensemble_size: int = 3
    """Number of ensemble members. 1 = single model (entropy uncertainty)."""

    # --- Eval during training ---
    eval_every: int = 50
    """Evaluate grid accuracy every N steps during pretraining."""

    # --- Results output ---
    results_dir: Optional[str] = None
    """If set, save metrics/curves data here for aggregation."""


# =============================================================================
# Env meta (maze)
# =============================================================================

class EnvMeta(NamedTuple):
    obs_dim:      int
    num_tiles:    int
    num_actions:  int
    grid_shape:   Tuple[int, int]
    make_env:     Callable
    extract_obs:  Callable
    oracle_step:  Callable
    obs_to_state: Callable


def make_maze_meta(maze_size: int = 10) -> EnvMeta:
    from jumanji.environments.routing.maze.env import Maze
    from jumanji.environments.routing.maze.generator import RandomGenerator
    from jumanji.environments.routing.maze.types import Position as MazePos

    ROWS, COLS = maze_size, maze_size
    OBS_DIM    = ROWS * COLS

    def extract(states):
        B    = states.walls.shape[0]
        grid = jnp.where(states.walls.reshape(B, OBS_DIM).astype(jnp.int32), 0, 1)
        ar, ac = states.agent_position.row, states.agent_position.col
        grid = grid.at[jnp.arange(B), ar * COLS + ac].set(2)
        tr, tc = states.target_position.row, states.target_position.col
        grid = grid.at[jnp.arange(B), tr * COLS + tc].set(3)
        return grid

    def oracle(key, obs, action):
        agent_idx = jnp.argmax(obs == 2)
        tgt_idx   = jnp.argmax(obs == 3)
        row, col  = jnp.divmod(agent_idx, COLS)
        dr = jnp.array([-1, 0, 1, 0])[action]
        dc = jnp.array([0, 1, 0, -1])[action]
        new_idx   = jnp.clip(row+dr, 0, ROWS-1)*COLS + jnp.clip(col+dc, 0, COLS-1)
        final_idx = jnp.where(obs[new_idx] == 0, agent_idx, new_idx)
        new_obs   = obs.at[agent_idx].set(jnp.where(agent_idx == tgt_idx, 3, 1))
        new_obs   = new_obs.at[final_idx].set(jnp.where(final_idx == tgt_idx, 3, 2))
        return new_obs

    def obs_to_state(obs_flat, real_state):
        agent_idx = jnp.argmax(obs_flat == 2)
        tgt_idx   = jnp.argmax(obs_flat == 3)
        ar, ac    = jnp.divmod(agent_idx, COLS)
        tr, tc    = jnp.divmod(tgt_idx,   COLS)
        return real_state.replace(
            agent_position  = MazePos(row=ar, col=ac),
            target_position = MazePos(row=tr, col=tc))

    return EnvMeta(obs_dim=OBS_DIM, num_tiles=4, num_actions=4,
                   grid_shape=(ROWS, COLS),
                   make_env=lambda: Maze(generator=RandomGenerator(
                       num_rows=ROWS, num_cols=COLS)),
                   extract_obs=extract, oracle_step=oracle,
                   obs_to_state=obs_to_state)


# =============================================================================
# Network building blocks
# =============================================================================

class ConvNextBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        r = x
        x = nn.Conv(self.channels, (3, 3), padding="SAME",
                     feature_group_count=self.channels)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.channels * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.channels)(x)
        return x + r


class _LatentEncoder(nn.Module):
    embed_dim: int; latent_channels: int; enc_layers: int
    num_tiles: int; grid_shape: Tuple[int, int]

    @nn.compact
    def __call__(self, obs):
        rows, cols = self.grid_shape
        B = obs.shape[0]
        emb = nn.Embed(self.num_tiles, self.embed_dim)(obs)
        x = emb.reshape(B, rows, cols, self.embed_dim)
        x = nn.Dense(self.latent_channels)(x)
        for _ in range(self.enc_layers):
            x = ConvNextBlock(self.latent_channels)(x)
        return x


class _LatentDynamics(nn.Module):
    latent_channels: int; dyn_layers: int; num_actions: int

    @nn.compact
    def __call__(self, z, action):
        act_emb = nn.Dense(self.latent_channels)(jax.nn.one_hot(action, self.num_actions))
        act_map = jnp.broadcast_to(act_emb[:, None, None, :], z.shape)
        x = nn.Dense(self.latent_channels)(jnp.concatenate([z, act_map], axis=-1))
        for _ in range(self.dyn_layers):
            x = ConvNextBlock(self.latent_channels)(x)
        return x


class _LatentDecoder(nn.Module):
    latent_channels: int; dec_layers: int; num_tiles: int; num_actions: int

    @nn.compact
    def __call__(self, z, action):
        act_emb = nn.Dense(self.latent_channels)(jax.nn.one_hot(action, self.num_actions))
        act_map = jnp.broadcast_to(act_emb[:, None, None, :], z.shape)
        x = nn.Dense(self.latent_channels)(jnp.concatenate([z, act_map], axis=-1))
        for _ in range(self.dec_layers):
            x = ConvNextBlock(self.latent_channels)(x)
        return nn.Dense(self.num_tiles)(x)


class LatentCNNNetwork(nn.Module):
    embed_dim: int; latent_channels: int; enc_layers: int; dyn_layers: int
    dec_layers: int; num_tiles: int; num_actions: int
    grid_shape: Tuple[int, int]
    # These are stored but only used by preferences_loss
    beta_bt: float = 1.0
    finetune_horizon: int = 1

    def setup(self):
        self.encoder = _LatentEncoder(
            self.embed_dim, self.latent_channels, self.enc_layers,
            self.num_tiles, self.grid_shape)
        self.dynamics = _LatentDynamics(
            self.latent_channels, self.dyn_layers, self.num_actions)
        self.decoder = _LatentDecoder(
            self.latent_channels, self.dec_layers, self.num_tiles, self.num_actions)

    def __call__(self, obs, action):
        z = self.encoder(obs)
        z_next = self.dynamics(z, action)
        return self.decoder(z_next, action)

    def predict(self, obs, action):
        return jnp.argmax(self(obs, action), axis=-1).reshape(obs.shape[0], -1)

    def decode(self, obs, action):
        rows, cols = self.grid_shape
        return self(obs, action).reshape(obs.shape[0], -1, self.num_tiles)

    def offline_loss(self, context_boards, context_actions, key):
        obs     = context_boards[:, :-1, :]
        actions = context_actions[:, :-1]
        targets = context_boards[:, 1:, :]
        B, T = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(B * T, -1)
        act_flat = actions.reshape(B * T)
        tgt_flat = targets.reshape(B * T, -1)
        logits = self.decode(obs_flat, act_flat)
        lp = jax.nn.log_softmax(logits, axis=-1)
        oh = jax.nn.one_hot(tgt_flat, self.num_tiles)
        l_recon = -(lp * oh).sum(-1).mean()
        return l_recon, dict(l_recon=l_recon)

    def preferences_loss(self, start_obs, acts1, acts2, gt1, gt2, key):
        H = self.finetune_horizon
        rng1, rng2 = jax.random.split(key)

        def _rollout(obs, actions, rng):
            all_samp, all_lp = [], []
            for t in range(H):
                sk = jax.random.fold_in(rng, t)
                logits = self.decode(obs, actions[:, t])
                samp = jax.random.categorical(sk, logits)
                lp = (jax.nn.log_softmax(logits, -1)
                      * jax.nn.one_hot(samp, self.num_tiles)).sum((-1, -2))
                all_samp.append(samp)
                all_lp.append(lp)
                obs = samp
            return jnp.stack(all_samp, 1), jnp.stack(all_lp, 1)

        samp1, lp1 = _rollout(start_obs, acts1, rng1)
        samp2, lp2 = _rollout(start_obs, acts2, rng2)
        err1 = jnp.abs(samp1 - gt1).sum((-1, -2)).astype(jnp.float32)
        err2 = jnp.abs(samp2 - gt2).sum((-1, -2)).astype(jnp.float32)
        total_lp1, total_lp2 = lp1.sum(-1), lp2.sum(-1)
        preferences = (err2 < err1).astype(jnp.float32)
        ties = (err1 == err2).astype(jnp.float32)
        pred_logits = self.beta_bt * (total_lp2 - total_lp1)
        bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
        l_bt = jnp.mean(bce * (1.0 - ties))
        total = self.beta_bt * l_bt
        no_tie = 1.0 - ties
        pref_acc = jnp.where(
            no_tie.mean() > 0,
            jnp.sum(((pred_logits > 0) == (preferences > 0.5)).astype(jnp.float32) * no_tie)
            / (jnp.sum(no_tie) + 1e-8), 0.5)
        return total, dict(bt_loss=l_bt, tie_rate=ties.mean(), pref_acc=pref_acc)


def build_network(maze_size: int, args) -> Tuple[LatentCNNNetwork, EnvMeta]:
    """Build network + meta from any args object with the right fields."""
    meta = make_maze_meta(maze_size)
    beta_bt = getattr(args, "beta_bt", 1.0)
    finetune_horizon = getattr(args, "finetune_horizon", 1)
    network = LatentCNNNetwork(
        embed_dim=args.embed_dim, latent_channels=args.latent_channels,
        enc_layers=args.enc_layers, dyn_layers=args.dyn_layers,
        dec_layers=args.dec_layers, num_tiles=meta.num_tiles,
        num_actions=meta.num_actions, grid_shape=meta.grid_shape,
        beta_bt=beta_bt, finetune_horizon=finetune_horizon,
    )
    return network, meta


# =============================================================================
# Grid-level evaluation
# =============================================================================

def _oracle_step_eval(key, obs, action, maze_size):
    agent_idx = jnp.argmax(obs == 2)
    tgt_idx   = jnp.argmax(obs == 3)
    row, col  = jnp.divmod(agent_idx, maze_size)
    dr, dc    = MOVES_DR[action], MOVES_DC[action]
    new_idx   = (jnp.clip(row + dr, 0, maze_size - 1) * maze_size +
                 jnp.clip(col + dc, 0, maze_size - 1))
    final_idx = jnp.where(obs[new_idx] == 0, agent_idx, new_idx)
    new_obs   = obs.at[agent_idx].set(jnp.where(agent_idx == tgt_idx, 3, 1))
    new_obs   = new_obs.at[final_idx].set(jnp.where(final_idx == tgt_idx, 3, 2))
    return new_obs


def evaluate_grid(network, params, walls: np.ndarray, target_cell: int,
                  maze_size: int, ensemble_params: list = None) -> Dict[str, Any]:
    """
    Evaluate grid-level transition accuracy.
    If ensemble_params is provided (list of param dicts), use ensemble
    disagreement as uncertainty. Otherwise fall back to single-model entropy.
    """
    obs_dim    = maze_size * maze_size
    walls_flat = walls.ravel().astype(bool)
    floor_cells = np.where(~walls_flat)[0]
    n_floor     = len(floor_cells)

    base_grid = np.where(walls_flat, 0, 1).astype(np.int32)
    base_grid[target_cell] = 3

    all_obs = []
    for cell in floor_cells:
        obs = base_grid.copy()
        obs[cell] = 2
        all_obs.append(obs)
    all_obs = np.array(all_obs)

    obs_expanded  = np.repeat(all_obs, NUM_ACTIONS, axis=0)
    acts_expanded = np.tile(np.arange(NUM_ACTIONS), n_floor)

    obs_j  = jnp.array(obs_expanded)
    acts_j = jnp.array(acts_expanded)
    keys   = jax.random.split(jax.random.PRNGKey(0), obs_expanded.shape[0])
    oracle_fn = jax.vmap(lambda k, o, a: _oracle_step_eval(k, o, a, maze_size))
    true_next = np.array(oracle_fn(keys, obs_j, acts_j))

    @jax.jit
    def _predict(params, obs, actions):
        return network.apply(params, obs, actions, method=LatentCNNNetwork.predict)

    @jax.jit
    def _get_probs(params, obs, actions):
        logits = network.apply(params, obs, actions, method=LatentCNNNetwork.decode)
        return jax.nn.softmax(logits, axis=-1)

    # Use first member (or single model) for accuracy
    eval_params = params

    CHUNK = 2048
    wm_preds = []
    for i in range(0, obs_expanded.shape[0], CHUNK):
        o = jnp.array(obs_expanded[i:i+CHUNK])
        a = jnp.array(acts_expanded[i:i+CHUNK])
        wm_preds.append(np.array(_predict(eval_params, o, a)))
    wm_next = np.concatenate(wm_preds, axis=0)

    # Compute uncertainty
    if ensemble_params is not None and len(ensemble_params) > 1:
        # Ensemble disagreement: entropy of mean probs - mean of entropies
        all_member_probs = []
        for mp in ensemble_params:
            member_probs = []
            for i in range(0, obs_expanded.shape[0], CHUNK):
                o = jnp.array(obs_expanded[i:i+CHUNK])
                a = jnp.array(acts_expanded[i:i+CHUNK])
                member_probs.append(np.array(_get_probs(mp, o, a)))
            all_member_probs.append(np.concatenate(member_probs, axis=0))
        # all_member_probs: list of (N, obs_dim, num_tiles) arrays
        stacked = np.stack(all_member_probs, axis=0)  # (M, N, obs_dim, num_tiles)
        mean_probs = stacked.mean(axis=0)  # (N, obs_dim, num_tiles)
        # Entropy of mean
        ent_mean = -(mean_probs * np.log(mean_probs + 1e-10)).sum(-1)  # (N, obs_dim)
        # Mean of entropies
        ent_each = -(stacked * np.log(stacked + 1e-10)).sum(-1)  # (M, N, obs_dim)
        mean_ent = ent_each.mean(axis=0)  # (N, obs_dim)
        # Mutual information = epistemic uncertainty
        epistemic = (ent_mean - mean_ent).mean(-1)  # (N,) averaged over grid positions
        entropies = epistemic
    else:
        # Single model fallback: softmax entropy
        all_entropies = []
        for i in range(0, obs_expanded.shape[0], CHUNK):
            o = jnp.array(obs_expanded[i:i+CHUNK])
            a = jnp.array(acts_expanded[i:i+CHUNK])
            p = np.array(_get_probs(eval_params, o, a))
            cell_ent = -(p * np.log(p + 1e-10)).sum(-1)
            all_entropies.append(cell_ent.mean(-1))
        entropies = np.concatenate(all_entropies, axis=0)

    correct = (wm_next == true_next).all(axis=-1)
    wrong   = ~correct

    agent_cells     = np.repeat(floor_cells, NUM_ACTIONS)
    true_agent_next = np.array([np.argmax(true_next[i] == 2) for i in range(len(true_next))])
    wm_agent_next   = np.array([np.argmax(wm_next[i] == 2) for i in range(len(wm_next))])
    wall_clips      = (true_agent_next == agent_cells) & (wm_agent_next != agent_cells)

    error_grid       = np.zeros((maze_size, maze_size, NUM_ACTIONS), dtype=int)
    uncertainty_grid = np.zeros((maze_size, maze_size, NUM_ACTIONS), dtype=float)
    for i, (cell, act) in enumerate(zip(np.repeat(floor_cells, NUM_ACTIONS),
                                         np.tile(np.arange(NUM_ACTIONS), n_floor))):
        r, c = divmod(cell, maze_size)
        if wrong[i]:
            error_grid[r, c, act] = 1
        uncertainty_grid[r, c, act] = entropies[i]

    n_valid = n_floor * NUM_ACTIONS
    target_r, target_c = divmod(target_cell, maze_size)
    return {
        "transition_acc":   float(correct.sum()) / max(n_valid, 1),
        "n_wrong":          int(wrong.sum()),
        "n_wall_clips":     int(wall_clips.sum()),
        "n_valid":          n_valid,
        "error_grid":       error_grid,
        "uncertainty_grid": uncertainty_grid,
        "walls":            walls,
        "target_rc":        (target_r, target_c),
    }


def get_maze_layout(maze_size: int, seed: int):
    """Get walls + target cell for a specific maze seed."""
    from jumanji.environments.routing.maze.env import Maze
    from jumanji.environments.routing.maze.generator import RandomGenerator
    env = Maze(generator=RandomGenerator(num_rows=maze_size, num_cols=maze_size))
    init_state, _ = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    tr = int(init_state.target_position.row)
    tc = int(init_state.target_position.col)
    return np.array(init_state.walls), tr * maze_size + tc, (tr, tc)


def is_maze_env(env_name: str) -> bool:
    return env_name.startswith("maze")


def evaluate_transitions(network, params, meta, n_samples=2000, seed=0):
    """
    Sample-based transition accuracy. Works for any environment.
    Rolls out random policies, compares WM predictions to oracle.
    """
    env = meta.make_env()
    rng = jax.random.PRNGKey(seed + 777)

    B = min(n_samples, 200)
    rollout_len = (n_samples // B) + 1

    rng, rk = jax.random.split(rng)
    states, _ = jax.vmap(env.reset)(jax.random.split(rk, B))

    step_jit = jax.jit(jax.vmap(env.step))
    all_obs, all_acts = [], []

    for t in range(rollout_len):
        obs = meta.extract_obs(states)
        rng, ak = jax.random.split(rng)
        actions = jax.random.randint(ak, (B,), 0, meta.num_actions)
        all_obs.append(np.array(obs))
        all_acts.append(np.array(actions))
        states, _ = step_jit(states, actions)

    all_obs = np.concatenate(all_obs, axis=0)[:n_samples]
    all_acts = np.concatenate(all_acts, axis=0)[:n_samples]
    N = len(all_obs)

    oracle_fn = jax.vmap(meta.oracle_step)
    keys = jax.random.split(jax.random.PRNGKey(seed + 778), N)

    @jax.jit
    def _predict(params, obs, actions):
        return network.apply(params, obs, actions, method=LatentCNNNetwork.predict)

    CHUNK = 2048
    all_true, all_pred = [], []
    for i in range(0, N, CHUNK):
        o = jnp.array(all_obs[i:i+CHUNK])
        a = jnp.array(all_acts[i:i+CHUNK])
        k = keys[i:i+CHUNK]
        all_true.append(np.array(oracle_fn(k, o, a)))
        all_pred.append(np.array(_predict(params, o, a)))

    true_next = np.concatenate(all_true, axis=0)
    pred_next = np.concatenate(all_pred, axis=0)

    correct = (pred_next == true_next).all(axis=-1)
    return {
        "transition_acc": float(correct.mean()),
        "n_wrong": int((~correct).sum()),
        "n_valid": N,
    }


# =============================================================================
# Heatmap helpers
# =============================================================================

def _add_walls_and_star(ax, walls, target_rc, maze_size):
    for r in range(maze_size):
        for c in range(maze_size):
            if walls[r, c]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            fc="#333333", ec="none", zorder=2))
    tr, tc = target_rc
    ax.plot(tc, tr, marker="*", markersize=18, color="gold",
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax.set_xticks(range(maze_size))
    ax.set_yticks(range(maze_size))


def make_error_heatmap(ax, metrics, maze_size):
    error_grid = metrics["error_grid"]
    walls = metrics["walls"]
    n_wrong = error_grid.sum(axis=-1).astype(float)
    n_wrong[walls] = np.nan
    im = ax.imshow(n_wrong, cmap="RdYlGn_r", vmin=0, vmax=4,
                   interpolation="nearest", origin="upper")
    for r in range(maze_size):
        for c in range(maze_size):
            if not walls[r, c] and error_grid[r, c].any():
                ax.text(c, r, str(int(n_wrong[r, c])), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold", zorder=3)
    acc = metrics["transition_acc"]
    ax.set_title(f"Errors — acc={acc*100:.1f}%", fontsize=10)
    _add_walls_and_star(ax, walls, metrics["target_rc"], maze_size)
    return im


def make_unc_heatmap(ax, metrics, maze_size):
    unc = metrics["uncertainty_grid"].mean(axis=-1).copy()
    walls = metrics["walls"]
    unc[walls] = np.nan
    im = ax.imshow(unc, cmap="inferno", interpolation="nearest", origin="upper")
    for r in range(maze_size):
        for c in range(maze_size):
            if not walls[r, c] and not np.isnan(unc[r, c]) and unc[r, c] > 0.01:
                ax.text(c, r, f"{unc[r,c]:.2f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=3)
    ax.set_title(f"Uncertainty — mean={np.nanmean(unc):.4f}", fontsize=10)
    _add_walls_and_star(ax, walls, metrics["target_rc"], maze_size)
    return im


def save_params(params, path):
    with open(path, "wb") as f:
        pickle.dump(jax.tree.map(np.array, params), f)
    print(f"Saved → {path}")


def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_ensemble(params_list, dir_path, prefix="pretrained"):
    """Save ensemble members as separate files."""
    os.makedirs(dir_path, exist_ok=True)
    for i, p in enumerate(params_list):
        save_params(p, os.path.join(dir_path, f"{prefix}_{i}.pkl"))
    print(f"  Saved {len(params_list)} ensemble members → {dir_path}")


def load_ensemble(dir_path, prefix="pretrained"):
    """Load ensemble members from separate files. Falls back to single model."""
    # Try loading ensemble files
    members = []
    i = 0
    while True:
        path = os.path.join(dir_path, f"{prefix}_{i}.pkl")
        if os.path.exists(path):
            members.append(load_params(path))
            i += 1
        else:
            break
    if members:
        print(f"  Loaded {len(members)} ensemble members from {dir_path}")
        return members
    # Fallback: single pretrained.pkl
    single_path = os.path.join(dir_path, f"{prefix}.pkl")
    if os.path.exists(single_path):
        print(f"  Loaded single model from {single_path}")
        return [load_params(single_path)]
    raise FileNotFoundError(f"No ensemble or single checkpoint found in {dir_path}")


# =============================================================================
# Training curve helpers (shared with finetune.py)
# =============================================================================

def save_loss_accuracy_curves(losses, accuracies, acc_steps, title, save_path):
    """Save a 2-panel figure: loss curve (left) + accuracy curve (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Loss
    ax1.plot(np.array(losses), linewidth=0.8, color="tab:blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — final: {float(losses[-1]):.4f}")
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(acc_steps, accuracies, "o-", linewidth=1.2, markersize=4, color="tab:green")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Transition Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax2.set_title(f"Accuracy — final: {accuracies[-1]*100:.1f}%")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves → {save_path}")


# =============================================================================
# Save metrics/curves to results dir (for multi-seed aggregation)
# =============================================================================

def save_results(results_dir: str, tag: str, metrics: Dict[str, Any],
                 losses: list = None, accuracies: list = None,
                 acc_steps: list = None):
    """Save evaluation metrics and optional training curves under results_dir/tag/."""
    d = os.path.join(results_dir, tag)
    os.makedirs(d, exist_ok=True)

    # Scalars → JSON
    scalars = {k: v for k, v in metrics.items()
               if not isinstance(v, np.ndarray)}
    if "target_rc" in scalars:
        scalars["target_rc"] = list(scalars["target_rc"])
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(scalars, f, indent=2)

    # Arrays → .npy
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(d, f"{k}.npy"), v)

    # Training curves → CSV
    if losses is not None:
        with open(os.path.join(d, "losses.csv"), "w") as f:
            f.write("step,loss\n")
            for i, loss in enumerate(losses):
                f.write(f"{i},{loss}\n")

    if accuracies is not None and acc_steps is not None:
        with open(os.path.join(d, "accuracies.csv"), "w") as f:
            f.write("step,accuracy\n")
            for step, acc in zip(acc_steps, accuracies):
                f.write(f"{step},{acc}\n")

    print(f"  Results → {d}")


# =============================================================================
# Data generation
# =============================================================================

def collect_offline_dataset(meta: EnvMeta, args) -> Dict[str, Any]:
    env  = meta.make_env()
    Tc   = args.context_len
    size = args.offline_dataset_size
    print(f"Collecting offline dataset ({size} sequences)...")

    @jax.jit
    def _collect(key):
        keys = jax.random.split(key, 3)
        states, _ = jax.vmap(env.reset)(jax.random.split(keys[0], size))
        def _scramble(i, s):
            k    = jax.random.fold_in(keys[1], i)
            acts = jax.random.randint(k, (size,), 0, meta.num_actions)
            s, _ = jax.vmap(env.step)(s, acts)
            return s
        states = jax.lax.fori_loop(0, args.scramble_steps, _scramble, states)
        ctx_boards, ctx_actions = [], []
        for t in range(Tc):
            acts = jax.random.randint(
                jax.random.fold_in(keys[2], t), (size,), 0, meta.num_actions)
            ctx_boards.append(meta.extract_obs(states))
            ctx_actions.append(acts)
            states, _ = jax.vmap(env.step)(states, acts)
        return dict(context_boards=jnp.stack(ctx_boards, 1),
                    context_actions=jnp.stack(ctx_actions, 1))

    dataset = jax.tree.map(
        lambda x: x.block_until_ready(),
        _collect(jax.random.PRNGKey(args.seed + 999_999)),
    )
    print(f"  {size} sequences × {Tc} steps")
    return dataset


# =============================================================================
# Pretraining (with periodic accuracy evaluation)
# =============================================================================

def pretrain_offline(network, stacked_params, stacked_opt_state, dataset,
                     meta, args, rng, ensemble_size=1):
    """
    Pretrain ensemble members in parallel using vmap.
    stacked_params/stacked_opt_state have leading dim = ensemble_size.
    dataset is collected once and shared across all members.
    Returns (stacked_params, rng, all_losses, acc_values, acc_steps).
    """
    size = dataset["context_boards"].shape[0]

    # Eval function (uses member 0 for accuracy)
    walls, target_cell, target_rc = get_maze_layout(args.maze_size, args.seed)
    def _eval(stacked_params):
        p0 = jax.tree.map(lambda x: x[0], stacked_params)
        return evaluate_grid(network, p0, walls, target_cell, args.maze_size)

    opt = optax.adam(args.pretrain_lr)

    @jax.jit
    def _sample(key):
        idxs = jax.random.randint(key, (args.pretrain_batch,), 0, size)
        return jax.tree.map(lambda x: x[idxs], dataset)

    def _single_member_step(params, opt_state, batch, sk):
        """Loss + grad + update for one member."""
        def loss_fn(p):
            return network.apply(
                p, batch["context_boards"], batch["context_actions"], sk,
                method=LatentCNNNetwork.offline_loss)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, info

    @jax.jit
    def _ensemble_step(stacked_params, stacked_opt_state, rng):
        rng, bk, sk = jax.random.split(rng, 3)
        batch = _sample(bk)
        # vmap over leading dim of params/opt_state, same batch and key for all
        new_params, new_opt_state, info = jax.vmap(
            lambda p, o: _single_member_step(p, o, batch, sk)
        )(stacked_params, stacked_opt_state)
        return new_params, new_opt_state, rng, info

    # lax.scan over chunks for speed, eval between chunks
    eval_every = args.eval_every

    @jax.jit
    def _scan_chunk(carry, _):
        stacked_params, stacked_opt_state, rng = carry
        stacked_params, stacked_opt_state, rng, info = _ensemble_step(
            stacked_params, stacked_opt_state, rng)
        loss = info["l_recon"][0]  # member 0's loss
        return (stacked_params, stacked_opt_state, rng), loss

    # Pre-compile scan for the standard chunk size
    @jax.jit
    def _run_chunk_standard(stacked_params, stacked_opt_state, rng):
        (stacked_params, stacked_opt_state, rng), losses = jax.lax.scan(
            _scan_chunk,
            (stacked_params, stacked_opt_state, rng),
            None, length=eval_every)
        return stacked_params, stacked_opt_state, rng, losses

    print(f"\n{'='*60}")
    label = f"Pretraining ensemble ({ensemble_size} members)" if ensemble_size > 1 else "Pretraining"
    print(f"{label} ({args.pretrain_steps} steps, eval every {eval_every})")
    print(f"{'='*60}")

    all_losses = []
    acc_steps = []
    acc_values = []

    t0 = time.time()
    steps_done = 0
    total = args.pretrain_steps

    # Run full chunks
    n_full_chunks = total // eval_every
    remainder = total % eval_every

    for chunk_idx in range(n_full_chunks):
        stacked_params, stacked_opt_state, rng, chunk_losses = _run_chunk_standard(
            stacked_params, stacked_opt_state, rng)
        jax.block_until_ready(stacked_params)

        all_losses.extend(np.array(chunk_losses).tolist())
        steps_done += eval_every

        metrics = _eval(stacked_params)
        acc = metrics["transition_acc"]
        acc_steps.append(steps_done - 1)
        acc_values.append(acc)
        elapsed = time.time() - t0
        print(f"  Step {steps_done:5d} | loss={float(chunk_losses[-1]):.4f} | "
              f"acc={acc*100:.1f}% | {elapsed:.1f}s")

    # Handle remainder with individual steps
    if remainder > 0:
        for _ in range(remainder):
            stacked_params, stacked_opt_state, rng, info = _ensemble_step(
                stacked_params, stacked_opt_state, rng)
            all_losses.append(float(info["l_recon"][0]))
        steps_done += remainder
        jax.block_until_ready(stacked_params)
        metrics = _eval(stacked_params)
        acc = metrics["transition_acc"]
        acc_steps.append(steps_done - 1)
        acc_values.append(acc)
        elapsed = time.time() - t0
        print(f"  Step {steps_done:5d} | loss={all_losses[-1]:.4f} | "
              f"acc={acc*100:.1f}% | {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Wall clock: {elapsed:.1f}s ({elapsed/total*1000:.1f} ms/step)")

    # Determine output dir
    results_dir = getattr(args, "results_dir", None)
    out_dir = results_dir if results_dir else OUT_DIR

    # Save combined loss + accuracy curves
    save_loss_accuracy_curves(
        all_losses, acc_values, acc_steps,
        "Pretraining",
        os.path.join(out_dir, "pretrain_curves.png"),
    )

    return stacked_params, stacked_opt_state, rng, all_losses, acc_values, acc_steps


# =============================================================================
# Main
# =============================================================================

def main(args: PretrainArgs) -> None:
    network, meta = build_network(args.maze_size, args)
    M = args.ensemble_size

    # Determine output dir
    out_dir = args.results_dir if args.results_dir else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Collect dataset ONCE
    dataset = collect_offline_dataset(meta, args)

    # Init all ensemble members with different seeds, stack into one pytree
    B, Tc, od = args.pretrain_batch, args.context_len, meta.obs_dim
    print(f"Initialising {M} LatentCNN member(s)...")
    opt = optax.adam(args.pretrain_lr)
    all_params = []
    all_opt_states = []
    for m in range(M):
        init_seed = args.seed + m * 1000
        init_rng = jax.random.PRNGKey(init_seed)
        params = network.init(
            init_rng,
            jnp.zeros((B, Tc, od), jnp.int32),
            jnp.zeros((B, Tc), jnp.int32),
            init_rng,
            method=LatentCNNNetwork.offline_loss,
        )
        all_params.append(params)
        all_opt_states.append(opt.init(params))

    # Stack into (M, ...) pytrees
    stacked_params = jax.tree.map(lambda *ps: jnp.stack(ps), *all_params)
    stacked_opt_state = jax.tree.map(lambda *os: jnp.stack(os), *all_opt_states)

    # Train
    train_rng = jax.random.PRNGKey(args.seed + 42)
    stacked_params, stacked_opt_state, _, pt_losses, pt_accs, pt_acc_steps = \
        pretrain_offline(network, stacked_params, stacked_opt_state, dataset,
                         meta, args, train_rng, ensemble_size=M)

    # Unstack for saving
    ensemble_params = [jax.tree.map(lambda x: x[i], stacked_params) for i in range(M)]

    # Save
    if M > 1:
        for i, p in enumerate(ensemble_params):
            save_params(p, os.path.join(out_dir, f"pretrained_{i}.pkl"))
    save_params(ensemble_params[0], os.path.join(out_dir, "pretrained.pkl"))

    # Evaluate using ensemble
    walls, target_cell, target_rc = get_maze_layout(args.maze_size, args.seed)
    metrics = evaluate_grid(network, ensemble_params[0], walls, target_cell,
                            args.maze_size,
                            ensemble_params=ensemble_params if M > 1 else None)
    print(f"\nGrid transition accuracy: {metrics['transition_acc']*100:.1f}%  "
          f"({metrics['n_wrong']} wrong / {metrics['n_valid']})")

    # Save results for aggregation
    save_results(out_dir, "pretrained", metrics,
                 losses=pt_losses, accuracies=pt_accs, acc_steps=pt_acc_steps)

    # Save heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    make_error_heatmap(axes[0], metrics, args.maze_size)
    make_unc_heatmap(axes[1], metrics, args.maze_size)
    unc_label = "Ensemble Disagreement" if M > 1 else "Entropy"
    fig.suptitle(f"After Pretraining ({unc_label})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pretrain_eval.png"), bbox_inches="tight")
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main(tyro.cli(PretrainArgs))