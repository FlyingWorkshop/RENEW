"""
world_models/from_scratch_gymnax.py
====================================
Section 4.1 (continuous): Learning world models from preferences (from scratch)
on Gymnax continuous-control environments.

Compares two training regimes:
    - Supervised:  N random (s,a,s') transitions (MSE loss)
    - DLHF:        M preference labels (Bradley-Terry loss)

Both train an ensemble of Gaussian MLP dynamics models (delta prediction).
The asymmetry is the point: demonstrations are expensive, preferences are
cheap. Can a large budget of binary comparisons match a small budget of
direct supervision?

Outputs:
    out/from_scratch_gymnax/{env}/seed_{s}/
        dlhf/curves.png
        supervised/curves.png
    out/from_scratch_gymnax/
        summary.png         -- multi-env MSE curves (mean ± 95% CI)
        summary.txt         -- results table

Usage:
    # Quick test
    python world_models/from_scratch_gymnax.py --seeds 0 \
        --supervised-transitions 50 --default-budget 5000

    # Full run (5 seeds)
    python world_models/from_scratch_gymnax.py

    # Override per-env budget
    python world_models/from_scratch_gymnax.py \
        --env-budgets.MountainCar-v0 200000
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import tyro
from flax import linen as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnax

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")


# =============================================================================
# Per-environment configuration
# =============================================================================

class EnvConfig(NamedTuple):
    gymnax_name: str
    obs_dim: int
    n_actions: int
    make_env_state: Callable
    default_start_obs: Callable


def _make_mountaincar_config() -> EnvConfig:
    from gymnax.environments.classic_control.mountain_car import EnvState
    def make_state(v):
        return EnvState(position=v[0], velocity=v[1], time=0)
    def start():
        return jnp.array([-0.5, 0.0])
    return EnvConfig("MountainCar-v0", 2, 3, make_state, start)


def _make_acrobot_config() -> EnvConfig:
    from gymnax.environments.classic_control.acrobot import EnvState
    def make_state(v):
        return EnvState(joint_angle1=jnp.arctan2(v[1], v[0]),
                        joint_angle2=jnp.arctan2(v[3], v[2]),
                        velocity_1=v[4], velocity_2=v[5], time=0)
    def start():
        return jnp.array([jnp.cos(0.05), jnp.sin(0.05),
                          jnp.cos(0.05), jnp.sin(0.05), 0.0, 0.0])
    return EnvConfig("Acrobot-v1", 6, 3, make_state, start)


ENV_REGISTRY = {
    "MountainCar-v0": _make_mountaincar_config,
    "Acrobot-v1": _make_acrobot_config,
}


# =============================================================================
# Args
# =============================================================================

@dataclass
class FromScratchGymnaxArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    envs: List[str] = field(default_factory=lambda: [
        "MountainCar-v0",
        "Acrobot-v1",
    ])

    # --- Budgets ---
    supervised_transitions: int = 1_000
    """Number of random (s,a,s') transitions for the supervised baseline."""

    supervised_steps: int = 2_500
    """Gradient steps for supervised baseline (samples from the fixed
    dataset with replacement)."""

    env_budgets: Dict[str, int] = field(default_factory=lambda: {
        "MountainCar-v0": 1_000_000,
        "Acrobot-v1":     1_000_000,
    })
    """Per-environment preference label budgets for DLHF."""
    default_budget: int = 1_000_000

    # --- Architecture ---
    hidden_dim: int = 128

    # --- Training (shared) ---
    lr:         float = 3e-4
    batch_size: int   = 64
    beta_pref:  float = 1.0

    # --- DLHF specifics ---
    segment_length: int = 1
    """Rollout horizon for preference segments."""
    num_candidates: int = 2
    """K — candidate rollouts per start state for K-candidate BT."""
    ensemble_size:  int = 1
    """Single model for from-scratch (no active selection)."""

    # --- Validation ---
    eval_every:   int = 100
    eval_samples: int = 2048

    # --- Output ---
    results_dir: Optional[str] = None
    overwrite: bool = False
    """If False, skip (env, seed, method) combos whose result.npz exists."""


def _get_budget(args: FromScratchGymnaxArgs, env: str) -> int:
    return args.env_budgets.get(env, args.default_budget)


# =============================================================================
# Model
# =============================================================================

class GaussianMLPDynamics(nn.Module):
    """Predicts delta: mean and heteroscedastic log-std per action."""
    obs_dim: int
    n_actions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, s, a) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a_oh = jax.nn.one_hot(a, self.n_actions)
        x = jnp.concatenate([s, a_oh], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.obs_dim)(x)
        log_std_table = self.param(
            "log_std_table", nn.initializers.constant(-2.0),
            (self.n_actions, self.obs_dim),
        )
        log_std = log_std_table[a]
        log_std = jnp.clip(log_std, -6.0, 1.0)
        return mean, log_std


def normal_log_prob(x, mean, log_std):
    z = (x - mean) * jnp.exp(-log_std)
    return jnp.sum(
        -0.5 * (z**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)), axis=-1)


# =============================================================================
# Oracle
# =============================================================================

def make_oracle_fn(env, env_params, env_cfg: EnvConfig):
    """Batched oracle: (B, obs_dim), (B,) -> (B, obs_dim)."""
    def true_next_state(state_vec, action):
        env_state = env_cfg.make_env_state(state_vec)
        obs, _, _, _, _ = env.step_env(
            jax.random.PRNGKey(0), env_state, action, env_params)
        return obs
    return jax.vmap(true_next_state)


# =============================================================================
# Shared helpers
# =============================================================================

def sample_random_states(key, n, obs_low, obs_high):
    return jax.random.uniform(key, shape=(n, obs_low.shape[0]),
                              minval=obs_low, maxval=obs_high)


def eval_mse(apply_fn, params, oracle_fn, key, obs_low, obs_high,
             n_actions, n):
    k1, k2 = jax.random.split(key)
    states = sample_random_states(k1, n, obs_low, obs_high)
    actions = jax.random.randint(k2, (n,), 0, n_actions, jnp.int32)
    true_ns = oracle_fn(states, actions)
    pred, _ = apply_fn({"params": params}, states, actions)
    return jnp.mean((pred - true_ns) ** 2)


# =============================================================================
# Imagination + preference generation (K-candidate BT)
# =============================================================================

def imagine_segment(params, apply_fn, s0, key, n_actions, T, obs_dim):
    key_a, key_eps = jax.random.split(key)
    a_seq = jax.random.randint(key_a, (T,), 0, n_actions, jnp.int32)
    eps = jax.random.normal(key_eps, (T, obs_dim))
    def step(s, inp):
        a, e = inp
        mean, log_std = apply_fn({"params": params}, s[None], a[None])
        ns = mean[0] + jnp.exp(log_std[0]) * e
        return ns, (s, a, ns)
    _, (s_seq, a_out, ns_seq) = lax.scan(step, s0, (a_seq, eps))
    return s_seq, a_out, ns_seq


def imagine_batch(params, apply_fn, s0s, key, n_actions, T, obs_dim):
    keys = jax.random.split(key, s0s.shape[0])
    return jax.vmap(lambda s, k: imagine_segment(
        params, apply_fn, s, k, n_actions, T, obs_dim))(s0s, keys)


def make_k_candidate_batch_fn(model, oracle_fn, obs_dim, n_actions, T, K):
    """K-candidate BT preference pairs with oracle labels."""

    def generate(params, s0, key, beta):
        B = s0.shape[0]
        k_keys = jax.random.split(key, K)
        def _one_k(k_key):
            return imagine_batch(params, model.apply, s0, k_key,
                                 n_actions, T, obs_dim)
        all_s, all_a, all_ns = jax.vmap(_one_k)(k_keys)
        # (K, B, T, obs_dim), (K, B, T), (K, B, T, obs_dim)

        def err_one(s, a, ns_im):
            ft = oracle_fn(s.reshape(-1, obs_dim), a.reshape(-1))
            d = ns_im.reshape(-1, obs_dim) - ft
            return jnp.sum(d * d, -1).reshape(B, T).sum(-1)
        errors = jax.vmap(err_one)(all_s, all_a, all_ns)  # (K, B)

        best_idx = jnp.argmin(errors, axis=0)
        b_arange = jnp.arange(B)
        best_s  = all_s[best_idx, b_arange]
        best_a  = all_a[best_idx, b_arange]
        best_ns = all_ns[best_idx, b_arange]
        best_err = errors[best_idx, b_arange]

        def _pair_one_k(ki, s_k, a_k, ns_k, err_k):
            is_best = (best_idx == ki)
            is_tie = (err_k == best_err)
            valid = ~is_best & ~is_tie
            label = jnp.where(valid, 1.0, 0.5)
            return s_k, a_k, ns_k, best_s, best_a, best_ns, label

        p_s1, p_a1, p_ns1, p_s2, p_a2, p_ns2, p_labels = jax.vmap(
            _pair_one_k)(jnp.arange(K), all_s, all_a, all_ns, errors)

        return {
            "s1":    p_s1.reshape(K * B, T, obs_dim),
            "a1":    p_a1.reshape(K * B, T),
            "ns1":   p_ns1.reshape(K * B, T, obs_dim),
            "s2":    p_s2.reshape(K * B, T, obs_dim),
            "a2":    p_a2.reshape(K * B, T),
            "ns2":   p_ns2.reshape(K * B, T, obs_dim),
            "label": p_labels.reshape(K * B),
        }
    return generate


def preference_loss_fn(params, apply_fn, batch, beta):
    var = {"params": params}
    def lp(s, a, ns):
        m, ls = apply_fn(var, s, a)
        return jnp.sum(normal_log_prob(ns, m, ls), -1)
    lp1 = lp(batch["s1"], batch["a1"], batch["ns1"])
    lp2 = lp(batch["s2"], batch["a2"], batch["ns2"])
    return jnp.mean(optax.sigmoid_binary_cross_entropy(
        beta * (lp2 - lp1), batch["label"]))


# =============================================================================
# Curves
# =============================================================================

def save_curves(losses, mses, eval_steps, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    axes[0].plot(losses, linewidth=0.6, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — final: {losses[-1]:.4f}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_steps, mses, linewidth=1.2, color="tab:red")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title(f"MSE — final: {mses[-1]:.6f}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Skip helper
# =============================================================================

def _load_cached(rd: str) -> Optional[dict]:
    path = os.path.join(rd, "result.npz")
    if not os.path.exists(path):
        return None
    data = dict(np.load(path, allow_pickle=True))
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim == 0:
            data[k] = v.item()
    return data


# =============================================================================
# Supervised baseline
# =============================================================================

def run_supervised(args: FromScratchGymnaxArgs, env_name: str, seed: int,
                   base_dir: str, env_cfg: EnvConfig, oracle_fn,
                   obs_low, obs_high):
    rd = os.path.join(base_dir, env_name, f"seed_{seed}", "supervised")

    if not args.overwrite:
        cached = _load_cached(rd)
        if cached is not None:
            print(f"  SKIP supervised {env_name} seed {seed} "
                  f"(final_mse={cached['final_mse']:.6f})")
            return cached

    os.makedirs(rd, exist_ok=True)

    N = args.supervised_transitions
    steps = args.supervised_steps
    B = args.batch_size
    obs_dim = env_cfg.obs_dim
    n_actions = env_cfg.n_actions

    model = GaussianMLPDynamics(obs_dim, n_actions, args.hidden_dim)
    opt = optax.adam(args.lr)

    # Init
    rng = jax.random.PRNGKey(seed)
    dummy_s = jnp.zeros((1, obs_dim))
    dummy_a = jnp.zeros((1,), jnp.int32)
    params = model.init(rng, dummy_s, dummy_a)["params"]
    opt_state = opt.init(params)

    # Collect fixed dataset
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + 999))
    ds_states = sample_random_states(k1, N, obs_low, obs_high)
    ds_actions = jax.random.randint(k2, (N,), 0, n_actions, jnp.int32)
    ds_next = oracle_fn(ds_states, ds_actions)

    @jax.jit
    def _eval(params):
        return eval_mse(model.apply, params, oracle_fn,
                        jax.random.PRNGKey(seed + 777),
                        obs_low, obs_high, n_actions, args.eval_samples)

    @jax.jit
    def _step(params, opt_state, rng):
        rng, bk = jax.random.split(rng)
        idx = jax.random.randint(bk, (B,), 0, N)
        s, a, ns = ds_states[idx], ds_actions[idx], ds_next[idx]
        def loss_fn(p):
            pred, _ = model.apply({"params": p}, s, a)
            return jnp.mean((pred - ns) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, rng, loss

    def _scan_chunk(carry, _):
        params, opt_state, rng = carry
        params, opt_state, rng, loss = _step(params, opt_state, rng)
        return (params, opt_state, rng), loss

    chunk = args.eval_every

    @jax.jit
    def _run_chunk(params, opt_state, rng):
        (params, opt_state, rng), losses = jax.lax.scan(
            _scan_chunk, (params, opt_state, rng), None, length=chunk)
        return params, opt_state, rng, losses

    print(f"\n{'='*60}")
    print(f"  SUPERVISED: {env_name} | seed {seed} | {steps} steps | "
          f"{N} transitions")
    print(f"{'='*60}")

    train_rng = jax.random.PRNGKey(seed + 456)
    all_losses = []
    mses = []
    eval_steps_list = []

    mse0 = float(_eval(params))
    mses.append(mse0)
    eval_steps_list.append(0)
    print(f"  init   | mse={mse0:.6f}")

    t0 = time.time()
    n_full = steps // chunk
    remainder = steps % chunk
    steps_done = 0

    for _ in range(n_full):
        params, opt_state, train_rng, chunk_losses = _run_chunk(
            params, opt_state, train_rng)
        jax.block_until_ready(params)
        steps_done += chunk
        all_losses.extend(np.array(chunk_losses).tolist())

        mse = float(_eval(params))
        mses.append(mse)
        eval_steps_list.append(steps_done)
        print(f"  step {steps_done:5d} | mse_loss={float(chunk_losses[-1]):.6f} | "
              f"val_mse={mse:.6f} | {time.time()-t0:.1f}s")

    if remainder > 0:
        for _ in range(remainder):
            params, opt_state, train_rng, loss = _step(
                params, opt_state, train_rng)
            all_losses.append(float(loss))
        steps_done += remainder
        jax.block_until_ready(params)
        mse = float(_eval(params))
        mses.append(mse)
        eval_steps_list.append(steps_done)
        print(f"  step {steps_done:5d} | val_mse={mse:.6f} | "
              f"{time.time()-t0:.1f}s")

    wall_time = time.time() - t0
    save_curves(all_losses, mses, eval_steps_list,
                os.path.join(rd, "curves.png"))

    result = dict(
        env=env_name, seed=seed, method="supervised",
        eval_steps=np.array(eval_steps_list),
        mses=np.array(mses),
        losses=np.array(all_losses),
        final_mse=mses[-1],
        wall_time=wall_time,
        total_steps=steps,
        supervised_transitions=N,
    )
    np.savez(os.path.join(rd, "result.npz"), **result)
    print(f"  Final mse={mses[-1]:.6f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# DLHF from scratch
# =============================================================================

def run_dlhf(args: FromScratchGymnaxArgs, env_name: str, seed: int,
             base_dir: str, budget: int, env_cfg: EnvConfig, oracle_fn,
             obs_low, obs_high):
    B = args.batch_size
    K = args.num_candidates
    labels_per_step = B * K  # K pairs per start state (including self)
    # Actually: K candidates -> K-1 non-self pairs, but the self pair
    # gets label 0.5 (tie), so all K pairs are used in the batch.
    # Total labels consumed per step = B * K.
    steps = budget // (B * K)

    rd = os.path.join(base_dir, env_name, f"seed_{seed}", "dlhf")

    if not args.overwrite:
        cached = _load_cached(rd)
        if cached is not None:
            print(f"  SKIP dlhf {env_name} seed {seed} "
                  f"(final_mse={cached['final_mse']:.6f})")
            return cached

    os.makedirs(rd, exist_ok=True)

    E = args.ensemble_size
    T = args.segment_length
    obs_dim = env_cfg.obs_dim
    n_actions = env_cfg.n_actions

    model = GaussianMLPDynamics(obs_dim, n_actions, args.hidden_dim)
    opt = optax.adam(args.lr)

    # Init (same seed scheme as supervised for fair comparison of init)
    rng = jax.random.PRNGKey(seed)
    dummy_s = jnp.zeros((1, obs_dim))
    dummy_a = jnp.zeros((1,), jnp.int32)

    if E == 1:
        params = model.init(rng, dummy_s, dummy_a)["params"]
        opt_state = opt.init(params)
    else:
        init_keys = jnp.stack([jax.random.PRNGKey(k)
                                for k in range(seed * 100, seed * 100 + E)])
        all_params = jax.vmap(
            lambda k: model.init(k, dummy_s, dummy_a)["params"])(init_keys)
        all_opt_states = jax.vmap(opt.init)(all_params)
        # Use member 0 for everything below
        params = jax.tree.map(lambda x: x[0], all_params)
        opt_state = jax.tree.map(lambda x: x[0], all_opt_states)

    gen_batch = make_k_candidate_batch_fn(
        model, oracle_fn, obs_dim, n_actions, T, K)

    @jax.jit
    def _eval(params):
        return eval_mse(model.apply, params, oracle_fn,
                        jax.random.PRNGKey(seed + 777),
                        obs_low, obs_high, n_actions, args.eval_samples)

    @jax.jit
    def _step(params, opt_state, rng):
        rng, k_sel, k_cand = jax.random.split(rng, 3)
        s0 = sample_random_states(k_sel, B, obs_low, obs_high)
        batch = gen_batch(params, s0, k_cand, args.beta_pref)

        loss, grads = jax.value_and_grad(preference_loss_fn)(
            params, model.apply, batch, args.beta_pref)
        updates, new_opt = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, rng, loss

    actual_labels = steps * B * K
    print(f"\n{'='*60}")
    print(f"  DLHF: {env_name} | seed {seed} | {steps} steps | "
          f"B={B} K={K} | {actual_labels:,} labels")
    print(f"{'='*60}")

    train_rng = jax.random.PRNGKey(seed + 123)
    all_losses = []
    mses = []
    eval_steps_list = []

    mse0 = float(_eval(params))
    mses.append(mse0)
    eval_steps_list.append(0)
    print(f"  init   | mse={mse0:.6f}")

    t0 = time.time()
    for i in range(1, steps + 1):
        params, opt_state, train_rng, loss = _step(
            params, opt_state, train_rng)
        all_losses.append(float(loss))

        if i % args.eval_every == 0:
            mse = float(_eval(params))
            mses.append(mse)
            eval_steps_list.append(i)
            print(f"  step {i:5d} | bt_loss={float(loss):.4f} | "
                  f"mse={mse:.6f} | {time.time()-t0:.1f}s")

    wall_time = time.time() - t0
    labels_at_eval = [s * B * K for s in eval_steps_list]

    save_curves(all_losses, mses, eval_steps_list,
                os.path.join(rd, "curves.png"))

    result = dict(
        env=env_name, seed=seed, method="dlhf",
        eval_steps=np.array(eval_steps_list),
        labels_at_eval=np.array(labels_at_eval),
        mses=np.array(mses),
        losses=np.array(all_losses),
        final_mse=mses[-1],
        wall_time=wall_time,
        total_steps=steps,
        preference_budget=budget,
    )
    np.savez(os.path.join(rd, "result.npz"), **result)
    print(f"  Final mse={mses[-1]:.6f} ({wall_time:.1f}s)")
    return result


# =============================================================================
# Summary plot
# =============================================================================

def save_summary_plot(all_results, args: FromScratchGymnaxArgs, base_dir):
    envs = args.envs
    n_envs = len(envs)
    cols = min(n_envs, 4)
    rows = (n_envs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), dpi=130,
                             squeeze=False)

    colors = {"dlhf": "tab:red", "supervised": "tab:blue"}

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]
        budget = _get_budget(args, env)

        labels_map = {
            "dlhf": f"DLHF ({budget:,} prefs)",
            "supervised": f"Supervised ({args.supervised_transitions:,} trans)",
        }

        for method in ["supervised", "dlhf"]:
            method_results = [
                r for r in all_results
                if r["env"] == env and r["method"] == method
            ]
            if not method_results:
                continue

            color = colors[method]

            # X-axis: training steps for both (different data regimes)
            ref_steps = method_results[0]["eval_steps"]
            aligned = [r["mses"] for r in method_results
                       if len(r["mses"]) == len(ref_steps)]

            # Individual seeds
            for r in method_results:
                if len(r["mses"]) == len(ref_steps):
                    ax.plot(r["eval_steps"], r["mses"],
                            linewidth=0.5, alpha=0.25, color=color)

            if aligned:
                arr = np.array(aligned)
                n = len(aligned)
                mean = arr.mean(0)
                ax.plot(ref_steps, mean, linewidth=2, color=color,
                        label=f"{labels_map[method]} (n={n})")
                if n > 1:
                    ci = 1.96 * arr.std(0, ddof=1) / np.sqrt(n)
                    ax.fill_between(ref_steps, mean - ci, mean + ci,
                                    alpha=0.12, color=color)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Val MSE")
        ax.set_title(env, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(n_envs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"From-Scratch: DLHF vs Supervised (Gymnax)\n"
        f"{len(args.seeds)} seeds",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(base_dir, "summary.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary plot -> {path}")


# =============================================================================
# Results table
# =============================================================================

def save_results_table(all_results, args: FromScratchGymnaxArgs, base_dir):
    print(f"\n{'='*80}")
    print(f"  FROM-SCRATCH RESULTS (Gymnax)")
    print(f"  Supervised: {args.supervised_transitions:,} transitions, "
          f"{args.supervised_steps:,} steps")
    print(f"{'='*80}")

    header = (f"  {'Env':<18s}  {'Budget':>8s}  "
              f"{'Supervised':>24s}  {'DLHF':>24s}  {'Gap':>12s}")
    sep = (f"  {'-'*18}  {'-'*8}  {'-'*24}  {'-'*24}  {'-'*12}")
    print(header); print(sep)

    lines = [header, sep]
    for env in args.envs:
        budget = _get_budget(args, env)

        def _fmt(method):
            results = [r for r in all_results
                       if r["env"] == env and r["method"] == method]
            if not results:
                return "---", None
            finals = [r["final_mse"] for r in results]
            m = np.mean(finals)
            n = len(finals)
            ci = 1.96 * np.std(finals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return f"{m:.6f} ± {ci:.6f}", m

        sup_str, sup_m = _fmt("supervised")
        dlhf_str, dlhf_m = _fmt("dlhf")
        gap = (f"{dlhf_m - sup_m:+.6f}"
               if sup_m is not None and dlhf_m is not None else "---")

        line = (f"  {env:<18s}  {budget:>8,}  "
                f"{sup_str:>24s}  {dlhf_str:>24s}  {gap:>12s}")
        print(line); lines.append(line)

    path = os.path.join(base_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("From-Scratch: DLHF vs Supervised (Gymnax)\n")
        f.write(f"Supervised: {args.supervised_transitions} transitions, "
                f"{args.supervised_steps} steps\n")
        f.write(f"Seeds: {args.seeds}\n\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Main
# =============================================================================

def main(args: FromScratchGymnaxArgs) -> None:
    base_dir = args.results_dir or os.path.join(OUT_DIR, "from_scratch_gymnax")
    os.makedirs(base_dir, exist_ok=True)

    B = args.batch_size
    K = args.num_candidates

    print(f"\n{'='*60}")
    print(f"  SECTION 4.1: FROM-SCRATCH DLHF vs SUPERVISED (Gymnax)")
    print(f"{'='*60}")
    print(f"  Envs:              {args.envs}")
    print(f"  Seeds:             {args.seeds}")
    print(f"  Supervised:        {args.supervised_transitions:,} "
          f"transitions, {args.supervised_steps:,} steps")
    for env in args.envs:
        budget = _get_budget(args, env)
        dlhf_steps = budget // (B * K)
        print(f"  DLHF ({env}): {budget:,} labels, {dlhf_steps:,} steps")
    print(f"  Batch size:        {B}")
    print(f"  K (candidates):    {K}")
    print(f"  Ensemble:          {args.ensemble_size}")
    print(f"  Overwrite:         {args.overwrite}")

    for env_name in args.envs:
        if env_name not in ENV_REGISTRY:
            raise ValueError(
                f"Unknown env '{env_name}'. "
                f"Supported: {list(ENV_REGISTRY.keys())}")

    all_results = []

    for env_name in args.envs:
        env_cfg = ENV_REGISTRY[env_name]()
        env, env_params = gymnax.make(env_cfg.gymnax_name)
        obs_space = env.observation_space(env_params)
        obs_low = jnp.array(obs_space.low)
        obs_high = jnp.array(obs_space.high)
        oracle_fn = make_oracle_fn(env, env_params, env_cfg)
        budget = _get_budget(args, env_name)

        for seed in args.seeds:
            # --- Supervised ---
            try:
                result = run_supervised(
                    args, env_name, seed, base_dir,
                    env_cfg, oracle_fn, obs_low, obs_high)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR: supervised {env_name} seed {seed}: {e}")
                import traceback; traceback.print_exc()

            # --- DLHF ---
            try:
                result = run_dlhf(
                    args, env_name, seed, base_dir, budget,
                    env_cfg, oracle_fn, obs_low, obs_high)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ERROR: dlhf {env_name} seed {seed}: {e}")
                import traceback; traceback.print_exc()

    if not all_results:
        print("No successful runs. Exiting.")
        return

    save_summary_plot(all_results, args, base_dir)
    save_results_table(all_results, args, base_dir)
    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(FromScratchGymnaxArgs))