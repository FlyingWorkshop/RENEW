"""
compare_gymnax.py
=================
Sample efficiency comparison: Naive DLHF vs RENEW on Gymnax envs.

Both methods train an ensemble of Gaussian MLP world models using only
preference labels. The ONLY difference:

    Naive:  random start states and actions (uniform)
    RENEW:  active (state, action) selection via ensemble disagreement

Fair comparison: both conditions use the SAME labels-per-step.
    - RENEW uses B start states and produces B*(K-1) labels per step.
    - Naive uses B*(K-1) as its batch size, also producing B*(K-1) labels
      per step.  This isolates active selection as the only variable.

Everything else is identical:
    - Same ensemble size (E members)
    - Same architecture, lr, beta, segment length
    - Same total preference label budget
    - Same number of training steps

Optionally, both methods can be warm-started with a short offline
(supervised MSE) pretraining phase on random transitions.

Output structure (no pretrain):
    out/compare_gymnax/{env}/seed_{s}/naive/result.npz, curves.png
    out/compare_gymnax/{env}/seed_{s}/renew/result.npz, curves.png
    out/compare_gymnax/comparison.png, summary.txt

Output structure (with pretrain):
    out/compare_gymnax_pretrained/{env}/seed_{s}/pretrained_0.pkl, ...
    out/compare_gymnax_pretrained/{env}/seed_{s}/naive/result.npz, curves.png
    out/compare_gymnax_pretrained/{env}/seed_{s}/renew/result.npz, curves.png
    out/compare_gymnax_pretrained/comparison.png, summary.txt

Usage:
    # Quick test
    python compare_gymnax.py --envs MountainCar-v0 --seeds 0

    # No pretraining
    python compare_gymnax.py --no-pretrain --envs MountainCar-v0 --seeds 0

    # Full run
    python compare_gymnax.py --envs MountainCar-v0 Acrobot-v1 \
        --seeds 0 1 2 3 4

    # Override budget
    python compare_gymnax.py --env-budgets.MountainCar-v0 50000

    # Rerun finetuning only
    python compare_gymnax.py --overwrite
"""

from __future__ import annotations

import os
import pickle
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
class CompareArgs:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # seeds: List[int] = field(default_factory=lambda: [0, 1])
    envs: List[str] = field(default_factory=lambda: [
        "MountainCar-v0",
        "Acrobot-v1",
    ])

    # --- Overwrite / rerun control ---
    overwrite: bool = False
    """Re-run finetuning even if result.npz exists. Also remakes plots."""
    overwrite_pretrain: bool = False
    """Re-run pretraining even if pretrained_*.pkl exist. Also remakes plots."""

    # --- Pretraining ---
    pretrain: bool = True
    """If True, warm-start with supervised MSE pretraining."""
    pretrain_steps: int = 500
    """Number of supervised pretraining gradient steps."""
    pretrain_batch: int = 32
    """Batch size for pretraining."""
    pretrain_dataset_size: int = 100
    """Number of random (s, a, s') transitions to collect for pretraining."""
    pretrain_checkpoint: Optional[str] = None
    """Path to directory with pretrained_0.pkl, ... to skip pretraining."""

    # --- Budget ---
    env_budgets: Dict[str, int] = field(default_factory=lambda: {
        "MountainCar-v0": 100_000,
        "Acrobot-v1":     100_000,
    })
    """Per-environment preference label budgets."""
    default_budget: int = 50_000
    """Fallback budget for envs not listed in env_budgets."""

    # --- Architecture ---
    hidden_dim: int = 128

    # --- Training ---
    lr:             float = 3e-3
    batch_size:     int   = 64
    segment_length: int   = 1
    beta_pref:      float = 1.0

    # --- Ensemble ---
    ensemble_size: int = 3

    # --- RENEW-specific ---
    num_candidates:   int = 2
    """K — candidate rollouts per start state."""
    candidate_pool:   int = 1024
    """Pool size for active start-state selection (= batch_size * 16)."""

    # --- Validation ---
    eval_every:   int = 10
    eval_samples: int = 2048

    # --- Output ---
    results_dir: Optional[str] = None


# =============================================================================
# Budget resolver
# =============================================================================

def _get_budget(cargs: CompareArgs, env: str) -> int:
    return cargs.env_budgets.get(env, cargs.default_budget)


def _budget_str(cargs: CompareArgs) -> str:
    return ", ".join(f"{e}={_get_budget(cargs, e):,}" for e in cargs.envs)


# =============================================================================
# Model
# =============================================================================

class GaussianMLPDynamics(nn.Module):
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
    return jnp.sum(-0.5 * (z**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)), axis=-1)


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
# Shared primitives
# =============================================================================

def sample_random_states(key, n, obs_low, obs_high):
    return jax.random.uniform(key, shape=(n, obs_low.shape[0]),
                              minval=obs_low, maxval=obs_high)


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


def imagine_segment_fixed(params, apply_fn, s0, a_seq, key, obs_dim):
    """Like imagine_segment but with pre-selected actions."""
    T = a_seq.shape[0]
    eps = jax.random.normal(key, (T, obs_dim))
    def step(s, inp):
        a, e = inp
        mean, log_std = apply_fn({"params": params}, s[None], a[None])
        ns = mean[0] + jnp.exp(log_std[0]) * e
        return ns, (s, a, ns)
    _, (s_seq, a_out, ns_seq) = lax.scan(step, s0, (a_seq, eps))
    return s_seq, a_out, ns_seq


def imagine_batch_fixed(params, apply_fn, s0s, actions, key, obs_dim):
    """Like imagine_batch but with pre-selected actions per example."""
    keys = jax.random.split(key, s0s.shape[0])
    return jax.vmap(lambda s, a, k: imagine_segment_fixed(
        params, apply_fn, s, a, k, obs_dim))(s0s, actions, keys)


def preference_loss_fn(params, apply_fn, batch, beta):
    var = {"params": params}
    def lp(s, a, ns):
        m, ls = apply_fn(var, s, a)
        return jnp.sum(normal_log_prob(ns, m, ls), -1)
    lp1 = lp(batch["s1"], batch["a1"], batch["ns1"])
    lp2 = lp(batch["s2"], batch["a2"], batch["ns2"])
    return jnp.mean(optax.sigmoid_binary_cross_entropy(
        beta * (lp2 - lp1), batch["label"]))


def eval_mse(apply_fn, params, oracle_fn, key, obs_low, obs_high,
             n_actions, n):
    k1, k2 = jax.random.split(key)
    states = sample_random_states(k1, n, obs_low, obs_high)
    actions = jax.random.randint(k2, (n,), 0, n_actions, jnp.int32)
    true_ns = oracle_fn(states, actions)
    pred, _ = apply_fn({"params": params}, states, actions)
    return jnp.mean((pred - true_ns) ** 2)


# =============================================================================
# Pretrain checkpoint I/O
# =============================================================================

def _save_pretrain_checkpoint(all_params, checkpoint_dir, ensemble_size):
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i in range(ensemble_size):
        p = jax.tree.map(lambda x: np.array(x[i]), all_params)
        path = os.path.join(checkpoint_dir, f"pretrained_{i}.pkl")
        with open(path, "wb") as f:
            pickle.dump(p, f)
    print(f"  Saved pretrain checkpoint ({ensemble_size} members) → {checkpoint_dir}")


def _load_pretrain_checkpoint(checkpoint_dir, ensemble_size):
    members = []
    for i in range(ensemble_size):
        path = os.path.join(checkpoint_dir, f"pretrained_{i}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pretrain checkpoint not found: {path}")
        with open(path, "rb") as f:
            members.append(pickle.load(f))
    stacked = jax.tree.map(lambda *ps: jnp.stack(ps), *members)
    print(f"  Loaded pretrain checkpoint ({ensemble_size} members) ← {checkpoint_dir}")
    return stacked


# =============================================================================
# Offline pretraining (supervised MSE)
# =============================================================================

def collect_offline_dataset(oracle_fn, obs_low, obs_high, n_actions,
                            size, seed):
    """Collect random (s, a, s') transitions."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed + 999_999))
    states = sample_random_states(k1, size, obs_low, obs_high)
    actions = jax.random.randint(k2, (size,), 0, n_actions, jnp.int32)
    next_states = oracle_fn(states, actions)
    print(f"  Collected offline dataset ({size} transitions)")
    return dict(states=states, actions=actions, next_states=next_states)


def pretrain_ensemble(model, all_params, all_opt_states, dataset,
                      cargs: CompareArgs, seed: int):
    """Supervised MSE pretraining for the full ensemble via vmap."""
    N = dataset["states"].shape[0]
    opt = optax.adam(cargs.lr)

    @jax.jit
    def _sample(key):
        idx = jax.random.randint(key, (cargs.pretrain_batch,), 0, N)
        return (dataset["states"][idx], dataset["actions"][idx],
                dataset["next_states"][idx])

    def _single_step(params, opt_state, s, a, ns):
        def loss_fn(p):
            pred, _ = model.apply({"params": p}, s, a)
            return jnp.mean((pred - ns) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss

    @jax.jit
    def _ensemble_step(all_p, all_o, rng):
        rng, bk = jax.random.split(rng)
        s, a, ns = _sample(bk)
        new_p, new_o, losses = jax.vmap(
            lambda p, o: _single_step(p, o, s, a, ns)
        )(all_p, all_o)
        return new_p, new_o, rng, losses[0]

    @jax.jit
    def _scan_chunk(carry, _):
        all_p, all_o, rng = carry
        all_p, all_o, rng, loss = _ensemble_step(all_p, all_o, rng)
        return (all_p, all_o, rng), loss

    chunk_size = min(50, cargs.pretrain_steps)
    def _run_chunk(all_p, all_o, rng, length):
        (all_p, all_o, rng), losses = jax.lax.scan(
            _scan_chunk, (all_p, all_o, rng), None, length=length)
        return all_p, all_o, rng, losses

    rng = jax.random.PRNGKey(seed + 42)
    total = cargs.pretrain_steps
    n_full = total // chunk_size
    remainder = total % chunk_size

    print(f"  Pretraining ensemble ({cargs.ensemble_size} members, "
          f"{total} steps, batch={cargs.pretrain_batch})...")
    t0 = time.time()
    all_losses = []

    for _ in range(n_full):
        all_params, all_opt_states, rng, chunk_losses = _run_chunk(
            all_params, all_opt_states, rng, chunk_size)
        jax.block_until_ready(all_params)
        all_losses.extend(np.array(chunk_losses).tolist())

    if remainder > 0:
        all_params, all_opt_states, rng, chunk_losses = _run_chunk(
            all_params, all_opt_states, rng, remainder)
        jax.block_until_ready(all_params)
        all_losses.extend(np.array(chunk_losses).tolist())

    elapsed = time.time() - t0
    print(f"  Pretrain done: {total} steps in {elapsed:.1f}s | "
          f"final loss={all_losses[-1]:.6f}")
    return all_params, all_opt_states


# =============================================================================
# K-candidate batch generation
# =============================================================================

def make_k_candidate_batch_fn(model, oracle_fn, obs_dim, n_actions, T, K):
    """Returns a function that generates K-candidate preference pairs."""

    def generate(params_0, s0, key, beta):
        B = s0.shape[0]
        # K candidate rollouts from member 0
        k_keys = jax.random.split(key, K)
        def _one_k(k_key):
            return imagine_batch(params_0, model.apply, s0, k_key,
                                 n_actions, T, obs_dim)
        all_s, all_a, all_ns = jax.vmap(_one_k)(k_keys)
        # (K, B, T, obs_dim), (K, B, T), (K, B, T, obs_dim)

        # Oracle errors
        def err_one(s, a, ns_im):
            ft = oracle_fn(s.reshape(-1, obs_dim), a.reshape(-1))
            d = ns_im.reshape(-1, obs_dim) - ft
            return jnp.sum(d * d, -1).reshape(B, T).sum(-1)
        errors = jax.vmap(err_one)(all_s, all_a, all_ns)  # (K, B)

        # Best candidate per example
        best_idx = jnp.argmin(errors, axis=0)  # (B,)
        b_arange = jnp.arange(B)
        best_s = all_s[best_idx, b_arange]
        best_a = all_a[best_idx, b_arange]
        best_ns = all_ns[best_idx, b_arange]
        best_err = errors[best_idx, b_arange]

        # Form K preference pairs: best vs each
        def _pair_one_k(ki, s_k, a_k, ns_k, err_k):
            is_best = (best_idx == ki)
            is_tie = (err_k == best_err)
            valid = ~is_best & ~is_tie
            label = jnp.where(valid, 1.0, 0.5)
            return s_k, a_k, ns_k, best_s, best_a, best_ns, label

        p_s1, p_a1, p_ns1, p_s2, p_a2, p_ns2, p_labels = jax.vmap(
            _pair_one_k)(jnp.arange(K), all_s, all_a, all_ns, errors)

        return {
            "s1": p_s1.reshape(K * B, T, obs_dim),
            "a1": p_a1.reshape(K * B, T),
            "ns1": p_ns1.reshape(K * B, T, obs_dim),
            "s2": p_s2.reshape(K * B, T, obs_dim),
            "a2": p_a2.reshape(K * B, T),
            "ns2": p_ns2.reshape(K * B, T, obs_dim),
            "label": p_labels.reshape(K * B),
        }
    return generate


def make_k_candidate_batch_fn_fixed(model, oracle_fn, obs_dim, T, K):
    """K-candidate preference pairs with pre-selected actions (for RENEW).

    Unlike the random-action version, all K candidates share the same
    (s0, action) pair and differ only in sampling noise. This pairs with
    joint (state, action) active selection.
    """

    def generate(params_0, s0, actions, key):
        B = s0.shape[0]
        k_keys = jax.random.split(key, K)
        def _one_k(k_key):
            return imagine_batch_fixed(params_0, model.apply, s0, actions,
                                       k_key, obs_dim)
        all_s, all_a, all_ns = jax.vmap(_one_k)(k_keys)
        # (K, B, T, obs_dim), (K, B, T), (K, B, T, obs_dim)

        # Oracle errors
        def err_one(s, a, ns_im):
            ft = oracle_fn(s.reshape(-1, obs_dim), a.reshape(-1))
            d = ns_im.reshape(-1, obs_dim) - ft
            return jnp.sum(d * d, -1).reshape(B, T).sum(-1)
        errors = jax.vmap(err_one)(all_s, all_a, all_ns)  # (K, B)

        # Best candidate per example
        best_idx = jnp.argmin(errors, axis=0)  # (B,)
        b_arange = jnp.arange(B)
        best_s = all_s[best_idx, b_arange]
        best_a = all_a[best_idx, b_arange]
        best_ns = all_ns[best_idx, b_arange]
        best_err = errors[best_idx, b_arange]

        # Form K preference pairs: best vs each
        def _pair_one_k(ki, s_k, a_k, ns_k, err_k):
            is_best = (best_idx == ki)
            is_tie = (err_k == best_err)
            valid = ~is_best & ~is_tie
            label = jnp.where(valid, 1.0, 0.5)
            return s_k, a_k, ns_k, best_s, best_a, best_ns, label

        p_s1, p_a1, p_ns1, p_s2, p_a2, p_ns2, p_labels = jax.vmap(
            _pair_one_k)(jnp.arange(K), all_s, all_a, all_ns, errors)

        return {
            "s1": p_s1.reshape(K * B, T, obs_dim),
            "a1": p_a1.reshape(K * B, T),
            "ns1": p_ns1.reshape(K * B, T, obs_dim),
            "s2": p_s2.reshape(K * B, T, obs_dim),
            "a2": p_a2.reshape(K * B, T),
            "ns2": p_ns2.reshape(K * B, T, obs_dim),
            "label": p_labels.reshape(K * B),
        }
    return generate


# =============================================================================
# Ensemble disagreement scoring
# =============================================================================

def make_score_fn(model):
    """Score (state, action) pairs by ensemble disagreement on predicted means."""
    def score(all_params, candidates, actions):
        def pred_one(p):
            m, _ = model.apply({"params": p}, candidates, actions)
            return m
        all_means = jax.vmap(pred_one)(all_params)  # (E, pool, obs_dim)
        return jnp.sum(jnp.var(all_means, axis=0), axis=-1)  # (pool,)
    return score


# =============================================================================
# Training curves
# =============================================================================

def save_curves(losses, mses, eval_steps, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    axes[0].plot(losses, linewidth=0.6, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("BT Loss")
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
    print(f"  Curves → {path}")


# =============================================================================
# Run one (env, seed, method)
# =============================================================================

def run_naive(cargs: CompareArgs, env_name: str, seed: int, out_dir: str,
              budget: int, env_cfg: EnvConfig, env, env_params,
              oracle_fn, obs_low, obs_high,
              pretrained_params=None):
    """Naive: random start states, K-candidate preferences."""
    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)
    naive_bs = labels_per_step
    steps = budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    E = cargs.ensemble_size
    T = cargs.segment_length
    obs_dim = env_cfg.obs_dim
    n_actions = env_cfg.n_actions

    model = GaussianMLPDynamics(obs_dim, n_actions, cargs.hidden_dim)
    opt = optax.adam(cargs.lr)

    if pretrained_params is not None:
        all_params = pretrained_params
        all_opt_states = jax.vmap(opt.init)(all_params)
    else:
        init_keys = jnp.stack([jax.random.PRNGKey(k)
                                for k in range(seed * 100, seed * 100 + E)])
        dummy_s = jnp.zeros((1, obs_dim))
        dummy_a = jnp.zeros((1,), jnp.int32)
        all_params = jax.vmap(
            lambda k: model.init(k, dummy_s, dummy_a)["params"])(init_keys)
        all_opt_states = jax.vmap(opt.init)(all_params)

    gen_batch = make_k_candidate_batch_fn(
        model, oracle_fn, obs_dim, n_actions, T, K)

    @jax.jit
    def _eval_mse(params):
        return eval_mse(model.apply, params, oracle_fn,
                        jax.random.PRNGKey(seed + 777),
                        obs_low, obs_high, n_actions, cargs.eval_samples)

    @jax.jit
    def _step(all_p, all_o, rng):
        rng, k_sel, k_cand = jax.random.split(rng, 3)
        s0 = sample_random_states(k_sel, naive_bs, obs_low, obs_high)
        p0 = jax.tree.map(lambda x: x[0], all_p)
        batch = gen_batch(p0, s0, k_cand, cargs.beta_pref)

        def _update_one(p, o):
            loss, grads = jax.value_and_grad(preference_loss_fn)(
                p, model.apply, batch, cargs.beta_pref)
            updates, new_o = opt.update(grads, o)
            return optax.apply_updates(p, updates), new_o, loss
        new_p, new_o, losses = jax.vmap(_update_one)(all_p, all_o)
        return new_p, new_o, rng, jnp.mean(losses)

    pt_tag = " (pretrained)" if pretrained_params is not None else ""
    print(f"\n{'='*60}")
    print(f"  NAIVE{pt_tag}: {env_name} | seed {seed} | {steps} steps | "
          f"B={naive_bs} | {labels_per_step} labels/step | "
          f"{budget:,} labels | E={E}")
    print(f"{'='*60}")

    rng = jax.random.PRNGKey(seed + 123)
    all_losses = []
    mses = []
    eval_steps_list = []

    p0 = jax.tree.map(lambda x: x[0], all_params)
    mse0 = float(_eval_mse(p0))
    mses.append(mse0)
    eval_steps_list.append(0)
    print(f"  init   | mse={mse0:.6f}")

    t0 = time.time()
    for i in range(1, steps + 1):
        all_params, all_opt_states, rng, loss = _step(
            all_params, all_opt_states, rng)
        all_losses.append(float(loss))

        if i % cargs.eval_every == 0:
            p0 = jax.tree.map(lambda x: x[0], all_params)
            mse = float(_eval_mse(p0))
            mses.append(mse)
            eval_steps_list.append(i)
            print(f"  step {i:5d} | bt_loss={float(loss):.4f} | "
                  f"mse={mse:.6f} | {time.time()-t0:.1f}s")

    wall_time = time.time() - t0
    labels_at_eval = [s * labels_per_step for s in eval_steps_list]

    save_curves(all_losses, mses, eval_steps_list,
                os.path.join(out_dir, "curves.png"))

    result = dict(
        env=env_name, seed=seed, method="naive",
        eval_steps=np.array(eval_steps_list),
        labels=np.array(labels_at_eval),
        mses=np.array(mses),
        losses=np.array(all_losses),
        final_mse=mses[-1],
        wall_time=wall_time,
        budget=budget,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final mse={mses[-1]:.6f} ({wall_time:.1f}s)")
    return result


def run_renew(cargs: CompareArgs, env_name: str, seed: int, out_dir: str,
              budget: int, env_cfg: EnvConfig, env, env_params,
              oracle_fn, obs_low, obs_high,
              pretrained_params=None):
    """RENEW: active (state, action) selection + K-candidate preferences."""
    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)
    steps = budget // labels_per_step
    os.makedirs(out_dir, exist_ok=True)

    E = cargs.ensemble_size
    T = cargs.segment_length
    obs_dim = env_cfg.obs_dim
    n_actions = env_cfg.n_actions
    pool_size = cargs.candidate_pool

    model = GaussianMLPDynamics(obs_dim, n_actions, cargs.hidden_dim)
    opt = optax.adam(cargs.lr)

    if pretrained_params is not None:
        all_params = pretrained_params
        all_opt_states = jax.vmap(opt.init)(all_params)
    else:
        init_keys = jnp.stack([jax.random.PRNGKey(k)
                                for k in range(seed * 100, seed * 100 + E)])
        dummy_s = jnp.zeros((1, obs_dim))
        dummy_a = jnp.zeros((1,), jnp.int32)
        all_params = jax.vmap(
            lambda k: model.init(k, dummy_s, dummy_a)["params"])(init_keys)
        all_opt_states = jax.vmap(opt.init)(all_params)

    gen_batch = make_k_candidate_batch_fn_fixed(
        model, oracle_fn, obs_dim, T, K)
    score_fn = make_score_fn(model)

    @jax.jit
    def _eval_mse(params):
        return eval_mse(model.apply, params, oracle_fn,
                        jax.random.PRNGKey(seed + 777),
                        obs_low, obs_high, n_actions, cargs.eval_samples)

    @jax.jit
    def _step(all_p, all_o, rng):
        rng, k_pool, k_acts, k_cand = jax.random.split(rng, 4)

        # Generate pool of (state, action) pairs, score jointly, select top-B
        pool_states = sample_random_states(k_pool, pool_size, obs_low, obs_high)
        pool_actions = jax.random.randint(k_acts, (pool_size,), 0, n_actions,
                                           jnp.int32)
        scores = score_fn(all_p, pool_states, pool_actions)
        top_idx = jax.lax.top_k(scores, B)[1]
        s0 = pool_states[top_idx]
        # Tile selected action to (B, T) for segment generation
        a0 = jnp.tile(pool_actions[top_idx][:, None], (1, T))

        # K-candidate batch with fixed actions + update
        p0 = jax.tree.map(lambda x: x[0], all_p)
        batch = gen_batch(p0, s0, a0, k_cand)

        def _update_one(p, o):
            loss, grads = jax.value_and_grad(preference_loss_fn)(
                p, model.apply, batch, cargs.beta_pref)
            updates, new_o = opt.update(grads, o)
            return optax.apply_updates(p, updates), new_o, loss
        new_p, new_o, losses = jax.vmap(_update_one)(all_p, all_o)
        return new_p, new_o, rng, jnp.mean(losses)

    pt_tag = " (pretrained)" if pretrained_params is not None else ""
    print(f"\n{'='*60}")
    print(f"  RENEW{pt_tag}: {env_name} | seed {seed} | {steps} steps | "
          f"B={B} | {labels_per_step} labels/step | "
          f"{budget:,} labels | E={E} K={K}")
    print(f"{'='*60}")

    rng = jax.random.PRNGKey(seed + 123)
    all_losses = []
    mses = []
    eval_steps_list = []

    p0 = jax.tree.map(lambda x: x[0], all_params)
    mse0 = float(_eval_mse(p0))
    mses.append(mse0)
    eval_steps_list.append(0)
    print(f"  init   | mse={mse0:.6f}")

    t0 = time.time()
    for i in range(1, steps + 1):
        all_params, all_opt_states, rng, loss = _step(
            all_params, all_opt_states, rng)
        all_losses.append(float(loss))

        if i % cargs.eval_every == 0:
            p0 = jax.tree.map(lambda x: x[0], all_params)
            mse = float(_eval_mse(p0))
            mses.append(mse)
            eval_steps_list.append(i)
            print(f"  step {i:5d} | bt_loss={float(loss):.4f} | "
                  f"mse={mse:.6f} | {time.time()-t0:.1f}s")

    wall_time = time.time() - t0
    labels_at_eval = [s * labels_per_step for s in eval_steps_list]

    save_curves(all_losses, mses, eval_steps_list,
                os.path.join(out_dir, "curves.png"))

    result = dict(
        env=env_name, seed=seed, method="renew",
        eval_steps=np.array(eval_steps_list),
        labels=np.array(labels_at_eval),
        mses=np.array(mses),
        losses=np.array(all_losses),
        final_mse=mses[-1],
        wall_time=wall_time,
        budget=budget,
    )
    np.savez(os.path.join(out_dir, "result.npz"), **result)
    print(f"  Final mse={mses[-1]:.6f} ({wall_time:.1f}s)")
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
                ax.plot(r["labels"], r["mses"],
                        linewidth=0.5, alpha=0.3, color=color)

            ref_labels = method_results[0]["labels"]
            aligned = [r["mses"] for r in method_results
                       if len(r["mses"]) == len(ref_labels)]
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
        ax.set_ylabel("Val MSE")
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


# =============================================================================
# Results table
# =============================================================================

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
              f"{cargs.pretrain_dataset_size} transitions")
    print(f"{'='*80}")
    header = (f"  {'Env':<18s}  {'Budget':>8s}  {init_label:>20s}  "
              f"{'Naive':>20s}  {'RENEW':>20s}  {'Gap':>10s}")
    sep = (f"  {'-'*18}  {'-'*8}  {'-'*20}  "
           f"{'-'*20}  {'-'*20}  {'-'*10}")
    print(header); print(sep)

    lines = [header, sep]
    for env in envs:
        budget = _get_budget(cargs, env)

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

        def _fmt_init():
            results = [r for r in all_results
                       if r["env"] == env and "mses" in r
                       and len(r["mses"]) > 0]
            if not results:
                return "---"
            inits = [float(r["mses"][0]) for r in results]
            m = np.mean(inits)
            n = len(inits)
            ci = 1.96 * np.std(inits, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            return f"{m:.6f} ± {ci:.6f}"

        init_str = _fmt_init()
        naive_str, naive_m = _fmt("naive")
        renew_str, renew_m = _fmt("renew")
        gap = (f"{renew_m - naive_m:+.6f}"
               if naive_m is not None and renew_m is not None else "---")
        line = (f"  {env:<18s}  {budget:>8,}  {init_str:>20s}  "
                f"{naive_str:>20s}  {renew_str:>20s}  {gap:>10s}")
        print(line); lines.append(line)

    path = os.path.join(base_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(f"Naive vs RENEW{pt_tag} Results\n")
        f.write(f"E={cargs.ensemble_size}, K={cargs.num_candidates}\n")
        f.write(f"Labels/step: {labels_per_step}\n")
        if cargs.pretrain:
            f.write(f"Pretrain: {cargs.pretrain_steps} steps on "
                    f"{cargs.pretrain_dataset_size} transitions\n")
        f.write("\n")
        f.write("\n".join(lines))
    print(f"  Table -> {path}")


# =============================================================================
# Per-seed printout + plot
# =============================================================================

def _print_seed_result(env, seed, seed_results):
    naive_r = next((r for r in seed_results if r["method"] == "naive"), None)
    renew_r = next((r for r in seed_results if r["method"] == "renew"), None)
    if naive_r is None or renew_r is None:
        return

    naive_mse = naive_r["final_mse"]
    renew_mse = renew_r["final_mse"]
    gap = naive_mse - renew_mse

    if gap > 0:
        winner = "RENEW wins"
    elif gap < 0:
        winner = "Naive wins"
    else:
        winner = "Tie"

    print(f"\n  --- {env} seed {seed} ---")
    print(f"  Naive MSE = {naive_mse:.6f}")
    print(f"  RENEW MSE = {renew_mse:.6f}")
    print(f"  Gap (naive - renew) = {gap:+.6f}  ({winner})")


def _save_seed_comparison_plot(seed_dir, env, seed, seed_results,
                               force=False, pretrain=False):
    path = os.path.join(seed_dir, "comparison.png")
    if os.path.exists(path) and not force:
        return

    naive_r = next((r for r in seed_results if r["method"] == "naive"), None)
    renew_r = next((r for r in seed_results if r["method"] == "renew"), None)
    if naive_r is None or renew_r is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)

    ax.plot(naive_r["labels"], naive_r["mses"],
            linewidth=1.5, color="tab:blue", label="Naive")
    ax.plot(renew_r["labels"], renew_r["mses"],
            linewidth=1.5, color="tab:red", label="RENEW")

    ax.set_xlabel("Preference Labels")
    ax.set_ylabel("Val MSE")
    if not pretrain:
        ax.set_yscale("log")
    ax.set_title(f"{env} — seed {seed}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    naive_final = naive_r["final_mse"]
    renew_final = renew_r["final_mse"]
    gap = naive_final - renew_final
    winner = "RENEW" if gap > 0 else ("Naive" if gap < 0 else "Tie")
    ax.annotate(
        f"Naive={naive_final:.6f}  RENEW={renew_final:.6f}\n"
        f"Gap={gap:+.6f} ({winner})",
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
        OUT_DIR, f"compare_gymnax{suffix}")
    os.makedirs(base_dir, exist_ok=True)

    B = cargs.batch_size
    K = cargs.num_candidates
    labels_per_step = B * (K - 1)

    print(f"\n{'='*60}")
    print(f"  NAIVE vs RENEW COMPARISON (Gymnax)")
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
              f"{cargs.pretrain_dataset_size} transitions "
              f"(batch={cargs.pretrain_batch})")
    else:
        print(f"  Pretrain:       off")
    print(f"  Overwrite:      {cargs.overwrite}")
    print(f"  Overwrite PT:   {cargs.overwrite_pretrain}")
    print(f"  Output:         {base_dir}")
    print(f"{'='*60}")

    # Validate envs
    for env_name in cargs.envs:
        if env_name not in ENV_REGISTRY:
            raise ValueError(
                f"Unknown env '{env_name}'. "
                f"Supported: {list(ENV_REGISTRY.keys())}")

    all_results = []

    for env_name in cargs.envs:
        env_cfg = ENV_REGISTRY[env_name]()
        env, env_params = gymnax.make(env_cfg.gymnax_name)
        obs_space = env.observation_space(env_params)
        obs_low = jnp.array(obs_space.low)
        obs_high = jnp.array(obs_space.high)
        oracle_fn = make_oracle_fn(env, env_params, env_cfg)
        budget = _get_budget(cargs, env_name)
        steps = budget // labels_per_step

        model = GaussianMLPDynamics(
            env_cfg.obs_dim, env_cfg.n_actions, cargs.hidden_dim)

        for seed in cargs.seeds:
            seed_dir = os.path.join(base_dir, env_name, f"seed_{seed}")

            # --- Check if both cached ---
            naive_npz = os.path.join(seed_dir, "naive", "result.npz")
            renew_npz = os.path.join(seed_dir, "renew", "result.npz")
            both_cached = (os.path.exists(naive_npz) and
                           os.path.exists(renew_npz) and
                           not cargs.overwrite)

            # --- Pretraining ---
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
                              f"{env_name} seed {seed} ---")
                        pretrained_params = _load_pretrain_checkpoint(
                            ckpt_dir, cargs.ensemble_size)
                    else:
                        E = cargs.ensemble_size
                        opt = optax.adam(cargs.lr)
                        init_keys = jnp.stack([jax.random.PRNGKey(k)
                            for k in range(seed * 100, seed * 100 + E)])
                        dummy_s = jnp.zeros((1, env_cfg.obs_dim))
                        dummy_a = jnp.zeros((1,), jnp.int32)
                        ap = jax.vmap(
                            lambda k: model.init(k, dummy_s, dummy_a)["params"]
                        )(init_keys)
                        ao = jax.vmap(opt.init)(ap)

                        print(f"\n--- Pretraining for {env_name} seed {seed} ---")
                        dataset = collect_offline_dataset(
                            oracle_fn, obs_low, obs_high, env_cfg.n_actions,
                            cargs.pretrain_dataset_size, seed)
                        ap, ao = pretrain_ensemble(
                            model, ap, ao, dataset, cargs, seed)
                        pretrained_params = ap

                        _save_pretrain_checkpoint(
                            pretrained_params, ckpt_dir, cargs.ensemble_size)

            # --- Run both methods ---
            seed_results = []
            run_kwargs = dict(
                env_cfg=env_cfg, env=env, env_params=env_params,
                oracle_fn=oracle_fn, obs_low=obs_low, obs_high=obs_high,
            )
            for method_name, run_fn in [("naive", run_naive),
                                         ("renew", run_renew)]:
                rd = os.path.join(seed_dir, method_name)
                npz_path = os.path.join(rd, "result.npz")

                if os.path.exists(npz_path) and not cargs.overwrite:
                    print(f"\n  {method_name} {env_name} seed {seed}: "
                          f"loading {npz_path}")
                    d = dict(np.load(npz_path, allow_pickle=True))
                    result = dict(
                        env=str(d.get("env", env_name)),
                        seed=int(d.get("seed", seed)),
                        method=str(d.get("method", method_name)),
                        eval_steps=d["eval_steps"],
                        labels=d["labels"],
                        mses=d["mses"],
                        final_mse=float(d["final_mse"]),
                    )
                else:
                    pp = (jax.tree.map(lambda x: x.copy(), pretrained_params)
                          if pretrained_params is not None else None)
                    try:
                        result = run_fn(
                            cargs, env_name, seed, rd,
                            budget=budget,
                            pretrained_params=pp, **run_kwargs)
                    except Exception as e:
                        print(f"\n  ERROR: {method_name} {env_name} "
                              f"seed {seed}: {e}")
                        import traceback; traceback.print_exc()
                        continue

                seed_results.append(result)
                all_results.append(result)

            _print_seed_result(env_name, seed, seed_results)
            _save_seed_comparison_plot(seed_dir, env_name, seed,
                                       seed_results, force=remake_plots,
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
        run_final_mse=np.array([r["final_mse"] for r in all_results]),
    )

    save_comparison(all_results, cargs.envs, cargs, base_dir)
    save_results_table(all_results, cargs.envs, cargs, base_dir)
    print("\nDone!")


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))