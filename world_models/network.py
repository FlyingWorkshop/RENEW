"""
world_models/network.py
=======================
LatentCNN world model: obs -> z -> z' -> next_obs.
Includes offline (reconstruction), DLHF (Bradley-Terry preference),
and RENEW K-candidate preference losses.
"""

from __future__ import annotations

import os
import pickle
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax


# =============================================================================
# Building blocks
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
        act_emb = nn.Dense(self.latent_channels)(
            jax.nn.one_hot(action, self.num_actions))
        act_map = jnp.broadcast_to(act_emb[:, None, None, :], z.shape)
        x = nn.Dense(self.latent_channels)(jnp.concatenate([z, act_map], axis=-1))
        for _ in range(self.dyn_layers):
            x = ConvNextBlock(self.latent_channels)(x)
        return x


class _LatentDecoder(nn.Module):
    latent_channels: int; dec_layers: int; num_tiles: int; num_actions: int

    @nn.compact
    def __call__(self, z, action):
        act_emb = nn.Dense(self.latent_channels)(
            jax.nn.one_hot(action, self.num_actions))
        act_map = jnp.broadcast_to(act_emb[:, None, None, :], z.shape)
        x = nn.Dense(self.latent_channels)(jnp.concatenate([z, act_map], axis=-1))
        for _ in range(self.dec_layers):
            x = ConvNextBlock(self.latent_channels)(x)
        return nn.Dense(self.num_tiles)(x)


# =============================================================================
# LatentCNN
# =============================================================================

class LatentCNNNetwork(nn.Module):
    embed_dim:       int
    latent_channels: int
    enc_layers:      int
    dyn_layers:      int
    dec_layers:      int
    num_tiles:       int
    num_actions:     int
    grid_shape:      Tuple[int, int]
    beta_bt:          float = 1.0
    finetune_horizon: int   = 1
    num_candidates:   int   = 8     # K for RENEW K-candidate selection
    exclude_ties:     bool  = True  # Whether to exclude ties from BT loss

    def setup(self):
        self.encoder = _LatentEncoder(
            self.embed_dim, self.latent_channels, self.enc_layers,
            self.num_tiles, self.grid_shape)
        self.dynamics = _LatentDynamics(
            self.latent_channels, self.dyn_layers, self.num_actions)
        self.decoder = _LatentDecoder(
            self.latent_channels, self.dec_layers, self.num_tiles, self.num_actions)

    def __call__(self, obs, action):
        z      = self.encoder(obs)
        z_next = self.dynamics(z, action)
        return self.decoder(z_next, action)

    def predict(self, obs, action):
        return jnp.argmax(self(obs, action), axis=-1).reshape(obs.shape[0], -1)

    def encode(self, obs):
        """Obs -> latent z."""
        return self.encoder(obs)

    def step_latent(self, z, action):
        """Single dynamics step in latent space: z -> z'."""
        return self.dynamics(z, action)

    def decode_latent(self, z, action):
        """Latent z -> obs logits -> argmax obs."""
        logits = self.decoder(z, action)
        B = z.shape[0]
        return jnp.argmax(logits, axis=-1).reshape(B, -1)

    def decode(self, obs, action):
        return self(obs, action).reshape(obs.shape[0], -1, self.num_tiles)

    def val_l1(self, obs, action, gt_next):
        pred = self.predict(obs, action)
        return jnp.abs(pred - gt_next).astype(jnp.float32).mean()

    def offline_loss(self, context_boards, context_actions, key):
        obs     = context_boards[:, :-1, :]
        actions = context_actions[:, :-1]
        targets = context_boards[:, 1:, :]
        B, T    = obs.shape[0], obs.shape[1]
        logits = self.decode(obs.reshape(B * T, -1), actions.reshape(B * T))
        lp = jax.nn.log_softmax(logits, axis=-1)
        oh = jax.nn.one_hot(targets.reshape(B * T, -1), self.num_tiles)
        l_recon = -(lp * oh).sum(-1).mean()
        return l_recon, dict(l_recon=l_recon)

    # -----------------------------------------------------------------
    # Original two-trajectory preference loss
    # -----------------------------------------------------------------
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

        ties = (err1 == err2).astype(jnp.float32)
        pred_logits = self.beta_bt * (total_lp2 - total_lp1)

        if self.exclude_ties:
            # Original behaviour: mask out ties entirely
            preferences = (err2 < err1).astype(jnp.float32)
            bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
            l_bt = jnp.mean(bce * (1.0 - ties))
        else:
            # Include ties with preference = 0.5 (no preference)
            preferences = jnp.where(
                ties > 0.5,
                0.5,
                (err2 < err1).astype(jnp.float32),
            )
            bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
            l_bt = jnp.mean(bce)

        total = self.beta_bt * l_bt

        no_tie = 1.0 - ties
        pref_acc = jnp.where(
            no_tie.mean() > 0,
            jnp.sum(((pred_logits > 0) == (preferences > 0.5)).astype(jnp.float32)
                     * no_tie) / (jnp.sum(no_tie) + 1e-8),
            0.5)
        return total, dict(bt_loss=l_bt, tie_rate=ties.mean(), pref_acc=pref_acc)

    # -----------------------------------------------------------------
    # RENEW: K-candidate preference loss
    # -----------------------------------------------------------------
    def k_candidate_preferences_loss(self, start_obs, actions, gt, key):
        """
        K-candidate preference loss for RENEW active learning.

        For each (s0, action_seq) in the batch, generates K rollout samples
        from the model by sampling from per-step categorical logits. The
        candidate closest to the oracle ground truth (by L1) is treated as
        the "winner"; we form K-1 Bradley-Terry pairs (winner, loser_k)
        and push the model to assign higher likelihood to the winner.

        This yields K-1 gradient signals per oracle query, significantly
        improving sample efficiency over the standard 2-trajectory approach.

        Args:
            start_obs:  (B, obs_dim)       starting observations
            actions:    (B, H)             action sequences
            gt:         (B, H, obs_dim)    oracle ground truth trajectories
            key:        PRNG key

        Returns:
            (total_loss, info_dict)
        """
        K = self.num_candidates
        H = self.finetune_horizon
        B = start_obs.shape[0]

        # --- Generate K candidate trajectories via vmap ---------------
        def _single_rollout(rng):
            """Sample one trajectory for the full batch."""
            obs = start_obs
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

        rngs = jax.random.split(key, K)
        # candidates: (K, B, H, obs_dim)   log_probs: (K, B, H)
        candidates, log_probs = jax.vmap(_single_rollout)(rngs)

        # --- Score each candidate by L1 error to GT ------------------
        # gt is (B, H, obs_dim), broadcast over K
        errors = jnp.abs(candidates - gt[None]).astype(jnp.float32)
        errors_per_cand = errors.sum(axis=(-1, -2))        # (K, B)

        # --- Identify winner (lowest error) per example ---------------
        best_idx = jnp.argmin(errors_per_cand, axis=0)     # (B,)
        b_arange = jnp.arange(B)

        best_total_lp  = log_probs[best_idx, b_arange].sum(-1)   # (B,)
        best_errors    = errors_per_cand[best_idx, b_arange]      # (B,)

        # --- Form K-1 pairs: (winner, candidate_k) -------------------
        all_total_lp = log_probs.sum(-1)                    # (K, B)

        # BT logit > 0  =>  model assigns higher prob to winner
        pred_logits = self.beta_bt * (
            best_total_lp[None, :] - all_total_lp           # (K, B)
        )

        # Self-pair mask: always exclude winner vs itself
        k_idx = jnp.arange(K)[:, None]                     # (K, 1)
        self_mask = (k_idx == best_idx[None, :]).astype(jnp.float32)  # (K, B)

        # Ties: candidates with same error as the winner
        ties = (errors_per_cand == best_errors[None, :]).astype(jnp.float32)  # (K, B)

        if self.exclude_ties:
            # Original behaviour: mask out all ties (includes self-pair)
            valid = 1.0 - ties
            preferences = jnp.ones((K, B))
            bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
            n_valid = jnp.sum(valid) + 1e-8
            l_bt = jnp.sum(bce * valid) / n_valid
        else:
            # Include ties with preference = 0.5; still exclude self-pair
            valid = 1.0 - self_mask
            # For tied non-self pairs, set preference to 0.5
            tied_non_self = ties * (1.0 - self_mask)
            preferences = jnp.where(tied_non_self > 0.5, 0.5, jnp.ones((K, B)))
            bce = optax.sigmoid_binary_cross_entropy(pred_logits, preferences)
            n_valid = jnp.sum(valid) + 1e-8
            l_bt = jnp.sum(bce * valid) / n_valid

        total = self.beta_bt * l_bt

        # --- Diagnostics ----------------------------------------------
        # pref_acc computed only on non-tied, non-self pairs for consistency
        non_tie_valid = (1.0 - ties) * (1.0 - self_mask)
        n_non_tie = jnp.sum(non_tie_valid) + 1e-8
        pref_acc = jnp.sum(
            ((pred_logits > 0) == (jnp.ones((K, B)) > 0.5)).astype(jnp.float32)
            * non_tie_valid
        ) / n_non_tie

        n_pairs = jnp.sum(valid)     # effective number of training pairs

        return total, dict(
            bt_loss=l_bt,
            tie_rate=ties.mean(),
            pref_acc=pref_acc,
            mean_error=errors_per_cand.mean(),
            best_error=best_errors.mean(),
            n_pairs=n_pairs,
        )

    # -----------------------------------------------------------------
    # Ensemble scoring: logits for active selection (called externally)
    # -----------------------------------------------------------------
    def rollout_logits(self, obs, actions):
        """
        Deterministic (argmax) multi-step rollout returning per-step logits.
        Used externally for ensemble disagreement scoring.

        Args:
            obs:      (B, obs_dim)
            actions:  (B, H)

        Returns:
            logits:   (B, H, obs_dim, num_tiles)
        """
        H = actions.shape[1]
        all_logits = []
        for t in range(H):
            logits = self.decode(obs, actions[:, t])       # (B, obs_dim, tiles)
            pred   = jnp.argmax(logits, axis=-1).reshape(obs.shape[0], -1)
            all_logits.append(logits)
            obs = pred   # feed argmax prediction forward
        return jnp.stack(all_logits, axis=1)


# =============================================================================
# Construction helpers
# =============================================================================

def build_network(meta, args) -> LatentCNNNetwork:
    return LatentCNNNetwork(
        embed_dim=args.embed_dim,
        latent_channels=args.latent_channels,
        enc_layers=args.enc_layers,
        dyn_layers=args.dyn_layers,
        dec_layers=args.dec_layers,
        num_tiles=meta.num_tiles,
        num_actions=meta.num_actions,
        grid_shape=meta.grid_shape,
        beta_bt=getattr(args, "beta_bt", 1.0),
        finetune_horizon=getattr(args, "horizon", 1),
        num_candidates=getattr(args, "num_candidates", 8),
        exclude_ties=getattr(args, "exclude_ties", True),
    )


def init_params(network, meta, batch_size: int, context_len: int, rng):
    od = meta.obs_dim
    return network.init(
        rng,
        jnp.zeros((batch_size, context_len, od), jnp.int32),
        jnp.zeros((batch_size, context_len), jnp.int32),
        rng,
        method=LatentCNNNetwork.offline_loss,
    )


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_params(params, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.tree.map(np.array, params), f)
    print(f"Saved → {path}")


def load_params(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)