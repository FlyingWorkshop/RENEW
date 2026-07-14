# RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences

Official code release for the RLC 2025 workshop paper
**"RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences"**
by Logan Bhamidipaty, Mykel Kochenderfer, and Subramanian Ramamoorthy.

> **TL;DR** — We supervise world-model *dynamics* directly from binary human
> preferences over imagined rollouts. **DLHF** (Dynamics Learning from Human
> Feedback) replaces the reward model in the Bradley–Terry framework with the
> trajectory log-likelihood under a learned dynamics model. **RENEW** makes this
> practical by using epistemic uncertainty to direct preference queries toward
> the transitions most vulnerable to model exploitation.

This repository contains everything needed to reproduce the figures and tables
in the paper. All experiments run on a single GPU (the paper used one NVIDIA
GeForce RTX 2080 Ti); the smaller environments also run on CPU.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The environments come from the public [Jumanji](https://github.com/instadeepai/jumanji)
(`jumanji==1.1.1`) and [Gymnax](https://github.com/RobertTLange/gymnax) suites — no
forked or modified environment code is required. For GPU acceleration, install a
CUDA-matched `jax`/`jaxlib` build instead of the default CPU wheels (see the
[JAX install guide](https://github.com/jax-ml/jax#installation)).

---

## Repository structure

```
world_models/          # DLHF + RENEW core, "from scratch" & sample-efficiency experiments, ablations
  network.py           # ConvNeXt latent dynamics model (Jumanji) + Gaussian MLP (classic control)
  env.py               # Jumanji/Gymnax wrappers, oracle solvers, preference-batch construction
  dlhf.py              # Core DLHF training loop (naive + RENEW active querying)
  heuristic_solvers.py # Heuristic action selection for rollout generation
  ...                  # experiment & plotting scripts (see mapping below)
renew/                 # Repair experiment (Section 4.3), scoped to Jumanji Maze
  pretrain.py          # Pretrain an ensemble world model on a few offline transitions
  finetune.py          # Finetune & compare RENEW vs. naive DLHF to repair exploitation
  plot_results.py      # Produce the repair figure (Figure 3)
  run_pretrain.sh      # Driver: pretrain over seeds
  run_finetune.sh      # Driver: finetune (naive + RENEW) over seeds
figures/               # Final figures as they appear in the paper
```

All scripts use [`tyro`](https://github.com/brentyi/tyro); run any of them with
`--help` to see the full set of options and defaults. By default they write
results and plots under `out/` (git-ignored).

---

## Reproducing the paper

### Figure & table → script map

| Paper artifact | Description | Script(s) |
|---|---|---|
| **Figure 1**, **Table 1** | World models learned from scratch from preferences (Jumanji) | `world_models/from_scratch.py` → `generate_from_scratch_rollouts.py` → `plot_from_scratch_rollouts.py` |
| **Figure 2**, **Table 2** | Sample efficiency: RENEW vs. naive DLHF (Jumanji) | `world_models/compare_jumanji.py` → `plot_jumanji_comparison.py` |
| **Figure 3**, **Table 3** | Repairing model exploitation (Jumanji Maze) | `renew/run_pretrain.sh` → `renew/run_finetune.sh` → `renew/plot_results.py` |
| **Figure 4**, **Table 6** | Continuous control (MountainCar, Acrobot) | `world_models/from_scratch_gymnax.py`, `compare_gymnax.py` → `plot_gymnax_comparison.py` |
| **Table 4** | Ablation over number of candidates *K* | `world_models/ablation_k.py` |
| **Table 5** | Ablation over tie handling | `world_models/ablation_ties.py` |

Run all commands from the repository root.

### Experiment 1 — Learning world models from scratch (Fig. 1, Table 1)

```bash
# Train supervised and DLHF models across the Jumanji environments and seeds
python world_models/from_scratch.py

# Generate imagined rollouts, then render Figure 1
python world_models/generate_from_scratch_rollouts.py
python world_models/plot_from_scratch_rollouts.py      # -> out/from_scratch_rollouts.png

# Continuous-control counterpart (Appendix)
python world_models/from_scratch_gymnax.py
```

### Experiment 2 — Repairing model exploitation (Fig. 3, Table 3)

```bash
# Pretrain an ensemble world model, then finetune naive vs. RENEW, over seeds 0-9.
# Override the environment/size via env vars, e.g. ENV=maze10 MAZE_SIZE=10.
ENV=maze10 MAZE_SIZE=10 bash renew/run_pretrain.sh
ENV=maze10 MAZE_SIZE=10 bash renew/run_finetune.sh

# Render the repair figure (transition error + epistemic uncertainty)
python renew/plot_results.py --env-name maze10       # -> out/maze10/plots/heatmaps_seed_*.png
```

The paper sweeps Maze sizes 5×5, 10×10, 15×15, and 20×20 (Table 3). Figure 3 shows
a single Maze 10×10 instance.

### Experiment 3 — Sample efficiency (Fig. 2, Table 2; Fig. 4, Table 6)

```bash
# Jumanji: pretrain + finetune naive vs. RENEW under a fixed label budget
python world_models/compare_jumanji.py --seeds 0 1 2 3 4
python world_models/plot_jumanji_comparison.py       # -> out/jumanji_comparison.png

# Classic control at K=2 and K=8
python world_models/compare_gymnax.py                # writes out/compare_gymnax_pretrained_k2, _k8
python world_models/plot_gymnax_comparison.py        # -> out/gymnax_comparison.png
```

### Ablations (Tables 4 & 5)

```bash
python world_models/ablation_k.py       # sweep K in {2, 3, 4, 6, 8}
python world_models/ablation_ties.py    # exclude vs. include tied preference pairs
```

---

## Method at a glance

- **DLHF loss.** For a dynamics model with trajectory log-likelihood
  `ℓ_θ(σ) = Σ_t log T_θ(s_{t+1} | s_t, a_t)`, the Bradley–Terry preference model
  becomes `P(σ⁰ ≻ σ¹) = logistic(ℓ_θ(σ⁰) − ℓ_θ(σ¹))`, trained with the standard
  preference cross-entropy. The dynamics model *is* the preference function — a
  trajectory is preferred if it is more likely under the learned dynamics.
- **RENEW.** Starting from a pretrained model and preferences only (no access to
  the pretraining data), each round (1) estimates epistemic uncertainty (ensemble
  disagreement by default), (2) samples start states proportional to uncertainty,
  (3) elicits `N/I` preferences, and (4) finetunes on the DLHF loss, then repeats.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bhamidipaty2026renew,
  title     = {{RENEW}: Towards Learning World Models and Repairing Model Exploitation from Preferences},
  author    = {Bhamidipaty, Logan and Kochenderfer, Mykel and Ramamoorthy, Subramanian},
  booktitle = {Finding the Frame Workshop at the Reinforcement Learning Conference (RLC)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
