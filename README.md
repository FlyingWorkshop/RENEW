# RENEW: Towards Learning World Models and Repairing Model Exploitation from Preferences

Code for **"RENEW: Towards Learning World Models and Repairing Model Exploitation
from Preferences"** (RLC 2026, Finding the Frame Workshop) by Logan Bhamidipaty,
Mykel Kochenderfer, and Subramanian Ramamoorthy.

**DLHF** (Dynamics Learning from Human Feedback) replaces the reward model in the
Bradley–Terry preference framework with the trajectory log-likelihood under a
learned dynamics model, so binary preferences supervise world-model *dynamics*
directly. **RENEW** makes this practical by directing preference queries toward
high-uncertainty transitions — the ones most vulnerable to model exploitation.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11. Environments come from the public
[Jumanji](https://github.com/instadeepai/jumanji) and
[Gymnax](https://github.com/RobertTLange/gymnax) suites. The default install is
CPU-only; for GPU, install a CUDA-matched `jax`/`jaxlib` build (see the
[JAX install guide](https://github.com/jax-ml/jax#installation)). All experiments
ran on a single NVIDIA GeForce RTX 2080 Ti.

## Structure

```
world_models/   # DLHF + RENEW core (network.py, env.py, dlhf.py) and experiment scripts
renew/          # Repair experiment on Jumanji Maze (pretrain → finetune → plot)
figures/        # Figures exactly as they appear in the paper
```

Every script is a `tyro` CLI — run with `--help` for all options. Script defaults
match the paper's configuration. Results and plots are written to `out/`
(git-ignored). Run everything from the repository root.

## Reproducing the paper

**Learning world models from scratch** (Figure 2, Table 1):

```bash
python world_models/from_scratch.py                     # Table 1 (supervised vs. DLHF)
python world_models/generate_from_scratch_rollouts.py   # trains at 1K/5K/1M labels
python world_models/plot_from_scratch_rollouts.py       # -> out/from_scratch_rollouts.png
```

**Sample efficiency, naive DLHF vs. RENEW** (Figure 3, Table 2):

```bash
python world_models/compare_jumanji.py
python world_models/plot_jumanji_comparison.py           # -> out/jumanji_comparison.png
```

**Repairing model exploitation** (Figure 4, Table 3):

```bash
ENV=maze10 MAZE_SIZE=10 bash renew/run_pretrain.sh       # pretrain ensembles, seeds 0-9
ENV=maze10 MAZE_SIZE=10 bash renew/run_finetune.sh       # finetune naive vs. RENEW
python renew/plot_results.py --env-name maze10 --maze-size 10
```

Table 3 sweeps `maze5`/`maze10`/`maze15`/`maze20` (set `ENV` and `MAZE_SIZE`
accordingly). Figure 4 is `out/maze10/plots/heatmaps_seed_3.png`.

**Continuous control** (Figure 5, Table 7):

```bash
python world_models/from_scratch_gymnax.py               # from-scratch comparison
python world_models/compare_gymnax.py --num-candidates 2 --results-dir out/compare_gymnax_pretrained_k2
python world_models/compare_gymnax.py --num-candidates 8 --results-dir out/compare_gymnax_pretrained_k8
python world_models/plot_gymnax_comparison.py            # -> out/gymnax_comparison.png
```

**Ablations** (Tables 5 & 6):

```bash
python world_models/ablation_k.py       # candidates K in {2, 3, 4, 6, 8}
python world_models/ablation_ties.py    # exclude vs. include tied pairs
```

## Citation

```bibtex
@inproceedings{bhamidipaty2026renew,
  title     = {{RENEW}: Towards Learning World Models and Repairing Model Exploitation from Preferences},
  author    = {Bhamidipaty, Logan and Kochenderfer, Mykel and Ramamoorthy, Subramanian},
  booktitle = {Finding the Frame Workshop at the Reinforcement Learning Conference (RLC)},
  year      = {2026}
}
```

## License

[MIT](LICENSE)
