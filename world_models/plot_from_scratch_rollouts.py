"""
plot_from_scratch_rollouts.py
=============================
Paper-quality rollout figure for Section 4.1.

Layout:
    Columns : Sokoban  |  Maze  |  Sliding
    Row 0   : True (ground truth)
    Row 1   : Model@1K
    Row 2   : Model@5K
    Row 3   : Model@1M

Each cell: 5 frames (t=0..4), action glyphs in the header.

Workflow:
  1. Run generate_from_scratch_rollouts.py --rollout-seeds 0 1 2
  2. Browse previews/ GIFs to pick best seeds per env
  3. Run this: --rollout-seeds 2 0 1
     (sokoban uses rseed=2, maze uses rseed=0, sliding uses rseed=1)

Usage:
    python plot_from_scratch_rollouts.py
    python plot_from_scratch_rollouts.py --rollout-seeds 2 0 1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image as PILImage, ImageDraw, ImageFont

from env import EnvMeta
from dlhf import make_meta, DLHFArgs


# =====================================================================
# Config
# =====================================================================

@dataclass
class Args:
    data: str = "out/from_scratch_rollouts/rollout_data.npz"
    output: str = "out/from_scratch_rollouts.png"

    envs:    List[str] = field(default_factory=lambda: [
        "sokoban", "maze10", "sliding5",
    ])
    budgets: List[int] = field(default_factory=lambda: [
        1_000, 5_000, 1_000_000,
    ])

    rollout_seeds: Optional[List[int]] = field(default_factory=lambda: [18, 5, 10])
    """Per-env rollout seed selection (positional, matching --envs order).
    E.g. --rollout-seeds 18 5 10 means sokoban=rs18, maze=rs5, sliding=rs10.
    Defaults to first available seed for each env if None."""

    render_dpi: int = 150
    frame_cache_dir: str = "out/from_scratch_rollouts/frame_cache"

    frame_px: int = 160
    col_sep_px:  int = 40
    row_sep_px:  int = 16
    left_margin: int = 40
    top_margin:  int = 90

    # Architecture (for meta reconstruction)
    embed_dim:       int = 128
    latent_channels: int = 64
    enc_layers:      int = 3
    dyn_layers:      int = 4
    dec_layers:      int = 3
    scramble_steps:  int = 0
    context_len:     int = 10


# =====================================================================
# Constants
# =====================================================================

ENV_NAMES: Dict[str, str] = {
    "sokoban":  "Sokoban",
    "maze5":    "Maze 5\u00d75",
    "maze10":   "Maze 10\u00d710",
    "maze15":   "Maze 15\u00d715",
    "maze20":   "Maze 20\u00d720",
    "sliding3": "Sliding 3\u00d73",
    "sliding5": "Sliding 5\u00d75",
}

ACTION_GLYPHS = {0: "\u2191", 1: "\u2192", 2: "\u2193", 3: "\u2190"}

GT_COLOR     = (46, 125, 50)      # green for True label
MODEL_COLOR  = (120, 81, 169)     # dark pastel purple for Model labels
ACTION_COLOR = (100, 100, 100)
TITLE_COLOR  = (50, 50, 50)
BG_COLOR     = (255, 255, 255)
BORDER_COLOR = (210, 210, 210)    # neutral frame border (GT rows)
CORRECT_CLR  = (46, 125, 50)     # green border — prediction matches GT
WRONG_CLR    = (196, 78, 82)     # red border — prediction differs from GT


def _budget_label(b: int) -> str:
    # Short tags (the caption clarifies these are preference-label budgets).
    if b >= 1_000_000:
        return f"{b // 1_000_000}M"
    if b >= 1_000:
        return f"{b // 1_000}K"
    return str(b)


def _get_font_bold(size: int):
    """Bold serif — Times-like, matching the paper body font (rlj.sty `times`)."""
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
        "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _get_font(size: int):
    """Regular serif — Times-like, matching the paper body font."""
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _get_arrow_font(size: int):
    """Action glyphs (U+2190..2193) — use a font with reliable arrow coverage."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# =====================================================================
# Env helpers
# =====================================================================

def _make_dlhf_args_for_meta(args: Args, env: str) -> DLHFArgs:
    return DLHFArgs(
        seed=0, env=env,
        embed_dim=args.embed_dim,
        latent_channels=args.latent_channels,
        enc_layers=args.enc_layers,
        dyn_layers=args.dyn_layers,
        dec_layers=args.dec_layers,
        lr=3e-4, steps=100, batch_size=64,
        horizon=1, beta_bt=1.0,
        scramble_steps=args.scramble_steps,
        context_len=args.context_len, ensemble_size=1,
        renew=False, val_size=100, val_scramble=30,
        eval_every=100, results_dir="/tmp", rollout_steps=5,
    )


def render_obs_to_file(env, meta: EnvMeta, obs_flat: np.ndarray,
                       template_state, path: str, dpi: int):
    state = meta.obs_to_state(jnp.array(obs_flat), template_state)
    env.render(state)
    fig = plt.gcf()
    fig.savefig(path, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    plt.close("all")


def get_template_state(env, meta, seed=0):
    state, _ = env.reset(jax.random.PRNGKey(seed))
    return state


# =====================================================================
# Data loading
# =====================================================================

def load_data(path: str):
    raw = np.load(path, allow_pickle=True)
    meta = {
        "envs":          list(raw["envs"]),
        "budgets":       list(raw["budgets"].astype(int)),
        "seed":          int(raw["seed"]),
        "rollout_seeds": list(raw["rollout_seeds"].astype(int)),
        "horizon":       int(raw["horizon"]),
    }

    gt_obs  = {}   # gt_obs[(env, rseed)] = (H+1, obs_dim)
    mod_obs = {}   # mod_obs[(env, budget, rseed)] = (H+1, obs_dim)
    actions = {}   # actions[(env, rseed)] = (H,)

    for env in meta["envs"]:
        for rseed in meta["rollout_seeds"]:
            rs_tag = f"rs{rseed}"
            ak = f"actions_{env}_{rs_tag}"
            gk = f"gt_obs_{env}_{rs_tag}"
            if ak in raw:
                actions[(env, rseed)] = raw[ak].astype(int)
            if gk in raw:
                gt_obs[(env, rseed)] = raw[gk]

            for budget in meta["budgets"]:
                mk = f"model_obs_{env}_{budget}_{rs_tag}"
                if mk in raw:
                    mod_obs[(env, budget, rseed)] = raw[mk]

    return meta, gt_obs, mod_obs, actions


# =====================================================================
# Rendering
# =====================================================================

def render_all_frames(
    data_meta, gt_obs, mod_obs, envs, budgets,
    seed_per_env, envs_and_metas, args,
) -> Dict[Tuple, PILImage.Image]:
    """Render frames, caching to disk. Returns (env, row_key, t) -> PIL."""
    T = data_meta["horizon"] + 1
    frames = {}
    cache_dir = args.frame_cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    rendered = 0
    cached = 0

    for env_str in envs:
        env_obj, meta, template = envs_and_metas[env_str]
        rseed = seed_per_env[env_str]

        for t in range(T):
            # GT frame
            fname = f"{env_str}_rs{rseed}_gt_t{t}.png"
            path = os.path.join(cache_dir, fname)
            if not os.path.exists(path):
                render_obs_to_file(
                    env_obj, meta, gt_obs[(env_str, rseed)][t],
                    template, path, args.render_dpi)
                rendered += 1
            else:
                cached += 1
            img = PILImage.open(path).convert("RGB")
            img = img.resize((args.frame_px, args.frame_px),
                             PILImage.LANCZOS)
            frames[(env_str, "gt", t)] = img

            # Model frames per budget
            for budget in budgets:
                mk = (env_str, budget, rseed)
                if mk not in mod_obs:
                    continue
                fname = f"{env_str}_rs{rseed}_{budget}_t{t}.png"
                path = os.path.join(cache_dir, fname)
                if not os.path.exists(path):
                    render_obs_to_file(
                        env_obj, meta, mod_obs[mk][t],
                        template, path, args.render_dpi)
                    rendered += 1
                else:
                    cached += 1
                img = PILImage.open(path).convert("RGB")
                img = img.resize((args.frame_px, args.frame_px),
                                 PILImage.LANCZOS)
                frames[(env_str, budget, t)] = img

    print(f"  Rendered {rendered} new, loaded {cached} from cache")
    return frames


# =====================================================================
# Composite
# =====================================================================

def composite_figure(
    frames, data_meta, actions, gt_obs, mod_obs,
    envs, budgets, seed_per_env, args,
) -> PILImage.Image:
    T       = data_meta["horizon"] + 1
    n_cols  = len(envs)
    n_rows  = 1 + len(budgets)

    fp = args.frame_px
    frame_gap = 4
    col_w = T * fp + (T - 1) * frame_gap
    col_sep = args.col_sep_px
    row_sep = args.row_sep_px

    # ── Sizing ────────────────────────────────────────────────────────
    # Author at the paper's print width (\textwidth) so text drawn in pixels
    # lands at real point sizes; main() embeds a matching DPI so the figure
    # drops in at the right size with a bare \includegraphics{...}.
    TARGET_W_IN = 5.5          # \textwidth in rlj.sty; paper uses width=\textwidth
    PT_PER_IN   = 72.27
    frames_w = n_cols * col_w + (n_cols - 1) * col_sep
    # px-per-point at the target print width (the frames dominate the width).
    px_per_pt = (frames_w + 120) / (TARGET_W_IN * PT_PER_IN)

    def _pt(p: float) -> int:
        return max(1, int(round(p * px_per_pt)))

    title_px, header_px, row_px, arrow_px = _pt(9.0), _pt(6.5), _pt(8.5), _pt(9.0)
    font_title  = _get_font(title_px)       # column env names
    font_header = _get_font(header_px)       # timestep indices
    font_arrow  = _get_arrow_font(arrow_px)  # action glyphs
    font_row    = _get_font_bold(row_px)     # row labels
    border_w    = max(3, _pt(1.3))

    left_margin = row_px + _pt(6)                 # room for rotated row labels
    top_margin  = title_px + header_px + _pt(10)  # title + timestep row + gaps
    arrow_space = _pt(12)

    canvas_w = left_margin + frames_w + _pt(6)
    canvas_h = top_margin + n_rows * fp + (n_rows - 1) * row_sep + arrow_space

    canvas = PILImage.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    row_labels = ["True"] + [_budget_label(b) for b in budgets]
    row_colors = [GT_COLOR] + [MODEL_COLOR] * len(budgets)
    row_keys   = ["gt"] + list(budgets)

    # ── Paste frames + colored borders ───────────────────────────
    for ri in range(n_rows):
        for ci in range(n_cols):
            env_str = envs[ci]
            rkey    = row_keys[ri]
            rseed   = seed_per_env[env_str]

            for t in range(T):
                x = left_margin + ci * (col_w + col_sep) + t * (fp + frame_gap)
                y = top_margin + ri * (fp + row_sep)

                fkey = (env_str, rkey, t)
                if fkey in frames:
                    canvas.paste(frames[fkey], (x, y))

                # Border color: neutral for GT, green/red for model
                if rkey == "gt":
                    border_clr = (60, 60, 60)
                else:
                    gt_key  = (env_str, rseed)
                    mod_key = (env_str, rkey, rseed)
                    if gt_key in gt_obs and mod_key in mod_obs:
                        match = np.array_equal(gt_obs[gt_key][t], mod_obs[mod_key][t])
                        border_clr = CORRECT_CLR if match else WRONG_CLR
                    else:
                        border_clr = BORDER_COLOR

                draw.rectangle(
                    [x, y, x + fp - 1, y + fp - 1],
                    outline=border_clr, width=border_w)

    # ── Column titles ────────────────────────────────────────────
    title_cy = _pt(3) + title_px // 2
    for ci, env_str in enumerate(envs):
        title = ENV_NAMES.get(env_str, env_str)
        cx = left_margin + ci * (col_w + col_sep) + col_w // 2
        draw.text((cx, title_cy), title, fill=TITLE_COLOR,
                  font=font_title, anchor="mm")

    # ── Timestep numbers above frames ────────────────────────────
    ts_cy = top_margin - header_px // 2 - _pt(2)
    for ci in range(n_cols):
        for t in range(T):
            cx = (left_margin + ci * (col_w + col_sep)
                  + t * (fp + frame_gap) + fp // 2)
            draw.text((cx, ts_cy), str(t), fill=ACTION_COLOR,
                      font=font_header, anchor="mm")

    # ── Action arrows below frames (under last row) ──────────────
    last_row_y = top_margin + (n_rows - 1) * (fp + row_sep) + fp
    arrow_cy = last_row_y + arrow_space // 2
    for ci, env_str in enumerate(envs):
        rseed = seed_per_env[env_str]
        acts = actions.get((env_str, rseed), [])
        for t in range(T):
            cx = (left_margin + ci * (col_w + col_sep)
                  + t * (fp + frame_gap) + fp // 2)
            if t < len(acts):
                glyph = ACTION_GLYPHS.get(int(acts[t]), "")
                draw.text((cx, arrow_cy), glyph, fill=ACTION_COLOR,
                          font=font_arrow, anchor="mm")

    # ── Row labels (rotated 90°) ─────────────────────────────────
    lbl_w, lbl_h = max(300, row_px * 8), int(row_px * 1.8)
    for ri in range(n_rows):
        y_center = top_margin + ri * (fp + row_sep) + fp // 2

        txt_img = PILImage.new("RGBA", (lbl_w, lbl_h), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((lbl_w // 2, lbl_h // 2), row_labels[ri],
                      fill=row_colors[ri], font=font_row, anchor="mm")

        bbox = txt_img.getbbox()
        if bbox:
            txt_img = txt_img.crop(bbox)

        txt_rot = txt_img.rotate(90, expand=True)

        paste_x = left_margin - _pt(4) - txt_rot.width
        paste_y = y_center - txt_rot.height // 2
        canvas.paste(txt_rot, (paste_x, paste_y), txt_rot)

    return canvas


# =====================================================================
# Main
# =====================================================================

def main(args: Args) -> None:
    if not os.path.exists(args.data):
        print(f"  ERROR: {args.data} not found.")
        print(f"  Run generate_from_scratch_rollouts.py first.")
        return

    data_meta, gt_obs, mod_obs, actions = load_data(args.data)
    envs    = args.envs
    budgets = args.budgets

    # Resolve per-env rollout seeds
    if args.rollout_seeds is not None:
        assert len(args.rollout_seeds) == len(envs), \
            (f"--rollout-seeds length ({len(args.rollout_seeds)}) "
             f"must match --envs ({len(envs)})")
        seed_per_env = {e: s for e, s in zip(envs, args.rollout_seeds)}
    else:
        # Default to first available seed
        default_rs = data_meta["rollout_seeds"][0]
        seed_per_env = {e: default_rs for e in envs}

    print(f"  Envs:    {envs}")
    print(f"  Budgets: {budgets}")
    print(f"  Seeds:   {seed_per_env}")

    # Build envs for rendering
    print("  Initialising environments...")
    envs_and_metas = {}
    for env_str in envs:
        dargs = _make_dlhf_args_for_meta(args, env_str)
        meta  = make_meta(dargs)
        env   = meta.make_env()
        template = get_template_state(env, meta, seed=data_meta["seed"])
        envs_and_metas[env_str] = (env, meta, template)

    # Render
    frames = render_all_frames(
        data_meta, gt_obs, mod_obs, envs, budgets,
        seed_per_env, envs_and_metas, args)

    # Composite
    print("  Compositing figure...")
    canvas = composite_figure(
        frames, data_meta, actions, gt_obs, mod_obs,
        envs, budgets, seed_per_env, args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Embed a DPI so the natural size equals the paper's print width (5.5in),
    # letting \includegraphics{...} render it correctly with no width= option.
    dpi_val = canvas.width / 5.5
    canvas.save(args.output, dpi=(dpi_val, dpi_val))
    print(f"\n  Saved -> {args.output}  "
          f"({os.path.getsize(args.output) / 1024:.0f} KB), "
          f"natural width {canvas.width / dpi_val:.2f} in")


if __name__ == "__main__":
    main(tyro.cli(Args))