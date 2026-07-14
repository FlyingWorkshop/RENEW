"""
world_models/env.py
===================
Environment metadata, oracle dynamics, data collection utilities,
and active learning pool generation for RENEW.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Env metadata container
# =============================================================================

class EnvMeta(NamedTuple):
    obs_dim:      int
    num_tiles:    int
    num_actions:  int
    grid_shape:   Tuple[int, int]
    make_env:     Callable
    extract_obs:  Callable   # (batched_states) -> (B, obs_dim) int32
    oracle_step:  Optional[Callable]   # (key, obs_flat, action) -> obs_flat  (None = use env.step)
    obs_to_state: Callable   # (obs_flat, real_state) -> patched state


# =============================================================================
# Maze environment
# =============================================================================

def make_maze_meta(maze_size: int = 10) -> EnvMeta:
    from jumanji.environments.routing.maze.env import Maze
    from jumanji.environments.routing.maze.generator import RandomGenerator
    from jumanji.environments.routing.maze.types import Position as MazePos

    ROWS, COLS = maze_size, maze_size
    OBS_DIM    = ROWS * COLS
    DR = jnp.array([-1, 0, 1, 0])
    DC = jnp.array([0, 1, 0, -1])

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
        dr, dc    = DR[action], DC[action]
        new_idx   = jnp.clip(row + dr, 0, ROWS - 1) * COLS + jnp.clip(col + dc, 0, COLS - 1)
        final_idx = jnp.where(obs[new_idx] == 0, agent_idx, new_idx)
        new_obs   = obs.at[agent_idx].set(jnp.where(agent_idx == tgt_idx, 3, 1))
        new_obs   = new_obs.at[final_idx].set(jnp.where(final_idx == tgt_idx, 3, 2))
        return new_obs

    def obs_to_state(obs_flat, real_state):
        grid = obs_flat.reshape(ROWS, COLS)
        walls = (grid == 0)  # 0=wall in our encoding
        agent_idx = jnp.argmax(obs_flat == 2)
        tgt_idx   = jnp.argmax(obs_flat == 3)
        ar, ac    = jnp.divmod(agent_idx, COLS)
        tr, tc    = jnp.divmod(tgt_idx,   COLS)
        return real_state.replace(
            walls=walls,
            agent_position  = MazePos(row=ar, col=ac),
            target_position = MazePos(row=tr, col=tc))

    return EnvMeta(
        obs_dim=OBS_DIM, num_tiles=4, num_actions=4,
        grid_shape=(ROWS, COLS),
        make_env=lambda: Maze(generator=RandomGenerator(
            num_rows=ROWS, num_cols=COLS)),
        extract_obs=extract, oracle_step=oracle,
        obs_to_state=obs_to_state,
    )


# =============================================================================
# Sliding Tile Puzzle environment
# =============================================================================

def make_sliding_tile_meta(grid_size: int = 3, num_random_moves: int = 200) -> EnvMeta:
    from jumanji.environments.logic.sliding_tile_puzzle.env import SlidingTilePuzzle
    from jumanji.environments.logic.sliding_tile_puzzle.generator import RandomWalkGenerator
    from jumanji.environments.logic.sliding_tile_puzzle.types import State as STPState

    N       = grid_size
    OBS_DIM = N * N
    DR = jnp.array([-1, 0, 1, 0])
    DC = jnp.array([0, 1, 0, -1])

    def extract(states):
        B = states.puzzle.shape[0]
        return states.puzzle.reshape(B, OBS_DIM)

    def oracle(key, obs, action):
        grid = obs.reshape(N, N)
        empty_idx = jnp.argmin(obs)
        er, ec = jnp.divmod(empty_idx, N)
        nr = jnp.clip(er + DR[action], 0, N - 1)
        nc = jnp.clip(ec + DC[action], 0, N - 1)
        moved = (nr != er) | (nc != ec)
        tile_val = grid[nr, nc]
        new_grid = grid.at[er, ec].set(jnp.where(moved, tile_val, 0))
        new_grid = new_grid.at[nr, nc].set(jnp.where(moved, 0, grid[nr, nc]))
        return new_grid.reshape(OBS_DIM)

    def obs_to_state(obs_flat, real_state):
        puzzle = obs_flat.reshape(N, N)
        empty_idx = jnp.argmin(obs_flat)
        er, ec = jnp.divmod(empty_idx, N)
        return real_state.replace(
            puzzle=puzzle,
            empty_tile_position=jnp.array([er, ec], dtype=jnp.int32),
        )

    return EnvMeta(
        obs_dim=OBS_DIM,
        num_tiles=N * N,
        num_actions=4,
        grid_shape=(N, N),
        make_env=lambda: SlidingTilePuzzle(
            generator=RandomWalkGenerator(grid_size=N, num_random_moves=num_random_moves)),
        extract_obs=extract,
        oracle_step=oracle,
        obs_to_state=obs_to_state,
    )


# =============================================================================
# Sokoban environment
# =============================================================================

def make_sokoban_meta() -> EnvMeta:
    from jumanji.environments.routing.sokoban.env import Sokoban
    from jumanji.environments.routing.sokoban.generator import (
        HuggingFaceDeepMindGenerator,
    )
    from jumanji.environments.routing.sokoban.types import State as SokobanState
    from jumanji.environments.routing.sokoban.constants import (
        AGENT, BOX, EMPTY, GRID_SIZE, TARGET, TARGET_AGENT, TARGET_BOX, WALL,
    )

    N       = GRID_SIZE
    OBS_DIM = N * N
    NUM_TILES = int(max(EMPTY, WALL, AGENT, BOX, TARGET, TARGET_AGENT, TARGET_BOX)) + 1
    DR = jnp.array([-1, 0, 1, 0])
    DC = jnp.array([0, 1, 0, -1])

    _EMPTY = int(EMPTY)
    _WALL  = int(WALL)
    _AGENT = int(AGENT)
    _BOX   = int(BOX)
    _TARGET       = int(TARGET)
    _TARGET_AGENT = int(TARGET_AGENT)
    _TARGET_BOX   = int(TARGET_BOX)

    print(f"Sokoban tiles: EMPTY={_EMPTY} WALL={_WALL} AGENT={_AGENT} "
          f"BOX={_BOX} TARGET={_TARGET} TARGET_AGENT={_TARGET_AGENT} "
          f"TARGET_BOX={_TARGET_BOX}  → num_tiles={NUM_TILES}")

    def extract(states):
        fg = states.fixed_grid
        vg = states.variable_grid
        mask_ta = (fg == TARGET) & (vg == AGENT)
        mask_tb = (fg == TARGET) & (vg == BOX)
        combined = jnp.where(mask_ta, _TARGET_AGENT,
                   jnp.where(mask_tb, _TARGET_BOX,
                   jnp.maximum(vg, fg)))
        return combined.reshape(combined.shape[0], OBS_DIM).astype(jnp.int32)

    def oracle(key, obs, action):
        grid = obs.reshape(N, N)
        is_agent = (grid == _AGENT) | (grid == _TARGET_AGENT)
        agent_idx = jnp.argmax(is_agent.ravel())
        ar, ac = jnp.divmod(agent_idx, N)

        dr, dc = DR[action], DC[action]
        nr, nc = ar + dr, ac + dc

        in_bounds = (nr >= 0) & (nr < N) & (nc >= 0) & (nc < N)
        nr_s = jnp.clip(nr, 0, N - 1)
        nc_s = jnp.clip(nc, 0, N - 1)

        dest = grid[nr_s, nc_s]
        is_wall = dest == _WALL
        has_box = (dest == _BOX) | (dest == _TARGET_BOX)

        br, bc = nr + dr, nc + dc
        b_in_bounds = (br >= 0) & (br < N) & (bc >= 0) & (bc < N)
        br_s = jnp.clip(br, 0, N - 1)
        bc_s = jnp.clip(bc, 0, N - 1)

        box_dest = grid[br_s, bc_s]
        box_blocked = (~b_in_bounds
                       | (box_dest == _WALL)
                       | (box_dest == _BOX)
                       | (box_dest == _TARGET_BOX))

        can_simple = in_bounds & ~is_wall & ~has_box
        can_push   = in_bounds & has_box & ~box_blocked

        agent_leaves = jnp.where(
            grid[ar, ac] == _TARGET_AGENT, _TARGET, _EMPTY)

        agent_arrives = jnp.where(dest == _TARGET, _TARGET_AGENT, _AGENT)
        simple_grid = (grid
            .at[ar, ac].set(agent_leaves)
            .at[nr_s, nc_s].set(agent_arrives))

        agent_on_box = jnp.where(
            dest == _TARGET_BOX, _TARGET_AGENT, _AGENT)
        box_on_new = jnp.where(
            box_dest == _TARGET, _TARGET_BOX, _BOX)
        push_grid = (grid
            .at[ar, ac].set(agent_leaves)
            .at[nr_s, nc_s].set(agent_on_box)
            .at[br_s, bc_s].set(box_on_new))

        new_grid = jnp.where(can_push, push_grid,
                   jnp.where(can_simple, simple_grid, grid))

        return new_grid.reshape(OBS_DIM)

    def obs_to_state(obs_flat, real_state):
        grid = obs_flat.reshape(N, N)
        fixed = jnp.where(
            grid == _WALL, WALL,
            jnp.where(
                (grid == _TARGET) | (grid == _TARGET_AGENT) | (grid == _TARGET_BOX),
                TARGET, EMPTY)).astype(jnp.uint8)
        variable = jnp.where(
            (grid == _AGENT) | (grid == _TARGET_AGENT), AGENT,
            jnp.where(
                (grid == _BOX) | (grid == _TARGET_BOX), BOX,
                EMPTY)).astype(jnp.uint8)
        is_agent = (grid == _AGENT) | (grid == _TARGET_AGENT)
        agent_idx = jnp.argmax(is_agent.ravel())
        ar, ac = jnp.divmod(agent_idx, N)
        return real_state.replace(
            fixed_grid=fixed,
            variable_grid=variable,
            agent_location=jnp.array([ar, ac], dtype=jnp.int32),
        )

    return EnvMeta(
        obs_dim=OBS_DIM,
        num_tiles=NUM_TILES,
        num_actions=4,
        grid_shape=(N, N),
        make_env=lambda: Sokoban(
            generator=HuggingFaceDeepMindGenerator(
                "unfiltered-train", proportion_of_files=0.01)),
        extract_obs=extract,
        oracle_step=oracle,
        obs_to_state=obs_to_state,
    )

# =============================================================================
# 2048 environment
# =============================================================================

def make_2048_meta(board_size=4, max_tile_exp=16):
    """2048 environment metadata.

    Board values are exponents of 2:
        0 = empty, 1 = tile 2, 2 = tile 4, ..., 11 = tile 2048
    max_tile_exp is the vocab size for the tile embedding.

    Dynamics are stochastic: after each valid move a random empty cell
    receives value 1 (90%) or 2 (10%).
    """
    from jumanji.environments.logic.game_2048.env import Game2048
    from jumanji.environments.logic.game_2048.types import State as G2048State
    from jumanji.environments.logic.game_2048.utils import move, can_move

    N = board_size
    OBS_DIM = N * N
    NUM_TILES = max_tile_exp

    def extract(states):
        B = states.board.shape[0]
        return states.board.reshape(B, OBS_DIM).astype(jnp.int32)

    def oracle(key, obs, action):
        board = obs.reshape(N, N)

        # Deterministic slide + merge
        merged, _reward = move(board, action)

        # Did the board change?
        changed = jnp.any(merged != board)

        # Stochastic tile spawn
        k1, k2 = jax.random.split(key)
        empty = (merged.ravel() == 0)
        has_empty = jnp.any(empty)

        tile_idx = jax.random.choice(
            k1, jnp.arange(OBS_DIM),
            p=empty.astype(jnp.float32),
        )
        pos = jnp.divmod(tile_idx, N)

        cell_value = jax.random.choice(
            k2,
            jnp.array([1, 2], dtype=jnp.int32),
            p=jnp.array([0.9, 0.1]),
        )

        spawned = merged.at[pos].set(cell_value)

        final = jnp.where(changed & has_empty, spawned, merged)
        return final.reshape(OBS_DIM)

    def obs_to_state(obs_flat, real_state):
        board = obs_flat.reshape(N, N).astype(jnp.int32)
        action_mask = jax.vmap(can_move, (None, 0))(
            board, jnp.arange(4))
        return real_state.replace(
            board=board,
            action_mask=action_mask,
        )

    return EnvMeta(
        obs_dim=OBS_DIM,
        num_tiles=NUM_TILES,
        num_actions=4,
        grid_shape=(N, N),
        make_env=lambda: Game2048(board_size=N),
        extract_obs=extract,
        oracle_step=oracle,
        obs_to_state=obs_to_state,
    )


# =============================================================================
# Connector environment  (multi-agent routing on a grid)
# =============================================================================

class _ConnectorJointActionWrapper:
    """Wraps Jumanji Connector to accept scalar joint actions.

    The rest of the codebase (_scramble, collect_offline_dataset, etc.)
    calls env.step(state, scalar_action). This wrapper decodes the scalar
    joint action into the (num_agents,) array that Connector expects.
    """

    def __init__(self, env, num_agents, actions_per_agent=5):
        self._env = env
        self._A = num_agents
        self._apa = actions_per_agent
        self._powers = jnp.array(
            [actions_per_agent ** i for i in range(num_agents)],
            dtype=jnp.int32)

    def _decode(self, joint_action):
        return (joint_action // self._powers) % self._apa

    def reset(self, key):
        return self._env.reset(key)

    def step(self, state, joint_action):
        multi = self._decode(joint_action)
        return self._env.step(state, multi)

    def render(self, state):
        return self._env.render(state)

    def animate(self, states, **kwargs):
        return self._env.animate(states, **kwargs)


def make_connector_meta(grid_size: int = 10, num_agents: int = 3) -> EnvMeta:
    """Connector: route N agent heads to their targets on a discrete grid.

    Tile encoding:
        0              = empty
        3*i + 1        = path laid by agent i  (i is 0-indexed)
        3*i + 2        = current head of agent i
        3*i + 3        = target of agent i
    So num_tiles = 3*num_agents + 1.

    Per-agent actions: 0=NOOP, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT
    Joint action: scalar in [0, 5^num_agents) flattened from per-agent.
    """
    from jumanji.environments.routing.connector.env import Connector
    from jumanji.environments.routing.connector.generator import RandomWalkGenerator

    N = grid_size
    OBS_DIM = N * N
    A = num_agents
    ACTIONS_PER_AGENT = 5
    NUM_TILES = 3 * A + 1
    NUM_JOINT_ACTIONS = ACTIONS_PER_AGENT ** A

    DR = jnp.array([0, -1, 0, 1, 0])
    DC = jnp.array([0, 0, 1, 0, -1])

    _powers = jnp.array(
        [ACTIONS_PER_AGENT ** i for i in range(A)], dtype=jnp.int32)

    def _joint_to_multi(joint_action):
        return (joint_action // _powers) % ACTIONS_PER_AGENT

    def extract(states):
        B = states.grid.shape[0]
        return states.grid.reshape(B, OBS_DIM).astype(jnp.int32)

    def oracle(key, obs, action):
        grid = obs.reshape(N, N)
        multi_act = _joint_to_multi(action)

        def _step_one_agent(grid, agent_idx):
            act = multi_act[agent_idx]
            head_val   = 3 * agent_idx + 2
            target_val = 3 * agent_idx + 3
            path_val   = 3 * agent_idx + 1

            is_head = (grid.ravel() == head_val)
            head_idx = jnp.argmax(is_head)
            has_head = is_head[head_idx]
            hr, hc = jnp.divmod(head_idx, N)

            nr = hr + DR[act]
            nc = hc + DC[act]
            in_bounds = (nr >= 0) & (nr < N) & (nc >= 0) & (nc < N)
            nr_s = jnp.clip(nr, 0, N - 1)
            nc_s = jnp.clip(nc, 0, N - 1)

            dest_val = grid[nr_s, nc_s]
            is_noop = (act == 0)
            can_move = (has_head & ~is_noop & in_bounds
                        & ((dest_val == 0) | (dest_val == target_val)))

            new_grid = jnp.where(
                can_move,
                grid.at[hr, hc].set(path_val).at[nr_s, nc_s].set(head_val),
                grid,
            )
            return new_grid, None

        grid, _ = jax.lax.scan(
            _step_one_agent, grid, jnp.arange(A))
        return grid.reshape(OBS_DIM)

    def obs_to_state(obs_flat, real_state):
        grid = obs_flat.reshape(N, N).astype(jnp.int32)
        return real_state.replace(grid=grid)

    print(f"Connector: {N}x{N} grid, {A} agents, "
          f"{NUM_TILES} tile types, {NUM_JOINT_ACTIONS} joint actions")

    return EnvMeta(
        obs_dim=OBS_DIM,
        num_tiles=NUM_TILES,
        num_actions=NUM_JOINT_ACTIONS,
        grid_shape=(N, N),
        make_env=lambda: _ConnectorJointActionWrapper(
            Connector(generator=RandomWalkGenerator(grid_size=N, num_agents=A)), A),
        extract_obs=extract,
        oracle_step=oracle,
        obs_to_state=obs_to_state,
    )


# =============================================================================
# PacMan environment
# =============================================================================

def make_pacman_meta() -> EnvMeta:
    """PacMan: classic maze with pellets, power-ups, and 4 ghosts.

    The observation is a composite grid built by layering entities onto
    the static wall layout:
        0 = wall
        1 = empty (passable, nothing on it)
        2 = pellet
        3 = power-up
        4 = player
        5 = ghost (normal)
        6 = ghost (frightened / scatter mode)
    So num_tiles = 7.

    Actions: 5 discrete — [0=left, 1=up, 2=right, 3=down, 4=noop]

    oracle_step is None because PacMan dynamics depend on hidden state
    (ghost AI memory, frightened timer, etc.) that can't be recovered
    from the flat obs grid alone. Data collection uses env.step instead.

    Grid indexing (from Jumanji source):
        grid is (x_size, y_size) and indexed as grid[pos.x, pos.y]
        Entity locations stored as [pos.y, pos.x] in arrays
    """
    from jumanji.environments.routing.pac_man.env import PacMan
    from jumanji.environments.routing.pac_man.constants import DEFAULT_MAZE
    from jumanji.environments.routing.pac_man.generator import AsciiGenerator
    from jumanji.environments.routing.pac_man.types import Position

    _gen = AsciiGenerator(DEFAULT_MAZE)
    XSIZE = _gen.x_size
    YSIZE = _gen.y_size
    OBS_DIM = XSIZE * YSIZE
    NUM_PELLET_SLOTS = _gen.pellet_spaces.shape[0]

    NUM_TILES = 7
    NUM_ACTIONS = 5

    print(f"PacMan: {XSIZE}x{YSIZE} grid, {NUM_PELLET_SLOTS} pellet slots, "
          f"{NUM_TILES} tile types, {NUM_ACTIONS} actions")

    def extract(states):
        """Build composite tile grid from PacMan state fields.

        Layering order (later overwrites earlier):
            grid (0=wall, 1=empty) → pellets (2) → power-ups (3) →
            player (4) → ghosts (5 or 6)
        """
        B = states.grid.shape[0]

        # Base: 0=wall, 1=passable
        composite = states.grid.astype(jnp.int32)             # (B, XSIZE, YSIZE)

        # --- Pellets (tile=2) ---
        # pellet_locations: (B, P, 2) stored as [pos.y, pos.x]
        # Collected pellets are zeroed out to [0, 0].
        pl = states.pellet_locations                            # (B, P, 2)
        pl_x = pl[:, :, 1]                                     # (B, P) — grid dim 0
        pl_y = pl[:, :, 0]                                     # (B, P) — grid dim 1
        pl_active = (pl_x != 0) | (pl_y != 0)                  # (B, P)
        bidx_p = jnp.broadcast_to(
            jnp.arange(B)[:, None], (B, NUM_PELLET_SLOTS)).ravel()
        composite = composite.at[
            bidx_p, pl_x.ravel(), pl_y.ravel()
        ].set(jnp.where(pl_active.ravel(), 2,
              composite[bidx_p, pl_x.ravel(), pl_y.ravel()]))

        # --- Power-ups (tile=3) ---
        pu = states.power_up_locations                          # (B, 4, 2)
        pu_x = pu[:, :, 1]                                     # (B, 4)
        pu_y = pu[:, :, 0]                                     # (B, 4)
        pu_active = (pu_x != 0) | (pu_y != 0)                  # (B, 4)
        bidx_4 = jnp.broadcast_to(
            jnp.arange(B)[:, None], (B, 4)).ravel()
        composite = composite.at[
            bidx_4, pu_x.ravel(), pu_y.ravel()
        ].set(jnp.where(pu_active.ravel(), 3,
              composite[bidx_4, pu_x.ravel(), pu_y.ravel()]))

        # --- Player (tile=4) ---
        px = states.player_locations.x                          # (B,) — grid dim 0
        py = states.player_locations.y                          # (B,) — grid dim 1
        composite = composite.at[jnp.arange(B), px, py].set(4)

        # --- Ghosts (tile=5 normal, tile=6 frightened) ---
        gl = states.ghost_locations                             # (B, 4, 2)
        gx = gl[:, :, 1]                                        # (B, 4) — grid dim 0
        gy = gl[:, :, 0]                                        # (B, 4) — grid dim 1
        frightened = (states.frightened_state_time > 0)         # (B,)
        ghost_tile = jnp.where(frightened[:, None],
                               jnp.full((1, 4), 6, jnp.int32),
                               jnp.full((1, 4), 5, jnp.int32))  # (B, 4)
        bidx_4g = jnp.broadcast_to(
            jnp.arange(B)[:, None], (B, 4)).ravel()
        composite = composite.at[
            bidx_4g, gx.ravel(), gy.ravel()
        ].set(ghost_tile.ravel())

        return composite.reshape(B, OBS_DIM)

    def _extract_positions(composite, tile_val, max_count):
        """Extract up to max_count positions of a tile value from the grid.

        Returns (max_count, 2) array in [pos.y, pos.x] format, zero-padded.
        Jumanji convention: locations stored as [y, x], grid indexed [x, y].
        """
        flat = composite.ravel()
        mask = (flat == tile_val)
        indices = jnp.arange(flat.size)
        # Push invalid indices to end so they sort last
        keyed = jnp.where(mask, indices, flat.size)
        sorted_idx = jnp.sort(keyed)[:max_count]
        valid = sorted_idx < flat.size
        x, y = jnp.divmod(sorted_idx, YSIZE)
        x = x * valid
        y = y * valid
        return jnp.stack([y, x], axis=-1).astype(jnp.int32)

    def obs_to_state(obs_flat, real_state):
        """Reconstruct full renderable state from composite obs grid.

        Extracts all entity positions so the renderer shows model
        predictions, not ground truth. Hidden state fields (ghost AI
        internals, visited_index, etc.) are kept from real_state.
        """
        composite = obs_flat.reshape(XSIZE, YSIZE).astype(jnp.int32)

        # Static grid: 0=wall, everything else=passable(1)
        grid = (composite != 0).astype(jnp.int32)

        # Player (tile==4)
        player_idx = jnp.argmax(composite.ravel() == 4)
        player_x, player_y = jnp.divmod(player_idx, YSIZE)

        # Ghosts: tiles 5 (normal) and 6 (frightened), need exactly 4
        ghost_5 = _extract_positions(composite, 5, 4)
        ghost_6 = _extract_positions(composite, 6, 4)
        # Prefer normal ghosts, fill remaining slots with frightened
        n_normal = jnp.sum(composite == 5)
        # Interleave: take normal first, then frightened for remaining slots
        all_ghosts = jnp.concatenate([ghost_5, ghost_6], axis=0)  # (8, 2)
        # Remove zero-padding duplicates: sort nonzero to front
        has_pos = (all_ghosts[:, 0] != 0) | (all_ghosts[:, 1] != 0)
        ghost_keys = jnp.where(has_pos, jnp.arange(8), 8)
        ghost_order = jnp.argsort(ghost_keys)[:4]
        ghost_locs = all_ghosts[ghost_order]  # (4, 2)

        # Frightened state: positive if any tile==6 exists
        has_frightened = jnp.any(composite == 6)
        frightened_time = jnp.where(has_frightened, jnp.int32(15), jnp.int32(0))

        # Pellets (tile==2), fixed shape (NUM_PELLET_SLOTS, 2)
        pellet_locs = _extract_positions(composite, 2, NUM_PELLET_SLOTS)

        # Power-ups (tile==3), fixed shape (4, 2)
        powerup_locs = _extract_positions(composite, 3, 4)

        # Count remaining pellets
        num_pellets = jnp.sum(composite == 2).astype(jnp.int32)

        return real_state.replace(
            grid=grid,
            player_locations=Position(x=player_x, y=player_y),
            ghost_locations=ghost_locs,
            pellet_locations=pellet_locs,
            power_up_locations=powerup_locs,
            frightened_state_time=frightened_time,
            pellets=num_pellets,
        )

    return EnvMeta(
        obs_dim=OBS_DIM,
        num_tiles=NUM_TILES,
        num_actions=NUM_ACTIONS,
        grid_shape=(XSIZE, YSIZE),
        make_env=lambda: PacMan(generator=AsciiGenerator(DEFAULT_MAZE)),
        extract_obs=extract,
        oracle_step=None,
        obs_to_state=obs_to_state,
    )


# =============================================================================
# Data collection helpers
# =============================================================================

def _scramble(env, meta, states, n_steps, key):
    B = jax.tree.leaves(states)[0].shape[0]
    def _step(i, carry):
        s, k = carry
        k = jax.random.fold_in(k, i)
        acts = jax.random.randint(k, (B,), 0, meta.num_actions)
        s, _ = jax.vmap(env.step)(s, acts)
        return s, k
    states, _ = jax.lax.fori_loop(0, n_steps, _step, (states, key))
    return states


def collect_offline_dataset(meta: EnvMeta, size: int, context_len: int,
                            scramble_steps: int = 0, seed: int = 0) -> Dict[str, Any]:
    env = meta.make_env()
    Tc  = context_len
    print(f"Collecting offline dataset ({size} seqs × {Tc} steps)...")

    @jax.jit
    def _collect(key):
        k1, k2, k3 = jax.random.split(key, 3)
        states, _ = jax.vmap(env.reset)(jax.random.split(k1, size))
        states = _scramble(env, meta, states, scramble_steps, k2)
        ctx_boards, ctx_actions = [], []
        for t in range(Tc):
            acts = jax.random.randint(jax.random.fold_in(k3, t), (size,), 0, meta.num_actions)
            ctx_boards.append(meta.extract_obs(states))
            ctx_actions.append(acts)
            states, _ = jax.vmap(env.step)(states, acts)
        return dict(context_boards=jnp.stack(ctx_boards, 1),
                    context_actions=jnp.stack(ctx_actions, 1))

    return jax.tree.map(
        lambda x: x.block_until_ready(),
        _collect(jax.random.PRNGKey(seed + 999_999)),
    )


def collect_val_set(meta: EnvMeta, size: int,
                    scramble_steps: int = 30, seed: int = 0) -> Dict[str, Any]:
    """Generic validation set: random (obs, action, gt_next_obs) tuples.

    Uses oracle_step if available, otherwise falls back to env.step.
    """
    env = meta.make_env()
    print(f"Collecting val set ({size} transitions)...")

    if meta.oracle_step is not None:
        @jax.jit
        def _collect(key):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            states, _ = jax.vmap(env.reset)(jax.random.split(k1, size))
            states = _scramble(env, meta, states, scramble_steps, k2)
            obs     = meta.extract_obs(states)
            actions = jax.random.randint(k3, (size,), 0, meta.num_actions)
            okeys   = jax.random.split(k4, size)
            gt_next = jax.vmap(meta.oracle_step)(okeys, obs, actions)
            return dict(obs=obs, action=actions, gt_next=gt_next)
    else:
        @jax.jit
        def _collect(key):
            k1, k2, k3 = jax.random.split(key, 3)
            states, _ = jax.vmap(env.reset)(jax.random.split(k1, size))
            states = _scramble(env, meta, states, scramble_steps, k2)
            obs     = meta.extract_obs(states)
            actions = jax.random.randint(k3, (size,), 0, meta.num_actions)
            next_states, _ = jax.vmap(env.step)(states, actions)
            gt_next = meta.extract_obs(next_states)
            return dict(obs=obs, action=actions, gt_next=gt_next)

    return jax.tree.map(
        lambda x: x.block_until_ready(),
        _collect(jax.random.PRNGKey(seed + 1_000_000)),
    )


def make_preferences_batch_fn(meta: EnvMeta, batch_size: int,
                              scramble_steps: int = 0,
                              horizon: int = 1) -> Callable:
    """Returns jitted function: key -> preference batch dict (two-trajectory).

    Uses oracle_step if available, otherwise falls back to env.step.
    """
    env = meta.make_env()
    B   = batch_size

    if meta.oracle_step is not None:
        @jax.jit
        def generate(key):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            states, _ = jax.vmap(env.reset)(jax.random.split(k1, B))
            states = _scramble(env, meta, states, scramble_steps, k2)

            start_obs = meta.extract_obs(states)
            acts1 = jax.random.randint(k3, (B, horizon), 0, meta.num_actions)
            acts2 = jax.random.randint(k4, (B, horizon), 0, meta.num_actions)

            def _oracle_rollout(obs, actions, rng):
                all_next = []
                for t in range(horizon):
                    okeys = jax.random.split(jax.random.fold_in(rng, t), B)
                    obs = jax.vmap(meta.oracle_step)(okeys, obs, actions[:, t])
                    all_next.append(obs)
                return jnp.stack(all_next, axis=1)

            rng_oracle = jax.random.fold_in(k1, 42)
            gt1 = _oracle_rollout(start_obs, acts1, rng_oracle)
            gt2 = _oracle_rollout(start_obs, acts2, jax.random.fold_in(rng_oracle, 99))

            return dict(start_obs=start_obs, acts1=acts1, acts2=acts2,
                        gt1=gt1, gt2=gt2)
    else:
        @jax.jit
        def generate(key):
            k1, k2, k3, k4 = jax.random.split(key, 4)
            states, _ = jax.vmap(env.reset)(jax.random.split(k1, B))
            states = _scramble(env, meta, states, scramble_steps, k2)

            start_obs = meta.extract_obs(states)
            acts1 = jax.random.randint(k3, (B, horizon), 0, meta.num_actions)
            acts2 = jax.random.randint(k4, (B, horizon), 0, meta.num_actions)

            def _env_rollout(s, actions):
                all_next = []
                for t in range(horizon):
                    s, _ = jax.vmap(env.step)(s, actions[:, t])
                    all_next.append(meta.extract_obs(s))
                return jnp.stack(all_next, axis=1)

            gt1 = _env_rollout(states, acts1)
            gt2 = _env_rollout(states, acts2)

            return dict(start_obs=start_obs, acts1=acts1, acts2=acts2,
                        gt1=gt1, gt2=gt2)

    return generate


# =============================================================================
# RENEW: Active learning pool + oracle rollout
# =============================================================================

def make_active_pool_fn(meta: EnvMeta, pool_size: int,
                        horizon: int, scramble_steps: int = 0) -> Callable:
    """
    Returns a jitted function: key -> (start_obs, actions)
    that generates a pool of candidate (s0, action_sequence) pairs.
    """
    env = meta.make_env()
    P = pool_size
    H = horizon

    @jax.jit
    def generate_pool(key):
        k1, k2, k3 = jax.random.split(key, 3)
        states, _ = jax.vmap(env.reset)(jax.random.split(k1, P))
        states = _scramble(env, meta, states, scramble_steps, k2)
        start_obs = meta.extract_obs(states)
        actions   = jax.random.randint(k3, (P, H), 0, meta.num_actions)
        return start_obs, actions

    return generate_pool


def make_oracle_rollout_fn(meta: EnvMeta, batch_size: int,
                           horizon: int) -> Callable:
    """
    Returns a jitted function: (start_obs, actions, key) -> gt

    Requires meta.oracle_step (not None). Envs with hidden state
    (e.g. PacMan) cannot use this — they don't support RENEW.
    """
    if meta.oracle_step is None:
        raise ValueError(
            "make_oracle_rollout_fn requires oracle_step but it's None. "
            "This env has hidden state and doesn't support RENEW active learning. "
            "Use standard DLHF instead.")

    B = batch_size
    H = horizon

    @jax.jit
    def oracle_rollout(start_obs, actions, key):
        obs = start_obs
        all_next = []
        for t in range(H):
            okeys = jax.random.split(jax.random.fold_in(key, t), B)
            obs = jax.vmap(meta.oracle_step)(okeys, obs, actions[:, t])
            all_next.append(obs)
        return jnp.stack(all_next, axis=1)

    return oracle_rollout


# =============================================================================
# Heuristic solvers (for visualisation only)
# =============================================================================

def solve_maze_bfs(obs_flat, cols: int) -> list[int]:
    from collections import deque

    obs = np.array(obs_flat).ravel()
    rows = len(obs) // cols
    agent = int(np.argmax(obs == 2))
    target = int(np.argmax(obs == 3))

    DR = [-1, 0, 1, 0]
    DC = [0, 1, 0, -1]

    visited = set()
    queue = deque([(agent, [])])
    visited.add(agent)

    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path
        r, c = divmod(pos, cols)
        for action in range(4):
            nr, nc = r + DR[action], c + DC[action]
            if 0 <= nr < rows and 0 <= nc < cols:
                npos = nr * cols + nc
                if npos not in visited and obs[npos] != 0:
                    visited.add(npos)
                    queue.append((npos, path + [action]))

    return list(np.random.randint(0, 4, size=20))


def solve_sliding_tile(obs_flat, grid_size: int, max_nodes: int = 100_000) -> list[int]:
    import heapq

    N = grid_size
    goal = tuple(range(N * N))
    start = tuple(int(x) for x in np.array(obs_flat).ravel())

    if start == goal:
        return list(np.random.randint(0, 4, size=20))

    goal_pos = {v: (v // N, v % N) for v in range(N * N)}

    def heuristic(state):
        h = 0
        for idx, val in enumerate(state):
            if val != 0:
                gr, gc = goal_pos[val]
                r, c = divmod(idx, N)
                h += abs(r - gr) + abs(c - gc)
        return h

    DR = [-1, 0, 1, 0]
    DC = [0, 1, 0, -1]

    empty_idx = start.index(0)
    h0 = heuristic(start)
    queue = [(h0, 0, start, empty_idx, [])]
    visited = {start}
    counter = 1

    while queue and counter < max_nodes:
        f, _, state, eidx, path = heapq.heappop(queue)
        er, ec = divmod(eidx, N)

        for action in range(4):
            nr, nc = er + DR[action], ec + DC[action]
            if 0 <= nr < N and 0 <= nc < N:
                nidx = nr * N + nc
                new_state = list(state)
                new_state[eidx], new_state[nidx] = new_state[nidx], new_state[eidx]
                new_state = tuple(new_state)

                if new_state == goal:
                    return path + [action]

                if new_state not in visited:
                    visited.add(new_state)
                    g = len(path) + 1
                    h = heuristic(new_state)
                    heapq.heappush(queue, (g + h, counter, new_state, nidx, path + [action]))
                    counter += 1

    return list(np.random.randint(0, 4, size=20))


# =============================================================================
# PacMan heuristic solver (reactive, one step at a time)
# =============================================================================

def solve_pacman_step(obs_flat, xsize: int, ysize: int) -> int:
    """Greedy BFS heuristic: find shortest path to nearest pellet/power-up
    while avoiding ghosts.

    Operates on the composite tile obs:
        0=wall, 1=empty, 2=pellet, 3=powerup, 4=player, 5=ghost, 6=frightened

    Actions: 0=left(dx=-1), 1=up(dy=-1), 2=right(dx=+1), 3=down(dy=+1), 4=noop

    Returns the single best action for this step.
    """
    from collections import deque

    obs = np.array(obs_flat).reshape(xsize, ysize)

    # Find player
    player_flat = int(np.argmax(np.array(obs_flat) == 4))
    px, py = divmod(player_flat, ysize)

    # Ghost positions (normal ghosts = 5)
    ghost_mask = (obs == 5)
    # Build danger zone: ghost cells + adjacent cells for normal ghosts
    danger = np.zeros((xsize, ysize), dtype=bool)
    danger |= ghost_mask
    DX = [-1, 0, 1, 0]
    DY = [0, -1, 0, 1]
    ghost_xs, ghost_ys = np.where(ghost_mask)
    for gx, gy in zip(ghost_xs, ghost_ys):
        for dx, dy in zip(DX, DY):
            nx, ny = (gx + dx) % xsize, (gy + dy) % ysize
            danger[nx, ny] = True

    # Targets: pellets (2), power-ups (3), frightened ghosts (6)
    targets = (obs == 2) | (obs == 3) | (obs == 6)

    # Action deltas: 0=left(dx=-1), 1=up(dy=-1), 2=right(dx=+1), 3=down(dy=+1)
    ACT_DX = [-1, 0, 1, 0]
    ACT_DY = [0, -1, 0, 1]

    def bfs(avoid_danger: bool):
        visited = set()
        visited.add((px, py))
        queue = deque()
        for act in range(4):
            nx = (px + ACT_DX[act]) % xsize
            ny = (py + ACT_DY[act]) % ysize
            if obs[nx, ny] == 0:
                continue
            if avoid_danger and danger[nx, ny]:
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if targets[nx, ny]:
                return act
            queue.append((nx, ny, act))

        while queue:
            cx, cy, first_act = queue.popleft()
            for dx, dy in zip(DX, DY):
                nx = (cx + dx) % xsize
                ny = (cy + dy) % ysize
                if (nx, ny) in visited:
                    continue
                if obs[nx, ny] == 0:
                    continue
                if avoid_danger and danger[nx, ny]:
                    continue
                visited.add((nx, ny))
                if targets[nx, ny]:
                    return first_act
                queue.append((nx, ny, first_act))
        return None

    # Try with ghost avoidance first
    act = bfs(avoid_danger=True)
    if act is not None:
        return act

    # Fall back: ignore danger zone
    act = bfs(avoid_danger=False)
    if act is not None:
        return act

    # Last resort: any valid non-wall move
    for a in range(4):
        nx = (px + ACT_DX[a]) % xsize
        ny = (py + ACT_DY[a]) % ysize
        if obs[nx, ny] != 0:
            return a

    return 4  # noop