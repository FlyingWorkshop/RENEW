"""
heuristic_solvers.py
====================
Heuristic policies for generating visually compelling rollouts.

Operates on FLAT 1-D int32 observations as produced by EnvMeta.extract_obs.
Tile encodings must match env.py:

    Maze:    0=wall, 1=empty, 2=agent, 3=goal
    Sliding: 0=blank, 1..N*N-1 = tiles (solved = [0,1,2,...,N*N-1])
    Sokoban: 0=empty, 1=wall, 2=target, 3=agent, 4=box, 5=target_agent, 6=target_box

Actions (all envs): 0=up, 1=right, 2=down, 3=left
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


# =====================================================================
# Maze — BFS shortest path (uses env.solve_maze_bfs if available)
# =====================================================================

def solve_maze(obs_flat: np.ndarray, cols: int) -> List[int]:
    """BFS from agent (tile=2) to goal (tile=3).

    obs_flat: 1-D int array, length = rows * cols.
    Returns full action sequence, or random fallback.
    """
    obs = np.asarray(obs_flat).ravel()
    rows = len(obs) // cols

    agent = int(np.argmax(obs == 2))
    target = int(np.argmax(obs == 3))

    if obs[agent] != 2 or obs[target] != 3:
        return list(np.random.randint(0, 4, size=20))

    DR = [-1, 0, 1, 0]
    DC = [0, 1, 0, -1]

    visited = {agent}
    queue = deque([(agent, [])])

    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path
        r, c = divmod(pos, cols)
        for action in range(4):
            nr, nc = r + DR[action], c + DC[action]
            if 0 <= nr < rows and 0 <= nc < cols:
                npos = nr * cols + nc
                if npos not in visited and obs[npos] != 0:  # 0 = wall
                    visited.add(npos)
                    queue.append((npos, path + [action]))

    return list(np.random.randint(0, 4, size=20))


# =====================================================================
# Sliding Tile — A* with Manhattan heuristic
# =====================================================================

def solve_sliding(obs_flat: np.ndarray, grid_size: int,
                  max_nodes: int = 100_000) -> List[int]:
    """A* search toward the solved configuration.

    Solved state: (0, 1, 2, ..., N*N-1) — blank=0 at top-left.
    Actions move the BLANK: 0=up, 1=right, 2=down, 3=left.
    """
    import heapq

    N = grid_size
    goal = tuple(range(N * N))
    start = tuple(int(x) for x in np.asarray(obs_flat).ravel())

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
                    heapq.heappush(queue, (g + h, counter, new_state,
                                           nidx, path + [action]))
                    counter += 1

    # A* exhausted budget — fall back to greedy
    return _greedy_sliding(start, N, n_steps=20)


def _greedy_sliding(start: tuple, N: int, n_steps: int) -> List[int]:
    """Greedy fallback: pick action that most reduces Manhattan distance."""
    goal_pos = {v: (v // N, v % N) for v in range(N * N)}
    DR = [-1, 0, 1, 0]
    DC = [0, 1, 0, -1]

    def manhattan(state):
        return sum(abs(divmod(i, N)[0] - goal_pos[v][0]) +
                   abs(divmod(i, N)[1] - goal_pos[v][1])
                   for i, v in enumerate(state) if v != 0)

    actions = []
    state = list(start)
    visited = {tuple(state)}

    for _ in range(n_steps):
        eidx = state.index(0)
        er, ec = divmod(eidx, N)
        best_a, best_score, best_s = None, manhattan(state), None

        order = list(range(4))
        np.random.shuffle(order)

        for a in order:
            nr, nc = er + DR[a], ec + DC[a]
            if 0 <= nr < N and 0 <= nc < N:
                nidx = nr * N + nc
                ns = list(state)
                ns[eidx], ns[nidx] = ns[nidx], ns[eidx]
                ts = tuple(ns)
                if ts not in visited:
                    sc = manhattan(ns)
                    if sc < best_score:
                        best_a, best_score, best_s = a, sc, ns

        if best_a is None:
            # Take any valid unvisited move
            for a in order:
                nr, nc = er + DR[a], ec + DC[a]
                if 0 <= nr < N and 0 <= nc < N:
                    nidx = nr * N + nc
                    ns = list(state)
                    ns[eidx], ns[nidx] = ns[nidx], ns[eidx]
                    if tuple(ns) not in visited:
                        best_a, best_s = a, ns
                        break
            if best_a is None:
                best_a = int(np.random.randint(4))
                nr, nc = er + DR[best_a], ec + DC[best_a]
                nr, nc = max(0, min(N-1, nr)), max(0, min(N-1, nc))
                nidx = nr * N + nc
                best_s = list(state)
                best_s[eidx], best_s[nidx] = best_s[nidx], best_s[eidx]

        actions.append(best_a)
        state = best_s
        visited.add(tuple(state))

    return actions


# =====================================================================
# Sokoban — greedy box-pushing toward targets
# =====================================================================
#
# Tiles: 0=empty, 1=wall, 2=target, 3=agent, 4=box, 5=target_agent, 6=target_box

_SOK_EMPTY        = 0
_SOK_WALL         = 1
_SOK_TARGET       = 2
_SOK_AGENT        = 3
_SOK_BOX          = 4
_SOK_TARGET_AGENT = 5
_SOK_TARGET_BOX   = 6

_DR = [-1, 0, 1, 0]
_DC = [0, 1, 0, -1]


def solve_sokoban(obs_flat: np.ndarray, grid_size: int,
                  max_steps: int = 30) -> List[int]:
    """Greedy heuristic: navigate agent to push nearest box toward target.

    Not optimal, but produces purposeful-looking movement for figures.
    """
    N = grid_size
    grid = np.asarray(obs_flat).reshape(N, N)

    actions = []
    for _ in range(max_steps):
        a = _sokoban_step(grid, N)
        if a is None:
            break
        actions.append(a)
        grid = _sokoban_apply(grid, a, N)

    # Pad with random if short
    while len(actions) < max_steps:
        actions.append(int(np.random.randint(4)))

    return actions


def _sokoban_step(grid: np.ndarray, N: int) -> Optional[int]:
    """Pick a single greedy action."""
    # Find agent
    agent_mask = (grid == _SOK_AGENT) | (grid == _SOK_TARGET_AGENT)
    if not np.any(agent_mask):
        return None
    agent_pos = np.argwhere(agent_mask)[0]
    ar, ac = int(agent_pos[0]), int(agent_pos[1])

    # Find unsolved boxes and targets
    boxes = [(int(r), int(c)) for r, c in
             np.argwhere((grid == _SOK_BOX))]
    targets = [(int(r), int(c)) for r, c in
               np.argwhere((grid == _SOK_TARGET) | (grid == _SOK_TARGET_AGENT))]
    # Also count target_agent as still a target (agent is standing on it
    # but we need that target for a box eventually)

    if not boxes or not targets:
        return None

    # Find closest (box, target) pair
    best_pair = None
    best_dist = float("inf")
    for br, bc in boxes:
        for tr, tc in targets:
            d = abs(br - tr) + abs(bc - tc)
            if 0 < d < best_dist:
                best_dist = d
                best_pair = ((br, bc), (tr, tc))

    if best_pair is None:
        return None  # all boxes on targets

    (br, bc), (tr, tc) = best_pair

    # Push direction: which way should box move to approach target?
    if abs(tr - br) >= abs(tc - bc) and tr != br:
        push_dr = 1 if tr > br else -1
        push_dc = 0
    elif tc != bc:
        push_dr = 0
        push_dc = 1 if tc > bc else -1
    else:
        return None

    # Agent needs to be at the opposite side of box from push direction
    push_from = (br - push_dr, bc - push_dc)

    # BFS navigate agent to push_from position
    path = _sokoban_bfs_agent(grid, N, (ar, ac), push_from)
    if path:
        return path[0]

    # Can't reach push position — try any move toward the box
    best_a = None
    best_d = abs(ar - br) + abs(ac - bc)
    for a in range(4):
        nr, nc = ar + _DR[a], ac + _DC[a]
        if 0 <= nr < N and 0 <= nc < N:
            tile = grid[nr, nc]
            if tile in (_SOK_EMPTY, _SOK_TARGET, _SOK_TARGET_AGENT):
                d = abs(nr - br) + abs(nc - bc)
                if d < best_d:
                    best_d = d
                    best_a = a

    return best_a if best_a is not None else int(np.random.randint(4))


def _sokoban_bfs_agent(grid: np.ndarray, N: int,
                       start: Tuple[int, int],
                       goal: Tuple[int, int]) -> List[int]:
    """BFS for agent movement only (no pushing)."""
    if start == goal:
        return []

    passable = {_SOK_EMPTY, _SOK_TARGET, _SOK_AGENT, _SOK_TARGET_AGENT}
    visited = {start}
    queue = deque([(start, [])])

    while queue:
        (r, c), path = queue.popleft()
        for a in range(4):
            nr, nc = r + _DR[a], c + _DC[a]
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in visited:
                tile = grid[nr, nc]
                if tile in passable or (nr, nc) == goal:
                    if (nr, nc) == goal:
                        return path + [a]
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [a]))
    return []


def _sokoban_apply(grid: np.ndarray, action: int, N: int) -> np.ndarray:
    """Apply an action to the Sokoban grid (numpy, for planning only)."""
    grid = grid.copy()
    agent_mask = (grid == _SOK_AGENT) | (grid == _SOK_TARGET_AGENT)
    if not np.any(agent_mask):
        return grid
    pos = np.argwhere(agent_mask)[0]
    ar, ac = int(pos[0]), int(pos[1])

    dr, dc = _DR[action], _DC[action]
    nr, nc = ar + dr, ac + dc

    if not (0 <= nr < N and 0 <= nc < N):
        return grid

    dest = grid[nr, nc]

    # Wall — no move
    if dest == _SOK_WALL:
        return grid

    # Box or box-on-target — try to push
    if dest in (_SOK_BOX, _SOK_TARGET_BOX):
        br, bc = nr + dr, nc + dc
        if not (0 <= br < N and 0 <= bc < N):
            return grid
        bdest = grid[br, bc]
        if bdest in (_SOK_WALL, _SOK_BOX, _SOK_TARGET_BOX):
            return grid
        # Push succeeds
        grid[ar, ac] = _SOK_TARGET if grid[ar, ac] == _SOK_TARGET_AGENT else _SOK_EMPTY
        grid[nr, nc] = _SOK_TARGET_AGENT if dest == _SOK_TARGET_BOX else _SOK_AGENT
        grid[br, bc] = _SOK_TARGET_BOX if bdest == _SOK_TARGET else _SOK_BOX
        return grid

    # Empty or target — simple move
    if dest in (_SOK_EMPTY, _SOK_TARGET):
        grid[ar, ac] = _SOK_TARGET if grid[ar, ac] == _SOK_TARGET_AGENT else _SOK_EMPTY
        grid[nr, nc] = _SOK_TARGET_AGENT if dest == _SOK_TARGET else _SOK_AGENT
        return grid

    return grid


# =====================================================================
# Dispatcher
# =====================================================================

def heuristic_plan(env_name: str, obs_flat: np.ndarray,
                   grid_shape: Tuple[int, int],
                   n_steps: int) -> List[int]:
    """Return up to n_steps actions from the heuristic solver.

    Args:
        env_name:   e.g. "maze10", "sliding5", "sokoban"
        obs_flat:   1-D int array from EnvMeta.extract_obs
        grid_shape: (rows, cols) from EnvMeta.grid_shape
        n_steps:    desired number of actions

    Pads with random actions if plan is shorter than requested.
    """
    name = env_name.lower()
    rows, cols = grid_shape

    if "maze" in name:
        plan = solve_maze(obs_flat, cols)
    elif "sliding" in name:
        plan = solve_sliding(obs_flat, cols)
    elif "sokoban" in name:
        plan = solve_sokoban(obs_flat, cols, max_steps=n_steps)
    else:
        plan = list(np.random.randint(0, 4, size=n_steps))

    # Truncate or pad
    while len(plan) < n_steps:
        plan.append(int(np.random.randint(4)))
    return plan[:n_steps]


def heuristic_step(env_name: str, obs_flat: np.ndarray,
                   grid_shape: Tuple[int, int]) -> int:
    """Single reactive step."""
    plan = heuristic_plan(env_name, obs_flat, grid_shape, n_steps=1)
    return plan[0]