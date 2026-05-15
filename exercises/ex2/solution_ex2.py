"""
Assignment 2 — Probabilistic Box Pushing
=========================================
RPG-driven solution.

  - Both Part 1 (online planning) and Part 2 (MPI) use the SAME relaxed-graph
    machinery: BFS in the box "push-graph" (i.e. ignoring agents and other
    boxes, only respecting walls).
  - Part 2's MPI uses the RPG distance as a value-function warm start
      V_init[s] = gamma ** sum_b ( push_BFS_dist(b, nearest_goal) )
  - Part 1's online planner picks next env actions by:
      1.  RPG assignment of boxes -> goals (min total push-distance).
      2.  Pick the next intended push step from each box's RPG path.
      3.  Single-agent BFS over (pos, dir) to reach the right "stance"
          (one cell behind the box, facing the push direction).
      4.  Forward (=push attempt) when the stance is correct.

  Currently configured for 2 agents + 2 small boxes + 2 goals (no heavy).
  The code also accepts maps with a heavy box / 3 goals.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from collections import deque
from itertools import permutations

import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
ASCII_MAP = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW",
]


# ===========================================================================
# RPG / BFS helpers (used by BOTH parts)
# ===========================================================================

def _scan_world(env):
    """Return (walls, goal_cells, small_boxes_sorted, heavy_or_None)."""
    walls, goal_cells, small, heavy = set(), set(), [], None
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls.add((x, y))
            elif cell.type == "goal":
                goal_cells.add((x, y))
            elif cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy = (x, y)
                else:
                    small.append((x, y))
    for gp in getattr(env, "goal_positions", []):
        goal_cells.add(tuple(gp))
    small.sort()
    return walls, goal_cells, small, heavy


def _push_neighbors(c, walls, width, height):
    """Cells reachable by ONE relaxed push from c (ignoring agents/other boxes).
    A push from c to c+d is valid iff c+d is in-bounds non-wall AND c-d is
    in-bounds non-wall (the agent has to stand at c-d to push)."""
    out = []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        nc     = (c[0] + dx, c[1] + dy)
        behind = (c[0] - dx, c[1] - dy)
        if not (0 <= nc[0] < width and 0 <= nc[1] < height): continue
        if not (0 <= behind[0] < width and 0 <= behind[1] < height): continue
        if nc in walls: continue
        if behind in walls: continue
        out.append((nc, (dx, dy)))
    return out


def _push_bfs_distances(source, walls, width, height):
    """All-pairs BFS distance in the push-graph FROM `source`.
    Returns dict cell -> distance."""
    dist = {source: 0}
    q = deque([source])
    while q:
        c = q.popleft()
        for nc, _ in _push_neighbors(c, walls, width, height):
            if nc not in dist:
                dist[nc] = dist[c] + 1
                q.append(nc)
    return dist


def _push_bfs_path(source, target, walls, width, height):
    """Return [(cell_1, push_vec_1), ..., (target, push_vec)] or None."""
    if source == target:
        return []
    parent = {source: None}      # cell -> (prev_cell, push_vec_used)
    q = deque([source])
    while q:
        c = q.popleft()
        for nc, vec in _push_neighbors(c, walls, width, height):
            if nc not in parent:
                parent[nc] = (c, vec)
                if nc == target:
                    path = []
                    cur = nc
                    while parent[cur] is not None:
                        prev, pv = parent[cur]
                        path.append((cur, pv))
                        cur = prev
                    return list(reversed(path))
                q.append(nc)
    return None


def _agent_bfs_first_action(start_pos, start_dir,
                            target_pos, target_dir,
                            walls, blocked, width, height):
    """Plan agent: BFS over (pos, dir).  Returns the FIRST primitive action
    (0=left, 1=right, 2=forward) on the shortest path to (target_pos, target_dir),
    or None if already there / unreachable."""
    start  = (start_pos, start_dir)
    target = (target_pos, target_dir)
    if start == target:
        return None

    parent = {start: None}     # state -> (prev_state, action_taken)
    q      = deque([start])
    while q:
        cur = q.popleft()
        pos, d = cur
        for act in (0, 1, 2):
            if act == 0:
                ns = (pos, (d - 1) % 4)
            elif act == 1:
                ns = (pos, (d + 1) % 4)
            else:
                vec  = DIR_TO_VEC[d]
                npos = (pos[0] + vec[0], pos[1] + vec[1])
                if not (0 <= npos[0] < width and 0 <= npos[1] < height): continue
                if npos in walls or npos in blocked: continue
                ns = (npos, d)
            if ns in parent:
                continue
            parent[ns] = (cur, act)
            if ns == target:
                # walk back and return the first action
                while parent[ns][0] != start:
                    ns = parent[ns][0]
                return parent[ns][1]
            q.append(ns)
    return None


def _assign_boxes_to_goals(boxes, goal_cells, walls, width, height):
    """Try all (boxes -> goals) permutations, pick the assignment minimising
    sum of push-BFS distances.  Returns dict {box_pos: goal_pos} or None."""
    goals = list(goal_cells)
    if len(boxes) > len(goals):
        return None
    best_cost, best_pairs = float("inf"), None
    for perm in permutations(goals, len(boxes)):
        cost = 0
        ok   = True
        for b, g in zip(boxes, perm):
            d = _push_bfs_distances(b, walls, width, height).get(g)
            if d is None:
                ok = False
                break
            cost += d
        if ok and cost < best_cost:
            best_cost, best_pairs = cost, list(zip(boxes, perm))
    return dict(best_pairs) if best_pairs else None


# ===========================================================================
# Part 1 — Online Planning (RPG-driven, no PDDL)
# ===========================================================================

def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Step-by-step replanning using the RPG/BFS pipeline:

      • RPG: build the box push-graph from the static walls.
      • Each iter:
          - choose the best box->goal assignment by push-BFS distance,
          - for each agent, pick the next push step its box needs,
          - BFS over (agent_pos, agent_dir) to take the first primitive
            action toward the push stance,
          - if already at the stance: forward (push attempt).

    Returns the total number of env steps taken.
    """
    env.reset()
    width, height = env.width, env.height

    # Pre-collect static walls (the map doesn't change during the episode).
    walls = set()
    for y in range(height):
        for x in range(width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.add((x, y))

    agents = env.possible_agents
    total_steps = 0

    for _ in range(max_replans):
        if env._all_boxes_on_goals():
            break
        if total_steps >= env.max_steps:
            break

        _, goal_cells, small, heavy = _scan_world(env)

        # Boxes that aren't yet on a goal.
        pending = [b for b in small if b not in goal_cells]
        if heavy is not None and heavy not in goal_cells:
            pending.append(heavy)
        if not pending:
            break

        # Goals not already occupied by some box.
        occupied = {b for b in small + ([heavy] if heavy else []) if b in goal_cells}
        free_goals = [g for g in goal_cells if g not in occupied]

        assignment = _assign_boxes_to_goals(pending, free_goals, walls, width, height)
        if assignment is None:
            break  # not solvable from here

        # ── For each agent, decide its next env action ──────────────────────
        # Simple agent-to-box pairing: assign each agent to the closest
        # pending non-heavy box; heavy needs both agents (handled at the end).
        small_pending = [b for b in pending if b != heavy]
        heavy_pending = (heavy is not None and heavy not in goal_cells)

        joint_action = {}

        if small_pending:
            # Pair agents with pending small boxes.
            # Greedy: agent 0 takes the smaller-key pending small box first.
            agent_box = {}
            free_agents = list(agents)
            for box in small_pending:
                if not free_agents:
                    break
                # nearest free agent
                free_agents.sort(
                    key=lambda a: abs(env.agent_positions[a][0] - box[0])
                                  + abs(env.agent_positions[a][1] - box[1]))
                pick = free_agents.pop(0)
                agent_box[pick] = box

            for a in agents:
                if a not in agent_box:
                    # No small-box task for this agent: idle rotation.
                    joint_action[a] = 0
                    continue
                box = agent_box[a]
                goal = assignment.get(box)
                if goal is None:
                    joint_action[a] = 0
                    continue

                path = _push_bfs_path(box, goal, walls, width, height)
                if not path:
                    joint_action[a] = 0
                    continue

                next_box_pos, push_vec = path[0]
                stance_pos = (box[0] - push_vec[0], box[1] - push_vec[1])
                stance_dir = None
                for d, v in enumerate(DIR_TO_VEC):
                    if (v[0], v[1]) == push_vec:
                        stance_dir = d
                        break

                # Blocked cells for navigation:
                # walls (handled in BFS), the box itself, the other agent,
                # and the other small boxes (treated as obstacles for now).
                blocked = set()
                blocked.add(box)
                if heavy is not None: blocked.add(heavy)
                for b2 in small:
                    if b2 != box: blocked.add(b2)
                for o in agents:
                    if o != a:
                        blocked.add(env.agent_positions[o])

                apos = env.agent_positions[a]
                adir = env.agent_dirs[a]
                act = _agent_bfs_first_action(
                    apos, adir, stance_pos, stance_dir,
                    walls, blocked, width, height)
                if act is None:
                    # already at stance — push!
                    joint_action[a] = 2
                else:
                    joint_action[a] = act

        elif heavy_pending:
            # Heavy box rendezvous: both agents converge to (heavy - push_vec)
            # facing push_vec, then both forward.
            path = _push_bfs_path(heavy, assignment[heavy], walls, width, height)
            if not path:
                break
            _, push_vec = path[0]
            stance_pos = (heavy[0] - push_vec[0], heavy[1] - push_vec[1])
            stance_dir = None
            for d, v in enumerate(DIR_TO_VEC):
                if (v[0], v[1]) == push_vec:
                    stance_dir = d
                    break

            for a in agents:
                blocked = set()
                blocked.add(heavy)
                for b2 in small:
                    blocked.add(b2)
                # other agent NOT in blocked (overlap allowed, and we WANT them
                # at the same stance cell).
                apos = env.agent_positions[a]
                adir = env.agent_dirs[a]
                act = _agent_bfs_first_action(
                    apos, adir, stance_pos, stance_dir,
                    walls, blocked, width, height)
                joint_action[a] = 2 if act is None else act
        else:
            break

        _, _, terms, truncs, _ = env.step(joint_action)
        total_steps += 1
        if any(terms.values()) or any(truncs.values()):
            break

    return total_steps


# ===========================================================================
# Part 2 — Modified Policy Iteration  (two-phase hierarchical)
# ===========================================================================
#
# Hierarchical decomposition (idea: heavy box is immovable until small boxes
# are placed, so split into two MUCH smaller MDPs):
#
#   Phase 1 MDP:  state = (agent_low_pos, agent_high_pos, b0_pos, b1_pos)
#                 heavy is treated as a WALL.
#                 goal: both small boxes on their assigned goal cells.
#
#   Phase 2 MDP:  state = (agent_low_pos, agent_high_pos, hb_pos)
#                 small boxes (now at their assigned goals) are walls.
#                 goal: heavy on its goal cell.
#
# Live execution:
#   - while not (small boxes on their goals): query phase-1 policy
#   - once they are:                          query phase-2 policy
# Each phase's policy is wrapped with _DirectionPolicyWrapper that converts
# cardinal-direction MPI actions into env primitives (rotate / forward).

def get_state(env) -> tuple:
    """Live env state as a 7-tuple (with directions) — see wrapper for use."""
    agents = env.possible_agents
    a0p = tuple(env.agent_positions[agents[0]])
    d0  = env.agent_dirs[agents[0]]
    if len(agents) >= 2:
        a1p = tuple(env.agent_positions[agents[1]])
        d1  = env.agent_dirs[agents[1]]
    else:
        a1p, d1 = a0p, d0
    _, _, small, heavy = _scan_world(env)
    b0 = small[0] if len(small) > 0 else None
    b1 = small[1] if len(small) > 1 else None
    hb = heavy
    return (a0p, d0, a1p, d1, b0, b1, hb)


def get_mpi_state(env) -> tuple:
    return get_state(env)


def _canon_boxes(b0, b1):
    if b1 is None: return (b0, None)
    if b0 is None: return (b1, None)
    return (b0, b1) if b0 <= b1 else (b1, b0)


def _canon_agents(a0p, a1p):
    """Lexicographic (x, y) order on grid cells.

    Returns (low_pos, high_pos, swapped) where `swapped` is True iff the
    first argument `a0p` was the lexicographically larger cell — used to map
    MPI joint directions back onto env.possible_agents order. Equal positions
    leave order unchanged (swapped False)."""
    if a1p < a0p:
        return a1p, a0p, True
    return a0p, a1p, False


# --------------------------------------------------------------------------
# Generic phase-MDP builder: handles BOTH phases via a uniform spec.
# --------------------------------------------------------------------------
def _build_phase_mdp(env, walls, init_state, goals_for, allow_heavy_push,
                     gamma, step_penalty, push_p, max_bfs_depth=50,
                     max_states=10000):
    """
    Build the transition table for ONE phase.

    init_state, transitions, etc., are pure tuples; this function is shared
    between Phase 1 (boxes in state) and Phase 2 (heavy in state).

    Parameters that VARY between phases are passed in:
      * walls        - cells that block both agent moves AND box pushes
                       (Phase 1: env walls + heavy's start cell;
                        Phase 2: env walls + small boxes' goal cells)
      * init_state   - the phase's initial state tuple
      * goals_for    - dict: box_index -> goal cell.  Terminal when every
                       box in the tuple is at its assigned goal.
      * allow_heavy_push - True only in Phase 2 (joint push semantics)
    """
    width, height = env.width, env.height
    p_move  = env.move_success_prob
    p_push  = push_p
    p_drift = (1.0 - p_move) / 2.0

    state_dim = len(init_state)
    n_agents  = 2
    n_objs    = state_dim - n_agents     # remaining tuple slots are box positions

    def is_invalid(p):
        return (p in walls or
                p[0] < 0 or p[0] >= width or
                p[1] < 0 or p[1] >= height)

    def is_terminal(s):
        objs = s[n_agents:]
        return all(obj == goals_for.get(i) for i, obj in enumerate(objs))

    def normalize_objs(objs):
        # Phase 1 has 2 small boxes (interchangeable -> canonical sort);
        # Phase 2 has 1 heavy box (no sort needed).
        if n_objs == 2:
            cb = _canon_boxes(objs[0], objs[1])
            return (cb[0], cb[1])
        return tuple(objs)

    def normalize(a0p, a1p, objs):
        lo, hi, _ = _canon_agents(a0p, a1p)
        return (lo, hi) + normalize_objs(objs)

    def forward_one(ap, d, objs):
        """One-agent stochastic forward."""
        vec    = DIR_TO_VEC[d]
        target = (ap[0] + vec[0], ap[1] + vec[1])
        if is_invalid(target):
            return [(1.0, ap, objs)]

        # Pushing logic for each object in `objs`.
        for i, obj in enumerate(objs):
            if obj is None or target != obj:
                continue
            # Heavy push by single agent: blocked (heavy in objs and no joint push)
            if allow_heavy_push and n_objs == 1:
                # Heavy needs both agents — single can't move it.
                return [(1.0, ap, objs)]
            push_dest = (target[0] + vec[0], target[1] + vec[1])
            # Can't push into walls, the heavy (here only in phase 2 — handled above),
            # or another box.
            if is_invalid(push_dest):
                return [(1.0, ap, objs)]
            if push_dest in objs:
                return [(1.0, ap, objs)]
            new_objs = tuple(push_dest if j == i else o for j, o in enumerate(objs))
            return [
                (p_push,       target, new_objs),
                (1.0 - p_push, ap,     objs),
            ]

        # Empty / goal cell: forward + drift.
        l_vec = DIR_TO_VEC[(d - 1) % 4]
        r_vec = DIR_TO_VEC[(d + 1) % 4]
        l_pos = (ap[0] + l_vec[0], ap[1] + l_vec[1])
        r_pos = (ap[0] + r_vec[0], ap[1] + r_vec[1])
        def drift_ok(p):
            if is_invalid(p): return False
            return p not in objs
        actual_l = l_pos if drift_ok(l_pos) else ap
        actual_r = r_pos if drift_ok(r_pos) else ap
        bucket = {}
        for pos, p in ((target, p_move), (actual_l, p_drift), (actual_r, p_drift)):
            bucket[pos] = bucket.get(pos, 0.0) + p
        return [(p, pos, objs) for pos, p in bucket.items()]

    def transitions_of(s):
        a0p, a1p = s[0], s[1]
        objs     = s[n_agents:]
        out = {}
        for d0 in (0, 1, 2, 3):
            for d1 in (0, 1, 2, 3):
                joint = (d0, d1)

                # PHASE 2 ONLY: heavy joint push when both agents share a cell.
                if allow_heavy_push and a0p == a1p and d0 == d1:
                    vec   = DIR_TO_VEC[d0]
                    target = (a0p[0] + vec[0], a0p[1] + vec[1])
                    hb = objs[0]
                    if target == hb:
                        push_dest = (hb[0] + vec[0], hb[1] + vec[1])
                        if not is_invalid(push_dest):
                            ns = normalize(hb, hb, (push_dest,))
                            r  = 1.0 if is_terminal(ns) else 0.0
                            out[joint] = [(p_push, ns, r),
                                          (1.0 - p_push, s, 0.0)]
                        else:
                            out[joint] = [(1.0, s, 0.0)]
                        continue

                oa = forward_one(a0p, d0, objs)
                ob = forward_one(a1p, d1, objs)
                combined = {}
                for (p1, na0, oa_objs) in oa:
                    for (p2, na1, ob_objs) in ob:
                        # Merge object updates (each agent updates at most one)
                        merged = []
                        for j in range(n_objs):
                            if oa_objs[j] != objs[j]:
                                merged.append(oa_objs[j])
                            elif ob_objs[j] != objs[j]:
                                merged.append(ob_objs[j])
                            else:
                                merged.append(objs[j])
                        ns = normalize(na0, na1, tuple(merged))
                        combined[ns] = combined.get(ns, 0.0) + p1 * p2
                out[joint] = [(p, ns, 1.0 if is_terminal(ns) else 0.0)
                              for ns, p in combined.items()]
        return out

    init_norm = normalize(init_state[0], init_state[1], init_state[n_agents:])
    trans = {}
    seen  = {init_norm: 0}          # state -> BFS depth
    queue = deque([(init_norm, 0)]) # (state, depth)
    cap_hit = False
    while queue:
        s, depth = queue.popleft()
        if is_terminal(s):
            trans[s] = {}; continue
        # States beyond depth/state caps get no outgoing transitions →
        # the policy wrapper uses its default action for them.  We still add
        # them to `trans` as terminal-like (empty) entries so that V/policy
        # has a key for them and successor lookups don't KeyError.
        if depth >= max_bfs_depth or len(seen) >= max_states:
            cap_hit = True
            trans[s] = {}
            continue
        t = transitions_of(s)
        trans[s] = t
        for outcomes in t.values():
            for _, ns, _ in outcomes:
                if ns not in seen:
                    if len(seen) >= max_states:
                        cap_hit = True
                        # Still register the successor so V is defined for it.
                        seen[ns] = depth + 1
                        trans[ns] = {}
                        continue
                    seen[ns] = depth + 1
                    queue.append((ns, depth + 1))
    if cap_hit:
        print(f"    (cap reached; {len(trans)} states total)")

    return trans, is_terminal, init_norm


def _run_mpi(trans, is_terminal, goal_dist_sum_fn,
             gamma, k, theta, max_outer, step_penalty,
             soft_stable_frac=0.005):
    """Run MPI on a phase MDP with RPG heuristic warm start.

    Convergence:
      - Inner partial-eval loop stops when max |ΔV| < theta.
      - Outer loop stops when fewer than `soft_stable_frac` of states had
        their policy action change in the improvement step (soft criterion
        — strict "no change at all" rarely fires because of tie-breaking).
    `goal_dist_sum_fn(state)` returns h(s) used for V_init = gamma**h."""
    V = {}
    pol = {}
    n_non_terminal = 0
    for s in trans:
        if is_terminal(s):
            V[s] = 0.0
        else:
            h = goal_dist_sum_fn(s)
            V[s] = 0.0 if h == float("inf") else gamma ** max(1, h)
            n_non_terminal += 1
        pol[s] = (1, 1)

    def q(s, a):
        total = 0.0
        for p, ns, r in trans[s][a]:
            total += p * (r + (0.0 if r > 0.0 else step_penalty)
                          + gamma * V[ns])
        return total

    threshold = max(1, int(n_non_terminal * soft_stable_frac))

    for outer in range(max_outer):
        # Partial policy evaluation — k sweeps with early stop on V-stability
        for _ in range(k):
            max_delta = 0.0
            for s in trans:
                if is_terminal(s):
                    V[s] = 0.0; continue
                if not trans[s]:        # boundary / cap-frontier: no actions to evaluate
                    continue
                old = V[s]
                V[s] = q(s, pol[s])
                d = abs(V[s] - old)
                if d > max_delta:
                    max_delta = d
            if max_delta < theta:
                break

        # Greedy policy improvement — count CHANGED actions
        changed = 0
        for s in trans:
            if is_terminal(s):
                continue
            if not trans[s]:            # boundary state — keep default policy
                continue
            best = max(trans[s].keys(), key=lambda a: q(s, a))
            if best != pol[s]:
                pol[s] = best
                changed += 1

        # Soft convergence: stop when policy is mostly stable
        if changed <= threshold:
            print(f"    Converged after {outer + 1} outer iterations "
                  f"(changed={changed}/{n_non_terminal}, threshold={threshold}).")
            break
    return pol, V


# --------------------------------------------------------------------------
# Two-phase Hierarchical Policy
# --------------------------------------------------------------------------
class _HierarchicalPolicy(dict):
    """At lookup time, picks Phase 1 or Phase 2 by inspecting box state.

    inner1 maps Phase-1 state (agent_low_pos, agent_high_pos, b0, b1) -> dirs.
    inner2 maps Phase-2 state (agent_low_pos, agent_high_pos, hb) -> dirs.
    Slots are lex-ordered agent cells (see _canon_agents); __getitem__ maps
    directions back to env.possible_agents order.
    small_goals is the set of cells that, when occupied by small boxes,
    triggers the switch to Phase 2.
    """

    def __init__(self, inner1, inner2, small_goals, heavy_goal):
        super().__init__()
        self.inner1 = inner1
        self.inner2 = inner2
        self.small_goals = set(small_goals or [])
        self.heavy_goal  = heavy_goal

    def __contains__(self, state):
        return isinstance(state, tuple) and len(state) == 7

    def __len__(self):
        return len(self.inner1) + len(self.inner2)

    def __getitem__(self, state):
        a0p, d0, a1p, d1, b0, b1, hb = state
        ca0, ca1, swapped = _canon_agents(a0p, a1p)

        # Phase 2 active iff both small boxes are at small-goal cells.
        small_done = (b0 in self.small_goals and
                      b1 in self.small_goals and
                      b0 is not None and b1 is not None)

        if small_done and self.inner2:
            target = self.inner2.get((ca0, ca1, hb), (1, 1))
        else:
            cb0, cb1 = _canon_boxes(b0, b1)
            target = self.inner1.get((ca0, ca1, cb0, cb1), (1, 1))

        td_lo, td_hi = target
        # Map canonical (low-cell, high-cell) dirs to env agent indices.
        td0, td1 = (td_hi, td_lo) if swapped else (td_lo, td_hi)

        def env_act(current, target_dir):
            if current == target_dir: return 2
            right = (target_dir - current) % 4
            left  = (current - target_dir) % 4
            return 1 if right <= left else 0

        return (env_act(d0, td0), env_act(d1, td1))


def build_transition_model(env):
    """Compatibility shim — Part 2 now uses two phase MDPs internally."""
    return None  # callers should use modified_policy_iteration directly


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 3,
    theta: float = 1e-4,
    max_outer_iters: int = 30,
    step_penalty: float = -0.02,
    max_bfs_depth: int = 50,         # BFS depth cap per phase
    max_states: int = 50_000,        # hard state-count cap per phase
    soft_stable_frac: float = 0.005, # outer-loop "near-stable" tolerance
):
    """Two-phase hierarchical MPI:
        Phase 1: small boxes only (heavy treated as wall).
        Phase 2: heavy box only  (small boxes at goals treated as walls).

    The two sub-policies are exposed via a single _HierarchicalPolicy that
    dispatches based on whether the small boxes are already at their goals.
    """
    env.reset()
    walls, goal_cells, small, heavy = _scan_world(env)
    width, height = env.width, env.height

    # ── RPG assignment: pick which 2 goals are for small boxes + which is heavy ──
    goals_list = sorted(goal_cells)
    p_push     = env.push_success_prob

    if heavy is not None and len(small) == 2 and len(goals_list) == 3:
        # Try each (heavy_goal_choice) × (perm of remaining 2 to small boxes).
        best = None
        for heavy_goal in goals_list:
            remaining = [g for g in goals_list if g != heavy_goal]
            # Phase 1's push-graph walls: env walls + heavy (heavy is immovable).
            p1_walls = walls | {heavy}
            for perm in permutations(remaining, len(small)):
                cost = 0
                ok = True
                for b, g in zip(small, perm):
                    d = _push_bfs_distances(b, p1_walls, width, height).get(g)
                    if d is None: ok = False; break
                    cost += d
                if not ok: continue
                # Phase 2's push-graph walls: env walls + small boxes at their goals.
                p2_walls = walls | set(perm)
                d_heavy  = _push_bfs_distances(heavy, p2_walls, width, height).get(heavy_goal)
                if d_heavy is None: continue
                total = cost + d_heavy
                if best is None or total < best[0]:
                    best = (total, list(perm), heavy_goal, p1_walls, p2_walls)
        if best is None:
            print("  No feasible assignment found.")
            return _HierarchicalPolicy({}, {}, [], None), {}
        _, small_goals, heavy_goal, p1_walls, p2_walls = best
    else:
        # No heavy or different shape: only Phase 1 needed.
        # Try all assignments of small boxes to goals.
        p1_walls = walls | ({heavy} if heavy is not None else set())
        best = None
        for perm in permutations(goals_list, len(small)) if small else [[]]:
            cost = 0
            ok = True
            for b, g in zip(small, perm):
                d = _push_bfs_distances(b, p1_walls, width, height).get(g)
                if d is None: ok = False; break
                cost += d
            if ok and (best is None or cost < best[0]):
                best = (cost, list(perm))
        small_goals = best[1] if best else []
        heavy_goal  = None
        p2_walls    = None

    print(f"  RPG assignment: small_goals={small_goals}, heavy_goal={heavy_goal}")

    # ── Phase 1 MDP ─────────────────────────────────────────────────────────
    print("  Phase 1 (small boxes; heavy as wall): building...")
    t0 = time.time()
    p1_init = (
        tuple(env.agent_positions[env.possible_agents[0]]),
        tuple(env.agent_positions[env.possible_agents[1]])
        if len(env.possible_agents) >= 2
        else tuple(env.agent_positions[env.possible_agents[0]]),
        small[0] if len(small) > 0 else None,
        small[1] if len(small) > 1 else None,
    )
    p1_goals = {0: small_goals[0] if len(small_goals) > 0 else None,
                1: small_goals[1] if len(small_goals) > 1 else None}
    p1_trans, p1_is_terminal, p1_init_norm = _build_phase_mdp(
        env, p1_walls, p1_init, p1_goals, allow_heavy_push=False,
        gamma=gamma, step_penalty=step_penalty, push_p=p_push,
        max_bfs_depth=max_bfs_depth, max_states=max_states)
    print(f"    {len(p1_trans)} states ({time.time()-t0:.2f}s).")

    # Phase 1 RPG heuristic
    p1_dist_tables = {
        g: _push_bfs_distances(g, p1_walls, width, height)
        for g in small_goals
    }
    def p1_h(s):
        objs = s[2:]
        total = 0
        for obj in objs:
            if obj is None: continue
            best = float("inf")
            for _, dm in p1_dist_tables.items():
                d = dm.get(obj, float("inf"))
                if d < best: best = d
            if best == float("inf"): return float("inf")
            total += best
        return total

    print("  Phase 1 MPI...")
    p1_pol, _ = _run_mpi(p1_trans, p1_is_terminal, p1_h,
                         gamma, k, theta, max_outer_iters, step_penalty,
                         soft_stable_frac=soft_stable_frac)

    # ── Phase 2 MDP (only if there's a heavy box) ───────────────────────────
    p2_pol = {}
    if heavy is not None and heavy_goal is not None:
        print("  Phase 2 (heavy box; small at goals as walls): building...")
        t0 = time.time()
        p2_init = (
            tuple(env.agent_positions[env.possible_agents[0]]),
            tuple(env.agent_positions[env.possible_agents[1]])
            if len(env.possible_agents) >= 2
            else tuple(env.agent_positions[env.possible_agents[0]]),
            heavy,
        )
        p2_goals = {0: heavy_goal}
        p2_trans, p2_is_terminal, _ = _build_phase_mdp(
            env, p2_walls, p2_init, p2_goals, allow_heavy_push=True,
            gamma=gamma, step_penalty=step_penalty, push_p=p_push,
            max_bfs_depth=max_bfs_depth, max_states=max_states)
        print(f"    {len(p2_trans)} states ({time.time()-t0:.2f}s).")

        p2_dist_table = _push_bfs_distances(heavy_goal, p2_walls, width, height)
        def p2_h(s):
            hb = s[2]
            d = p2_dist_table.get(hb, float("inf"))
            return d

        print("  Phase 2 MPI...")
        p2_pol, _ = _run_mpi(p2_trans, p2_is_terminal, p2_h,
                             gamma, k, theta, max_outer_iters, step_penalty,
                             soft_stable_frac=soft_stable_frac)

    policy = _HierarchicalPolicy(p1_pol, p2_pol, small_goals, heavy_goal)
    V = {"phase1_size": len(p1_pol), "phase2_size": len(p2_pol)}
    return policy, V




# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================
# Evaluation (do not modify)
# ===========================================================================

def evaluate_policy(policy_fn, env, n_runs: int = 100, max_steps: int = 500):
    """
    Run *policy_fn* for n_runs episodes and return (mean_steps, std_steps).

    Parameters
    ----------
    policy_fn : callable(env, obs) -> dict[agent -> action]
    env       : StochasticMultiAgentBoxPushEnv (reset inside each run)
    n_runs    : number of independent episodes
    max_steps : episode length cap (counts as a failure if hit)
    """
    steps_per_run = []

    for _ in range(n_runs):
        obs, _ = env.reset()
        steps  = 0
        done   = False

        while not done and steps < max_steps:
            actions = policy_fn(env, obs)
            obs, rewards, terms, truncs, _ = env.step(actions)
            steps += 1
            done = any(terms.values()) or any(truncs.values())

        steps_per_run.append(steps)

    return float(np.mean(steps_per_run)), float(np.std(steps_per_run))


# ===========================================================================
# Main — run both algorithms and print results
# ===========================================================================

if __name__ == "__main__":
    env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)

    # -- Part 1: Online Planning ---------------------------------------------
    print("=" * 60)
    print("Part 1 -- Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    # Wrap run_online_planning as a policy function for the evaluator
    def online_planning_policy(env, obs):
        """
        This wrapper runs one COMPLETE episode internally and is only a shim
        for the evaluator.  evaluate_policy will reset the env before each
        call, so we hand control back immediately with a do-nothing action
        after the first step -- the real logic is inside run_online_planning.

        NOTE: because run_online_planning drives the env loop itself, you
        should call it directly (see the manual loop below) for the 100-run
        evaluation; or adapt the evaluate_policy call to suit your design.
        """
        raise NotImplementedError(
            "Adapt this shim or call run_online_planning directly in a loop."
        )

    # Direct evaluation loop for online planning
    online_steps = []
    for i in range(100):
        env_ep = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
        steps = run_online_planning(env_ep)
        online_steps.append(steps)
        if (i + 1) % 10 == 0:
            print(f"  run {i+1}/100 -- steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  ->  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # -- Part 2: Modified Policy Iteration ------------------------------------
    print("=" * 60)
    print("Part 2 -- Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state        = get_mpi_state(env)
        joint_action = policy[state]
        # joint_action is a tuple (action_agent0, action_agent1)
        agents = env.possible_agents
        return {agents[0]: joint_action[0], agents[1]: joint_action[1]}

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              ->  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # -- Summary --------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")
