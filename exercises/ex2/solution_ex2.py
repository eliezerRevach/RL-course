"""
Assignment 2 — Probabilistic Box Pushing
=========================================
Fill in the three TODO sections below:
  1. run_online_planning  — online replanning loop
  2. build_transition_model — MDP transition model (used by MPI)
  3. modified_policy_iteration — MPI algorithm

Do NOT modify evaluate_policy or the __main__ block.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

# ---------------------------------------------------------------------------
# Map used in both parts (same as Assignment 1)
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
# Part 1 — Online Planning
# ===========================================================================

def run_online_planning(env, max_replans: int = 300) -> int:
    """
    Execute one episode using online planning:
      replan from the current state → execute only the first PDDL action → repeat.

    Returns
    -------
    int
        Number of *env* steps taken (counting each rotate/forward individually).
        Returns max_replans * <average_actions_per_plan_step> as a large sentinel
        if the goal was never reached within max_replans replanning calls.
    """
    obs, _ = env.reset()
    total_env_steps = 0
    done = False

    for _ in range(max_replans):
        if done:
            break

        # ── 1. Export current state ──────────────────────────────────
        domain_path, problem_path = generate_pddl_for_env(env)

        # ── 2. Plan ──────────────────────────────────────────────────
        plan = solve_pddl(domain_path, problem_path)
        if not plan or len(plan.actions) == 0:
            break  # goal already reached (planner returns empty plan)

        # ── 3. Execute the first PDDL action ─────────────────────────
        pddl_action   = plan.actions[0]
        agent_targets = extract_target_pos(pddl_action)

        if not agent_targets:
            break

        # Build per-agent action queues (rotations + forward)
        agents_in_action = list(agent_targets.keys())
        action_queues = {
            a: get_required_actions(env, a, agent_targets[a])
            for a in agents_in_action
        }

        # Pad shorter queues so all agents execute their final forward together
        max_len = max(len(q) for q in action_queues.values())
        for a in agents_in_action:
            action_queues[a] = (
                [None] * (max_len - len(action_queues[a])) + action_queues[a]
            )

        # Step through the queue
        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {}
            for a in agents_in_action:
                if action_queues[a]:
                    act = action_queues[a].pop(0)
                    if act is not None:
                        step_actions[a] = act

            obs, rewards, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1

            if any(terms.values()) or any(truncs.values()):
                done = True
                break

    return total_env_steps


# ===========================================================================
# Part 2 — Modified Policy Iteration
# ===========================================================================

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------
# A state is a tuple:
#   (agent0_pos, agent0_dir, agent1_pos, agent1_dir,
#    box0_pos,   box1_pos,   heavy_pos)
#
# where positions are (col, row) tuples and directions are 0-3.
#
# Feel free to simplify (e.g. drop agent directions if you argue they are
# irrelevant) as long as you justify it in your live demo.

def get_state(env) -> tuple:
    """Extract the current state tuple from a live environment."""
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    a0_dir = env.agent_dirs[agents[0]]
    a1_pos = env.agent_positions[agents[1]]
    a1_dir = env.agent_dirs[agents[1]]

    # Collect box positions by scanning the grid
    small_boxes = []
    heavy_boxes = []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_boxes.append((x, y))
                else:
                    small_boxes.append((x, y))

    # Sort for a canonical order
    small_boxes.sort()
    heavy_boxes.sort()

    box0_pos   = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos   = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos  = heavy_boxes[0] if heavy_boxes else None

    return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)


def build_transition_model(env):
    """
    Build the full MDP transition model analytically via BFS over reachable states.

    State = (a0_pos, a1_pos, box0_pos, box1_pos, heavy_pos)
        - positions are (col, row) tuples
        - agent directions are dropped (rotation is free/deterministic)

    Actions per agent = 4 directional moves (RIGHT=0, DOWN=1, LEFT=2, UP=3)
        - mapped to MiniGrid DIR_TO_VEC convention
        - joint_action = (a0_action, a1_action), 16 combinations total

    Returns
    -------
    transitions : dict
        transitions[state][joint_action] = [(prob, next_state, reward), ...]
    """
    from minigrid.core.constants import DIR_TO_VEC
    from collections import deque

    env.reset()

    width, height = env.width, env.height
    p_move  = env.move_success_prob          # 0.8
    p_push  = env.push_success_prob          # 0.8
    p_drift = (1.0 - p_move) / 2.0          # 0.1

    # ── Read static grid layout ───────────────────────────────────────────────
    walls      = set()
    goal_cells = set()

    for y in range(height):
        for x in range(width):
            cell = env.core_env.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls.add((x, y))
            elif cell.type == "goal":
                goal_cells.add((x, y))

    # ── Read initial object positions ─────────────────────────────────────────
    agents = env.possible_agents
    a0_init = env.agent_positions[agents[0]]
    a1_init = env.agent_positions[agents[1]]

    small_boxes, heavy_boxes = [], []
    for y in range(height):
        for x in range(width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_boxes.append((x, y))
                else:
                    small_boxes.append((x, y))

    small_boxes.sort()
    heavy_boxes.sort()
    box0_init  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_init  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_init = heavy_boxes[0] if heavy_boxes else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def in_bounds(pos):
        return 0 <= pos[0] < width and 0 <= pos[1] < height

    def is_terminal(state):
        """Episode ends when ALL boxes are on goal cells."""
        _, _, b0, b1, hb = state
        return b0 in goal_cells and b1 in goal_cells and hb in goal_cells

    def box_at(pos, state):
        """Return which box is at pos, or None."""
        _, _, b0, b1, hb = state
        if pos == b0:  return "box0"
        if pos == b1:  return "box1"
        if pos == hb:  return "heavy"
        return None

    def passable_for_agent(pos, state, other_agent_pos):
        """Can an agent step into pos (ignoring the moving agent itself)?"""
        if not in_bounds(pos) or pos in walls:
            return False
        if box_at(pos, state) is not None:   # any box blocks
            return False
        if pos == other_agent_pos:
            return False
        return True

    def agent_outcomes(agent_pos, direction, state, other_pos):
        """
        Enumerate (prob, new_agent_pos, new_b0, new_b1, new_hb) for one agent.

        Rules
        -----
        - MOVE (target is empty/goal): stochastic 0.8/0.1/0.1
        - PUSH-SMALL (target has small box, cell behind is free): stochastic p_push / (1-p_push)
        - PUSH-HEAVY (single agent): no-op (heavy requires two agents)
        - anything else (wall, out-of-bounds, blocked push): no-op (prob 1.0)
        """
        _, _, b0, b1, hb = state
        vec    = DIR_TO_VEC[direction]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])

        # ── wall / out-of-bounds ──────────────────────────────────────────────
        if not in_bounds(target) or target in walls:
            return [(1.0, agent_pos, b0, b1, hb)]

        # ── other agent blocks ────────────────────────────────────────────────
        if target == other_pos:
            return [(1.0, agent_pos, b0, b1, hb)]

        box_kind = box_at(target, state)

        # ── single agent cannot push heavy box ───────────────────────────────
        if box_kind == "heavy":
            return [(1.0, agent_pos, b0, b1, hb)]

        # ── small box push ────────────────────────────────────────────────────
        if box_kind in ("box0", "box1"):
            box_dest = (target[0] + vec[0], target[1] + vec[1])
            can_push = (
                in_bounds(box_dest)
                and box_dest not in walls
                and box_at(box_dest, state) is None
                and box_dest != other_pos
            )
            if not can_push:
                return [(1.0, agent_pos, b0, b1, hb)]

            new_b0_s = box_dest if box_kind == "box0" else b0
            new_b1_s = box_dest if box_kind == "box1" else b1
            return [
                (p_push,       target,    new_b0_s, new_b1_s, hb),
                (1.0 - p_push, agent_pos, b0,       b1,       hb),
            ]

        # ── empty / goal cell: stochastic move ───────────────────────────────
        left_vec  = DIR_TO_VEC[(direction - 1) % 4]
        right_vec = DIR_TO_VEC[(direction + 1) % 4]
        left_pos  = (agent_pos[0] + left_vec[0],  agent_pos[1] + left_vec[1])
        right_pos = (agent_pos[0] + right_vec[0], agent_pos[1] + right_vec[1])

        actual_left  = left_pos  if passable_for_agent(left_pos,  state, other_pos) else agent_pos
        actual_right = right_pos if passable_for_agent(right_pos, state, other_pos) else agent_pos

        # Merge duplicate destinations (e.g. both drifts blocked → agent stays)
        pos_prob = {}
        pos_prob[target]       = pos_prob.get(target,       0.0) + p_move
        pos_prob[actual_left]  = pos_prob.get(actual_left,  0.0) + p_drift
        pos_prob[actual_right] = pos_prob.get(actual_right, 0.0) + p_drift

        return [(p, pos, b0, b1, hb) for pos, p in pos_prob.items()]

    # ── BFS over reachable states ─────────────────────────────────────────────
    DIRS = [0, 1, 2, 3]   # RIGHT, DOWN, LEFT, UP

    initial_state = (a0_init, a1_init, box0_init, box1_init, heavy_init)
    transitions   = {}
    visited       = {initial_state}
    queue         = deque([initial_state])

    while queue:
        state = queue.popleft()
        transitions[state] = {}

        if is_terminal(state):
            continue   # absorbing — no outgoing transitions needed

        a0p, a1p, b0, b1, hb = state

        for a0_act in DIRS:
            for a1_act in DIRS:
                joint = (a0_act, a1_act)

                vec0 = DIR_TO_VEC[a0_act]
                vec1 = DIR_TO_VEC[a1_act]
                t0   = (a0p[0] + vec0[0], a0p[1] + vec0[1])
                t1   = (a1p[0] + vec1[0], a1p[1] + vec1[1])

                # ── Heavy-box push: both agents same cell, same dir, target heavy ──
                if (hb is not None
                        and t0 == hb and t1 == hb
                        and a0_act == a1_act
                        and a0p == a1p):

                    heavy_dest = (hb[0] + vec0[0], hb[1] + vec0[1])
                    can_push   = (
                        in_bounds(heavy_dest)
                        and heavy_dest not in walls
                        and heavy_dest not in (b0, b1)
                    )
                    if can_push:
                        # Both agents move to where heavy box was
                        s_succ = (hb, hb, b0, b1, heavy_dest)
                        r_succ = 1.0 if is_terminal(s_succ) else 0.0
                        outcomes = [
                            (p_push,       s_succ, r_succ),
                            (1.0 - p_push, state,  0.0),
                        ]
                        transitions[state][joint] = outcomes
                        for _, ns, _ in outcomes:
                            if ns not in visited:
                                visited.add(ns)
                                queue.append(ns)
                        continue

                # ── Independent agent outcomes ────────────────────────────────
                a0_outs = agent_outcomes(a0p, a0_act, state, a1p)
                a1_outs = agent_outcomes(a1p, a1_act, state, a0p)

                # ── Combine: multiply probabilities, handle conflicts ──────────
                combined = {}
                for (p0, na0, nb0_0, nb1_0, nhb_0) in a0_outs:
                    for (p1, na1, nb0_1, nb1_1, nhb_1) in a1_outs:

                        # Conflict: both agents try to move to the same cell → both stay
                        final_a0, final_a1 = na0, na1
                        if na0 == na1 and na0 != a0p:
                            final_a0, final_a1 = a0p, a1p

                        # Merge box changes (each agent affects at most one box)
                        final_b0 = nb0_0 if nb0_0 != b0 else nb0_1
                        final_b1 = nb1_0 if nb1_0 != b1 else nb1_1
                        final_hb = nhb_0 if nhb_0 != hb else nhb_1

                        ns   = (final_a0, final_a1, final_b0, final_b1, final_hb)
                        r    = 1.0 if is_terminal(ns) else 0.0
                        prob = p0 * p1

                        if ns in combined:
                            combined[ns] = (combined[ns][0] + prob, r)
                        else:
                            combined[ns] = (prob, r)

                outcomes = [(p, ns, r) for ns, (p, r) in combined.items()]
                transitions[state][joint] = outcomes

                for _, ns, _ in outcomes:
                    if ns not in visited:
                        visited.add(ns)
                        queue.append(ns)

    return transitions


def modified_policy_iteration(
    env,
    gamma: float = 0.95,
    k: int = 10,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
):
    """
    TODO — Modified Policy Iteration.

    Parameters
    ----------
    env   : StochasticMultiAgentBoxPushEnv (used only to build the model)
    gamma : discount factor
    k     : number of partial policy-evaluation sweeps per iteration
    theta : convergence threshold for value change
    max_outer_iters : safety cap on outer iterations

    Returns
    -------
    policy : dict  state -> joint_action
    V      : dict  state -> float
    """
    raise NotImplementedError("TODO: implement modified_policy_iteration")


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

    # ── Part 1: Online Planning ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Online Planning (classical planner on stochastic env)")
    print("=" * 60)

    # Wrap run_online_planning as a policy function for the evaluator
    def online_planning_policy(env, obs):
        """
        This wrapper runs one COMPLETE episode internally and is only a shim
        for the evaluator.  evaluate_policy will reset the env before each
        call, so we hand control back immediately with a do-nothing action
        after the first step — the real logic is inside run_online_planning.

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
            print(f"  run {i+1}/100 — steps so far: {steps}")

    mean_ol, std_ol = float(np.mean(online_steps)), float(np.std(online_steps))
    print(f"\nOnline Planning  →  mean = {mean_ol:.2f}  std = {std_ol:.2f}\n")

    # ── Part 2: Modified Policy Iteration ───────────────────────────────────
    print("=" * 60)
    print("Part 2 — Modified Policy Iteration")
    print("=" * 60)

    env_mpi = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=500)
    policy, V = modified_policy_iteration(env_mpi)

    def mpi_policy_fn(env, obs):
        """Convert current env state to a joint action using the MPI policy."""
        state = get_state(env)
        joint_action = policy[state]
        # joint_action is a tuple (action_agent0, action_agent1)
        agents = env.possible_agents
        return {agents[0]: joint_action[0], agents[1]: joint_action[1]}

    mean_mpi, std_mpi = evaluate_policy(mpi_policy_fn, env_mpi, n_runs=100)
    print(f"\nMPI              →  mean = {mean_mpi:.2f}  std = {std_mpi:.2f}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12}")
    print("-" * 50)
    print(f"{'Online Planning':<25} {mean_ol:>12.2f} {std_ol:>12.2f}")
    print(f"{'MPI':<25} {mean_mpi:>12.2f} {std_mpi:>12.2f}")
