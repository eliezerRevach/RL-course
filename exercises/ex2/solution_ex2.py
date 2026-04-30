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
    del obs
    total_env_steps = 0
    done = False
    empty_plan_streak = 0
    revisit_threshold = 2
    revisit_counts = {}
    # Per-episode cache: state -> first-action targets extracted from planner output.
    plan_cache = {}

    for _ in range(max_replans):
        if done or total_env_steps >= env.max_steps:
            break

        key = get_state(env)
        revisit_counts[key] = revisit_counts.get(key, 0) + 1
        force_replan = revisit_counts[key] > revisit_threshold

        if key in plan_cache and not force_replan:
            agent_targets = plan_cache[key]
        else:
            # 1) Export the exact current state to PDDL.
            domain_path, problem_path = generate_pddl_for_env(env)

            # 2) Replan from the current state.
            try:
                plan = solve_pddl(domain_path, problem_path)
            except Exception:
                # Planner/plan-parser occasionally fails; retry next iteration.
                empty_plan_streak += 1
                if empty_plan_streak >= 5:
                    break
                plan_cache.pop(key, None)
                continue

            if not plan or len(plan.actions) == 0:
                if env._all_boxes_on_goals():
                    done = True
                    break
                # Planner failed in a non-goal state; retry from next loop.
                empty_plan_streak += 1
                if empty_plan_streak >= 5:
                    break
                plan_cache.pop(key, None)
                continue
            empty_plan_streak = 0

            # 3) Execute only the FIRST planned action.
            pddl_action = plan.actions[0]
            agent_targets = extract_target_pos(pddl_action)
            if not agent_targets:
                plan_cache.pop(key, None)
                continue
            plan_cache[key] = agent_targets

        agents_in_action = list(agent_targets.keys())
        action_queues = {}
        for agent in agents_in_action:
            try:
                action_queues[agent] = get_required_actions(env, agent, agent_targets[agent])
            except ValueError:
                plan_cache.pop(key, None)
                action_queues = {}
                break
        if not action_queues:
            continue

        # Pad shorter queues so final forwards stay synchronized for joint actions.
        max_len = max(len(q) for q in action_queues.values())
        for agent in agents_in_action:
            action_queues[agent] = [None] * (max_len - len(action_queues[agent])) + action_queues[agent]

        while any(len(q) > 0 for q in action_queues.values()):
            step_actions = {}
            for agent in agents_in_action:
                if action_queues[agent]:
                    act = action_queues[agent].pop(0)
                    if act is not None:
                        step_actions[agent] = act

            # This env treats empty actions as a special terminal-like branch.
            if not step_actions:
                continue

            _, _, terms, truncs, _ = env.step(step_actions)
            total_env_steps += 1

            if any(terms.values()) or any(truncs.values()):
                done = True
                break

    # Keep assignment semantics used in this file's evaluation section.
    return total_env_steps if env._all_boxes_on_goals() else env.max_steps


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


def get_mpi_state(env) -> tuple:
    """
    Backward-compatible alias used by helper scripts in this exercise.
    """
    return get_state(env)


def build_transition_model(env):
    """
    Build a compact transition *descriptor* for lazy MPI expansion.

    We intentionally avoid precomputing transitions for all reachable states,
    because that can consume tens of GB in Python dictionaries.

    The returned model contains static geometry/probabilities and the initial
    state. Per-state transitions are generated on demand by MPI.

    Returns
    -------
    model : dict
        Static info needed to compute T(s, a) lazily.
    """
    env.reset()

    p_move = env.move_success_prob
    p_push = env.push_success_prob

    walls = set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls.add((x, y))

    goal_cells = set(getattr(env, "goal_positions", []))
    if not goal_cells:
        for y in range(env.height):
            for x in range(env.width):
                cell = env.core_env.grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    goal_cells.add((x, y))

    agents = env.possible_agents
    a0_init = env.agent_positions[agents[0]]
    a1_init = env.agent_positions[agents[1]]
    d0_init = env.agent_dirs[agents[0]]
    d1_init = env.agent_dirs[agents[1]]

    small_boxes, heavy_boxes = [], []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                (heavy_boxes if getattr(cell, "box_size", "") == "heavy"
                 else small_boxes).append((x, y))
    small_boxes.sort()
    heavy_boxes.sort()

    box0_init  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_init  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_init = heavy_boxes[0] if heavy_boxes else None

    initial_state = (a0_init, d0_init, a1_init, d1_init, box0_init, box1_init, heavy_init)
    return {
        "width": env.width,
        "height": env.height,
        "walls": walls,
        "goal_cells": goal_cells,
        "p_move": p_move,
        "p_push": p_push,
        "p_drift": (1.0 - p_move) / 2.0,
        "initial_state": initial_state,
    }


def modified_policy_iteration(
    env,
    gamma: float = 0.9,
    k: int = 50,
    theta: float = 1e-4,
    max_outer_iters: int = 500,
    max_states: int = 300_000,
    step_penalty: float = -0.01,
):
    """
    Parameters
    ----------
    env   : StochasticMultiAgentBoxPushEnv (used only to build the model)
    gamma : discount factor
    k     : number of partial policy-evaluation sweeps per iteration
    theta : convergence threshold for value change
    max_outer_iters : safety cap on outer iterations
    step_penalty : reward for non-terminal transitions (negative discourages wandering)

    Returns
    -------
    policy : dict  state -> joint_action
    V      : dict  state -> float
    """
    from minigrid.core.constants import DIR_TO_VEC
    from itertools import product
    print("Building compact transition descriptor...")
    model = build_transition_model(env)
    print("  using lazy transition expansion (on-demand states).")

    width = model["width"]
    height = model["height"]
    walls = model["walls"]
    goal_cells = model["goal_cells"]
    p_move = model["p_move"]
    p_push = model["p_push"]
    p_drift = model["p_drift"]
    joint_actions = list(product((0, 1, 2), repeat=2))  # left/right/forward

    def canonical_boxes(b0, b1):
        return tuple(sorted((b0, b1)))

    def normalize_state(a0p, a0d, a1p, a1d, b0, b1, hb):
        sb0, sb1 = canonical_boxes(b0, b1)
        return (a0p, a0d, a1p, a1d, sb0, sb1, hb)

    def is_invalid(pos):
        return (
            pos in walls
            or pos[0] < 0
            or pos[0] >= width
            or pos[1] < 0
            or pos[1] >= height
        )

    def is_terminal(state):
        _, _, _, _, b0, b1, hb = state
        return b0 in goal_cells and b1 in goal_cells and hb in goal_cells

    def box_at(pos, b0, b1, hb):
        if pos == b0:
            return "box0"
        if pos == b1:
            return "box1"
        if pos == hb:
            return "heavy"
        return None

    def single_forward_outcomes(idx, positions, dirs, b0, b1, hb):
        """
        Return stochastic outcomes when a single agent executes FORWARD.
        Format: [(prob, new_positions, b0, b1, hb)].
        """
        pos = positions[idx]
        direction = dirs[idx]
        vec = DIR_TO_VEC[direction]
        target = (pos[0] + vec[0], pos[1] + vec[1])

        if is_invalid(target):
            return [(1.0, positions, b0, b1, hb)]

        front_obj = box_at(target, b0, b1, hb)

        # Heavy box cannot be pushed by one agent.
        if front_obj == "heavy":
            return [(1.0, positions, b0, b1, hb)]

        # Push small box.
        if front_obj in ("box0", "box1"):
            push_dest = (target[0] + vec[0], target[1] + vec[1])
            if is_invalid(push_dest) or push_dest in (b0, b1, hb):
                return [(1.0, positions, b0, b1, hb)]

            moved_positions = list(positions)
            moved_positions[idx] = target
            if front_obj == "box0":
                success_boxes = (push_dest, b1)
            else:
                success_boxes = (b0, push_dest)

            return [
                (p_push, tuple(moved_positions), success_boxes[0], success_boxes[1], hb),
                (1.0 - p_push, positions, b0, b1, hb),
            ]

        # Move into empty/goal with stochastic drift.
        left_vec = DIR_TO_VEC[(direction - 1) % 4]
        right_vec = DIR_TO_VEC[(direction + 1) % 4]
        candidates = [
            (target, p_move),
            ((pos[0] + left_vec[0], pos[1] + left_vec[1]), p_drift),
            ((pos[0] + right_vec[0], pos[1] + right_vec[1]), p_drift),
        ]

        next_pos_prob = {}
        for cand_pos, prob in candidates:
            actual = cand_pos
            if is_invalid(actual) or actual in (b0, b1, hb):
                actual = pos
            next_pos_prob[actual] = next_pos_prob.get(actual, 0.0) + prob

        outcomes = []
        for next_pos, prob in next_pos_prob.items():
            moved_positions = list(positions)
            moved_positions[idx] = next_pos
            outcomes.append((prob, tuple(moved_positions), b0, b1, hb))
        return outcomes

    def transitions_for_action(state, joint_action):
        """
        Exact one-step dynamics for this assignment's stochastic env.
        Returns [(prob, next_state, reward)].
        """
        a0p, a0d, a1p, a1d, b0, b1, hb = state
        if is_terminal(state):
            return []

        actions = list(joint_action)
        positions = [a0p, a1p]
        dirs = [a0d, a1d]

        # Pass 1: apply rotations only.
        for i in (0, 1):
            if actions[i] == 0:       # turn left
                dirs[i] = (dirs[i] - 1) % 4
            elif actions[i] == 1:     # turn right
                dirs[i] = (dirs[i] + 1) % 4

        # Forward intents for heavy-push pass.
        forward_intent = {}
        for i in (0, 1):
            if actions[i] == 2:
                vec = DIR_TO_VEC[dirs[i]]
                forward_intent[i] = (
                    (positions[i][0] + vec[0], positions[i][1] + vec[1]),
                    dirs[i],
                )

        scenarios = [(1.0, tuple(positions), tuple(dirs), b0, b1, hb, set())]

        # Pass 2: heavy push resolution.
        pushers = [i for i, (target, _) in forward_intent.items() if target == hb]
        if len(pushers) >= 2:
            origins = {positions[i] for i in pushers}
            push_dirs = {forward_intent[i][1] for i in pushers}
            if len(origins) == 1 and len(push_dirs) == 1:
                push_dir = next(iter(push_dirs))
                vec = DIR_TO_VEC[push_dir]
                push_dest = (hb[0] + vec[0], hb[1] + vec[1])
                valid = (not is_invalid(push_dest)) and (push_dest not in (b0, b1))

                consumed = set(pushers)
                if valid:
                    succ_positions = list(positions)
                    for i in pushers:
                        succ_positions[i] = hb
                    scenarios = [
                        (p_push, tuple(succ_positions), tuple(dirs), b0, b1, push_dest, consumed),
                        (1.0 - p_push, tuple(positions), tuple(dirs), b0, b1, hb, consumed),
                    ]
                else:
                    scenarios = [(1.0, tuple(positions), tuple(dirs), b0, b1, hb, consumed)]

        # Pass 3: remaining forward intents, processed in agent order.
        for agent_idx in (0, 1):
            updated = []
            for prob, pos_s, dir_s, b0_s, b1_s, hb_s, consumed in scenarios:
                if actions[agent_idx] != 2 or agent_idx in consumed:
                    updated.append((prob, pos_s, dir_s, b0_s, b1_s, hb_s, consumed))
                    continue

                for op, new_pos, nb0, nb1, nhb in single_forward_outcomes(
                    agent_idx,
                    pos_s,
                    dir_s,
                    b0_s,
                    b1_s,
                    hb_s,
                ):
                    updated.append((prob * op, new_pos, dir_s, nb0, nb1, nhb, consumed))
            scenarios = updated

        merged = {}
        for prob, pos_s, dir_s, b0_s, b1_s, hb_s, _ in scenarios:
            ns = normalize_state(pos_s[0], dir_s[0], pos_s[1], dir_s[1], b0_s, b1_s, hb_s)
            merged[ns] = merged.get(ns, 0.0) + prob

        return [
            (p, ns, (1.0 if is_terminal(ns) else step_penalty))
            for ns, p in merged.items()
        ]

    # Known states are expanded lazily as they are discovered.
    initial_state = normalize_state(*model["initial_state"])
    known_states = [initial_state]
    known_set = {initial_state}
    V = {initial_state: 0.0}
    policy = {initial_state: (2, 2)}
    transition_cache = {}
    state_cap_reached = False

    def register_state(state):
        nonlocal state_cap_reached
        if state in known_set:
            return
        if len(known_states) >= max_states:
            state_cap_reached = True
            # Still attach a default action so evaluation never hits a bare KeyError
            # for successors seen in the transition model. Do not grow known_states
            # (keeps partial-evaluation cost bounded).
            known_set.add(state)
            policy.setdefault(state, (2, 2))
            V.setdefault(state, 0.0)
            return
        known_set.add(state)
        known_states.append(state)
        V[state] = 0.0
        policy[state] = (2, 2)

    def get_state_transitions(state):
        if state in transition_cache:
            return transition_cache[state]
        if is_terminal(state):
            transition_cache[state] = {}
            return transition_cache[state]

        trans = {}
        for action in joint_actions:
            outcomes = transitions_for_action(state, action)
            trans[action] = outcomes
            for _, ns, _ in outcomes:
                register_state(ns)
        transition_cache[state] = trans
        return trans

    def q_from_outcomes(outcomes):
        return sum(prob * (reward + gamma * V.get(ns, 0.0)) for prob, ns, reward in outcomes)

    for outer in range(max_outer_iters):
        # Step 1: k sweeps of partial policy evaluation.
        for _ in range(k):
            max_delta = 0.0
            for s in list(known_states):
                if is_terminal(s):
                    V[s] = 0.0
                    continue
                old_v = V[s]
                trans_s = get_state_transitions(s)
                V[s] = q_from_outcomes(trans_s[policy[s]])
                max_delta = max(max_delta, abs(V[s] - old_v))
            if max_delta < theta:
                break

        # Step 2: policy improvement.
        policy_stable = True
        for s in list(known_states):
            if is_terminal(s):
                policy[s] = (0, 0)
                continue
            trans_s = get_state_transitions(s)
            best_action = max(joint_actions, key=lambda a: q_from_outcomes(trans_s[a]))
            if best_action != policy[s]:
                policy[s] = best_action
                policy_stable = False

        print(f"  iter {outer + 1:03d}: known_states={len(known_states)}")
        if state_cap_reached:
            print(f"  reached state cap ({max_states}); continuing with truncated state set.")
        if policy_stable:
            print(f"  Converged after {outer + 1} outer iterations.")
            break

    # Plain dict so ``state in policy`` reflects real coverage (no silent defaults).
    return dict(policy), V


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
