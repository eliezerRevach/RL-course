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
# Full live tuple from ``get_state`` (includes facing directions):
#   (agent0_pos, agent0_dir, agent1_pos, agent1_dir, box0, box1, heavy)
#
# MPI uses the same full tuple so forward/rotation dynamics are represented
# exactly:
#   (agent0_pos, agent0_dir, agent1_pos, agent1_dir, box0, box1, heavy)
#
# Agent order follows ``env.possible_agents``; boxes follow row-major scan.

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

    box0_pos   = small_boxes[0] if len(small_boxes) > 0 else None
    box1_pos   = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_pos  = heavy_boxes[0] if heavy_boxes else None

    return (a0_pos, a0_dir, a1_pos, a1_dir, box0_pos, box1_pos, heavy_pos)


def get_mpi_state(env) -> tuple:
    """MPI policy key (same tuple as ``get_state``)."""
    return get_state(env)


def build_transition_model(env):
    """
    Build the full MDP transition model analytically via BFS over reachable states.

    State = (agent0_pos, agent0_dir, agent1_pos, agent1_dir, box0, box1, heavy)
        Matches ``get_state`` / ``get_mpi_state``.

    Joint action = (act0, act1), act in {0=LEFT, 1=RIGHT, 2=FORWARD}
        9 combinations total (3 per agent), actions are parallel.

    All data comes from env — NOT from PDDL:
        env.move_success_prob  → p_move (0.8)
        env.push_success_prob  → p_push (0.8)
        env.core_env.grid      → walls, goals
        env.agent_positions    → initial agent positions
        grid scan              → initial box positions

    Returns
    -------
    transitions : dict
        transitions[state][joint_action] = [(prob, next_state, reward), ...]
        reward = 1.0 when next_state is terminal, 0.0 otherwise
    """
    from minigrid.core.constants import DIR_TO_VEC
    from collections import deque

    # ── Read everything from env ──────────────────────────────────────────────
    env.reset()

    p_move  = env.move_success_prob
    p_push  = env.push_success_prob
    p_drift = (1.0 - p_move) / 2.0

    walls, goal_cells = set(), set()
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls.add((x, y))
            elif cell.type == "goal":
                goal_cells.add((x, y))

    agents  = env.possible_agents
    a0_init = env.agent_positions[agents[0]]
    a1_init = env.agent_positions[agents[1]]

    small_boxes, heavy_boxes = [], []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                (heavy_boxes if getattr(cell, "box_size", "") == "heavy"
                 else small_boxes).append((x, y))

    box0_init  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_init  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_init = heavy_boxes[0] if heavy_boxes else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_invalid(pos):
        return (pos in walls or
                pos[0] < 0 or pos[0] >= env.width or
                pos[1] < 0 or pos[1] >= env.height)

    def is_terminal(state):
        b0, b1, hb = state[4], state[5], state[6]
        return b0 in goal_cells and b1 in goal_cells and hb in goal_cells

    def box_on_vector(agent_pos, direction, b0, b1, hb):
        vec    = DIR_TO_VEC[direction]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])
        if target == b0:  return "box0"
        if target == b1:  return "box1"
        if target == hb:  return "heavy"
        return None

    def single_forward_outcomes(agent_pos, direction, b0, b1, hb, blocked_agents):
        """Forward-only outcomes for one agent under current occupancy."""
        vec    = DIR_TO_VEC[direction]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])

        def blocked(pos):
            return pos in blocked_agents or pos in (b0, b1, hb)

        if is_invalid(target):
            return [(1.0, agent_pos, b0, b1, hb)]

        box = box_on_vector(agent_pos, direction, b0, b1, hb)
        if box == "heavy":
            return [(1.0, agent_pos, b0, b1, hb)]

        if box in ("box0", "box1"):
            push_dest = (target[0] + vec[0], target[1] + vec[1])
            if is_invalid(push_dest) or blocked(push_dest):
                return [(1.0, agent_pos, b0, b1, hb)]
            new_b0 = push_dest if box == "box0" else b0
            new_b1 = push_dest if box == "box1" else b1
            return [
                (p_push,       target,    new_b0, new_b1, hb),
                (1.0 - p_push, agent_pos, b0,     b1,     hb),
            ]

        if blocked(target):
            return [(1.0, agent_pos, b0, b1, hb)]

        l_vec = DIR_TO_VEC[(direction - 1) % 4]
        r_vec = DIR_TO_VEC[(direction + 1) % 4]
        l_pos = (agent_pos[0] + l_vec[0], agent_pos[1] + l_vec[1])
        r_pos = (agent_pos[0] + r_vec[0], agent_pos[1] + r_vec[1])

        def drift_ok(pos):
            return not is_invalid(pos) and not blocked(pos)

        actual_l = l_pos if drift_ok(l_pos) else agent_pos
        actual_r = r_pos if drift_ok(r_pos) else agent_pos

        pos_prob = {}
        for pos, p in [(target, p_move), (actual_l, p_drift), (actual_r, p_drift)]:
            pos_prob[pos] = pos_prob.get(pos, 0.0) + p
        return [(p, pos, b0, b1, hb) for pos, p in pos_prob.items()]

    def apply_joint_action(state, joint_action):
        """
        Mirrors env.step phases:
          1) rotations + forward intents
          2) heavy joint push (stochastic)
          3) remaining forwards in fixed agent order
        """
        a0p, d0, a1p, d1, b0, b1, hb = state
        act0, act1 = joint_action

        if act0 == 0:
            d0 = (d0 - 1) % 4
        elif act0 == 1:
            d0 = (d0 + 1) % 4
        if act1 == 0:
            d1 = (d1 - 1) % 4
        elif act1 == 1:
            d1 = (d1 + 1) % 4

        intents = {}
        if act0 == 2:
            vec0 = DIR_TO_VEC[d0]
            intents[0] = {
                "agent_pos": a0p,
                "dir": d0,
                "target_pos": (a0p[0] + vec0[0], a0p[1] + vec0[1]),
                "vec": vec0,
            }
        if act1 == 2:
            vec1 = DIR_TO_VEC[d1]
            intents[1] = {
                "agent_pos": a1p,
                "dir": d1,
                "target_pos": (a1p[0] + vec1[0], a1p[1] + vec1[1]),
                "vec": vec1,
            }

        base = {
            "a0p": a0p, "d0": d0,
            "a1p": a1p, "d1": d1,
            "b0": b0, "b1": b1, "hb": hb,
        }
        branches = [(1.0, base, intents)]

        if 0 in intents and 1 in intents:
            i0, i1 = intents[0], intents[1]
            if (i0["target_pos"] == i1["target_pos"] == hb and
                    i0["agent_pos"] == i1["agent_pos"] and
                    i0["dir"] == i1["dir"]):
                push_dest = (hb[0] + i0["vec"][0], hb[1] + i0["vec"][1])
                push_valid = not is_invalid(push_dest) and push_dest not in (b0, b1)
                if push_valid:
                    succ_state = dict(base)
                    succ_state["a0p"] = hb
                    succ_state["a1p"] = hb
                    succ_state["hb"] = push_dest
                    rem_intents = {}
                    fail_state = dict(base)
                    branches = [
                        (p_push, succ_state, rem_intents),
                        (1.0 - p_push, fail_state, rem_intents),
                    ]
                else:
                    branches = [(1.0, base, {})]

        next_state_prob = {}
        for p_branch, branch_state, branch_intents in branches:
            b_states = [(1.0, branch_state)]
            for idx in [0, 1]:
                if idx not in branch_intents:
                    continue
                new_b_states = []
                for p_prev, st in b_states:
                    if idx == 0:
                        blocked_agents = {st["a1p"]}
                        out = single_forward_outcomes(
                            st["a0p"], st["d0"], st["b0"], st["b1"], st["hb"], blocked_agents
                        )
                        for p_out, na, nb0, nb1, nhb in out:
                            st2 = dict(st)
                            st2["a0p"] = na
                            st2["b0"], st2["b1"], st2["hb"] = nb0, nb1, nhb
                            new_b_states.append((p_prev * p_out, st2))
                    else:
                        blocked_agents = {st["a0p"]}
                        out = single_forward_outcomes(
                            st["a1p"], st["d1"], st["b0"], st["b1"], st["hb"], blocked_agents
                        )
                        for p_out, na, nb0, nb1, nhb in out:
                            st2 = dict(st)
                            st2["a1p"] = na
                            st2["b0"], st2["b1"], st2["hb"] = nb0, nb1, nhb
                            new_b_states.append((p_prev * p_out, st2))
                b_states = new_b_states

            for p_fin, st in b_states:
                ns = (st["a0p"], st["d0"], st["a1p"], st["d1"], st["b0"], st["b1"], st["hb"])
                next_state_prob[ns] = next_state_prob.get(ns, 0.0) + (p_branch * p_fin)

        return [(p, ns, 1.0 if is_terminal(ns) else 0.0) for ns, p in next_state_prob.items()]

    # ── BFS over reachable states ─────────────────────────────────────────────
    ACTIONS       = [0, 1, 2]
    d0_init       = env.agent_dirs[agents[0]]
    d1_init       = env.agent_dirs[agents[1]]
    initial_state = (a0_init, d0_init, a1_init, d1_init, box0_init, box1_init, heavy_init)
    transitions   = {}
    visited       = {initial_state}
    queue         = deque([initial_state])

    while queue:
        state = queue.popleft()
        transitions[state] = {}

        if is_terminal(state):
            continue                    # absorbing — no outgoing transitions

        for action1 in ACTIONS:
            for action2 in ACTIONS:
                joint = (action1, action2)
                transitions[state][joint] = apply_joint_action(state, joint)

                for _, ns, _ in transitions[state][joint]:
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
    # ── Build transition model from Part 1 ───────────────────────────────────
    print("Building transition model...")
    transitions = build_transition_model(env)
    all_states  = list(transitions.keys())
    print(f"  {len(all_states)} reachable states found.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_terminal(state):
        return transitions[state] == {}   # absorbing: no outgoing transitions

    def q_value(state, action, V):
        """Expected value of taking action in state under V."""
        return sum(prob * (r + gamma * V.get(s_, 0.0))
                   for prob, s_, r in transitions[state][action])

    # ── Init ─────────────────────────────────────────────────────────────────
    V      = {s: 0.0    for s in all_states}
    policy = {s: (0, 0) for s in all_states}   # arbitrary initial joint action

    # ── Main loop ─────────────────────────────────────────────────────────────
    for outer in range(max_outer_iters):

        # ── Step 1: Partial policy evaluation — k sweeps with early stop ─────
        for _ in range(k):
            max_delta = 0.0
            for s in all_states:
                if is_terminal(s):
                    V[s] = 0.0
                    continue
                old_v = V[s]
                V[s]  = q_value(s, policy[s], V)
                max_delta = max(max_delta, abs(V[s] - old_v))

            if max_delta < theta:
                break                          # values stable — stop early

        # ── Step 2: Policy improvement — greedy argmax ───────────────────────
        new_policy = {}
        for s in all_states:
            if is_terminal(s):
                new_policy[s] = (0, 0)         # arbitrary, never executed
                continue
            new_policy[s] = max(transitions[s].keys(),
                                key=lambda a: q_value(s, a, V))

        # ── Convergence check ─────────────────────────────────────────────────
        if new_policy == policy:
            print(f"  Converged after {outer + 1} outer iterations.")
            break

        policy = new_policy

    return policy, V


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
        state = get_mpi_state(env)   # full tuple: positions + headings + boxes
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
