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


def get_mpi_state(env) -> tuple:
    """
    Extract normalized state for MPI policy lookup.
    State = (a_min, a_max, b_min, b_max, heavy_pos)
        - directions dropped (rotation is free, doesn't affect value)
        - agents and boxes sorted (canonical form, reduces state space ~4x)
    """
    agents = env.possible_agents
    a0_pos = env.agent_positions[agents[0]]
    a1_pos = env.agent_positions[agents[1]]

    small_boxes, heavy_boxes = [], []
    for y in range(env.height):
        for x in range(env.width):
            cell = env.core_env.grid.get(x, y)
            if cell is not None and cell.type == "box":
                if getattr(cell, "box_size", "") == "heavy":
                    heavy_boxes.append((x, y))
                else:
                    small_boxes.append((x, y))

    small_boxes.sort()
    heavy_boxes.sort()

    agents_sorted = tuple(sorted([a0_pos, a1_pos]))
    heavy_pos     = heavy_boxes[0] if heavy_boxes else None

    return (agents_sorted[0], agents_sorted[1],
            small_boxes[0],   small_boxes[1],   heavy_pos)


def build_transition_model(env):
    """
    Build the full MDP transition model analytically via BFS over reachable states.

    State = (a_min, a_max, b_min, b_max, heavy_pos)
        Agents and small boxes are stored sorted (canonical form) so
        (a1,a2) and (a2,a1) map to the same state — reduces state space ~4x.

    Joint action = (dir1, dir2)  dir in {0=RIGHT, 1=DOWN, 2=LEFT, 3=UP}
        16 combinations total (4 per agent), actions are parallel.

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
    small_boxes.sort()
    heavy_boxes.sort()

    box0_init  = small_boxes[0] if len(small_boxes) > 0 else None
    box1_init  = small_boxes[1] if len(small_boxes) > 1 else None
    heavy_init = heavy_boxes[0] if heavy_boxes else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_agent_loc(a1, a2):
        return tuple(sorted([a1, a2]))

    def set_box_loc(b1, b2):
        return tuple(sorted([b1, b2]))

    def normalize(a1, a2, b1, b2, hb):
        ag = set_agent_loc(a1, a2)
        bx = set_box_loc(b1, b2)
        return (ag[0], ag[1], bx[0], bx[1], hb)

    def is_invalid(pos):
        return (pos in walls or
                pos[0] < 0 or pos[0] >= env.width or
                pos[1] < 0 or pos[1] >= env.height)

    def is_terminal(state):
        _, _, b0, b1, hb = state
        return b0 in goal_cells and b1 in goal_cells and hb in goal_cells

    def box_on_vector(agent_pos, direction, state):
        _, _, b0, b1, hb = state
        vec    = DIR_TO_VEC[direction]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])
        if target == b0:  return "box0"
        if target == b1:  return "box1"
        if target == hb:  return "heavy"
        return None

    def single_agent_outcomes(agent_pos, direction, state):
        """
        Returns [(prob, new_agent_pos, new_b0, new_b1, new_hb)].

        Cases
        -----
        wall / out-of-bounds → no-op (1.0, same)
        heavy in front       → no-op (single agent can't push heavy)
        small box in front   → push: (p_push, success), (1-p_push, fail)
        empty / goal         → stochastic move: (0.8, intended), (0.1, left), (0.1, right)
        """
        _, _, b0, b1, hb = state
        vec    = DIR_TO_VEC[direction]
        target = (agent_pos[0] + vec[0], agent_pos[1] + vec[1])

        if is_invalid(target):
            return [(1.0, agent_pos, b0, b1, hb)]

        box = box_on_vector(agent_pos, direction, state)

        # single agent cannot push heavy box
        if box == "heavy":
            return [(1.0, agent_pos, b0, b1, hb)]

        # small box push
        if box in ("box0", "box1"):
            push_dest = (target[0] + vec[0], target[1] + vec[1])
            if is_invalid(push_dest) or push_dest in (b0, b1, hb):
                return [(1.0, agent_pos, b0, b1, hb)]
            new_b0 = push_dest if box == "box0" else b0
            new_b1 = push_dest if box == "box1" else b1
            return [
                (p_push,       target,    new_b0, new_b1, hb),
                (1.0 - p_push, agent_pos, b0,     b1,     hb),
            ]

        # empty / goal: stochastic move (agents don't block each other)
        l_vec = DIR_TO_VEC[(direction - 1) % 4]
        r_vec = DIR_TO_VEC[(direction + 1) % 4]
        l_pos = (agent_pos[0] + l_vec[0], agent_pos[1] + l_vec[1])
        r_pos = (agent_pos[0] + r_vec[0], agent_pos[1] + r_vec[1])

        def drift_ok(pos):
            return not is_invalid(pos) and pos not in (b0, b1, hb)

        actual_l = l_pos if drift_ok(l_pos) else agent_pos
        actual_r = r_pos if drift_ok(r_pos) else agent_pos

        pos_prob = {}
        for pos, p in [(target, p_move), (actual_l, p_drift), (actual_r, p_drift)]:
            pos_prob[pos] = pos_prob.get(pos, 0.0) + p
        return [(p, pos, b0, b1, hb) for pos, p in pos_prob.items()]

    def combine(outcomes_a1, outcomes_a2, state):
        """
        Cross-product of both agents' outcomes.
        Multiply probs, extract only what changed per agent, normalize.
        Sum probs for duplicate next_states.
        """
        _, _, b0, b1, hb = state
        combined = {}
        for (p1, na1, nb0_1, nb1_1, nhb_1) in outcomes_a1:
            for (p2, na2, nb0_2, nb1_2, nhb_2) in outcomes_a2:
                prob     = p1 * p2
                final_b0 = nb0_1 if nb0_1 != b0 else nb0_2
                final_b1 = nb1_1 if nb1_1 != b1 else nb1_2
                final_hb = nhb_1 if nhb_1 != hb else nhb_2
                ns       = normalize(na1, na2, final_b0, final_b1, final_hb)
                combined[ns] = combined.get(ns, 0.0) + prob
        return [(p, ns, 1.0 if is_terminal(ns) else 0.0)
                for ns, p in combined.items()]

    # ── BFS over reachable states ─────────────────────────────────────────────
    DIRS          = [0, 1, 2, 3]
    initial_state = normalize(a0_init, a1_init, box0_init, box1_init, heavy_init)
    transitions   = {}
    visited       = {initial_state}
    queue         = deque([initial_state])

    while queue:
        state = queue.popleft()
        transitions[state] = {}

        if is_terminal(state):
            continue                    # absorbing — no outgoing transitions

        a0p, a1p, b0, b1, hb = state

        for action1 in DIRS:
            for action2 in DIRS:
                joint = (action1, action2)
                vec   = DIR_TO_VEC[action1]

                # ── heavy box joint push ──────────────────────────────────────
                if (a0p == a1p and
                        action1 == action2 and
                        box_on_vector(a0p, action1, state) == "heavy"):
                    push_dest  = (hb[0] + vec[0], hb[1] + vec[1])
                    push_valid = not is_invalid(push_dest) and push_dest not in (b0, b1)
                    if push_valid:
                        s_succ = normalize(hb, hb, b0, b1, push_dest)
                        r_succ = 1.0 if is_terminal(s_succ) else 0.0
                        transitions[state][joint] = [
                            (p_push,       s_succ, r_succ),
                            (1.0 - p_push, state,  0.0),
                        ]
                    else:
                        transitions[state][joint] = [(1.0, state, 0.0)]

                    for _, ns, _ in transitions[state][joint]:
                        if ns not in visited:
                            visited.add(ns)
                            queue.append(ns)
                    continue

                # ── general case: parallel independent actions ────────────────
                outcomes_a1 = single_agent_outcomes(a0p, action1, state)
                outcomes_a2 = single_agent_outcomes(a1p, action2, state)
                transitions[state][joint] = combine(outcomes_a1, outcomes_a2, state)

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
        state = get_mpi_state(env)   # normalized: no dirs, agents/boxes sorted
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
