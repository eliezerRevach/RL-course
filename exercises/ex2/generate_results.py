"""
Generate Assignment 2 evaluation results.

This script runs 100 episodes for each required method by default and writes
the required mean/std output to exercises/ex2/results.txt.

Usage from the repository root:
    python exercises/ex2/generate_results.py

For a short smoke test:
    python exercises/ex2/generate_results.py --runs 2 --skip-mpi
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import unified_planning.shortcuts as up

    up.get_environment().credits_stream = None
except Exception:
    pass

from environment.pddl_extractor import generate_pddl_for_env
from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from planner.pddl_solver import solve_pddl
from visualize_plan import extract_target_pos, get_required_actions

from exercises.ex2.solution_ex2 import (
    ASCII_MAP,
    get_mpi_state,
    modified_policy_iteration,
)


DEFAULT_RESULTS_PATH = Path(__file__).with_name("results.txt")
TMP_ROOT_BASE = Path(__file__).with_name(".tmp_pddl_results")


class Tee:
    """Write each line to stdout and to results.txt."""

    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        print(text, end="")
        self.file.write(text)
        self.file.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def close(self) -> None:
        self.file.close()


def quiet_call(fn, *args, **kwargs):
    """Run noisy course utilities without flooding results.txt."""

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def online_state_key(env: StochasticMultiAgentBoxPushEnv):
    """Hashable exact state key for caching deterministic first actions."""

    agents = tuple(
        (agent, env.agent_positions[agent], env.agent_dirs[agent])
        for agent in env.possible_agents
    )
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

    return agents, tuple(sorted(small_boxes)), tuple(sorted(heavy_boxes))


def run_online_planning_isolated(
    env: StochasticMultiAgentBoxPushEnv,
    pddl_dir: Path,
    max_replans: int,
    max_steps: int,
    plan_cache: dict,
    cache_stats: dict,
    revisit_threshold: int = 2,
    max_stuck_replans: int = 25,
) -> tuple[int, bool]:
    """
    Run one online-planning episode.

    Returns (steps, solved). If the episode does not terminate successfully,
    steps is returned as max_steps so failures are counted conservatively.
    """

    obs, _ = env.reset()
    del obs
    total_steps = 0
    visit_counts: dict = {}
    stuck_without_env_step = 0

    def bump_stuck() -> bool:
        """True if caller should abort the episode (too many no-op replans)."""
        nonlocal stuck_without_env_step
        stuck_without_env_step += 1
        if stuck_without_env_step >= max_stuck_replans:
            cache_stats["stuck_abort"] = cache_stats.get("stuck_abort", 0) + 1
            return True
        return False

    for _ in range(max_replans):
        if total_steps >= max_steps:
            return max_steps, False

        state_key = online_state_key(env)
        visit_counts[state_key] = visit_counts.get(state_key, 0) + 1
        force_replan = visit_counts[state_key] > revisit_threshold

        if force_replan:
            cache_stats["forced_replans"] = cache_stats.get("forced_replans", 0) + 1

        if state_key in plan_cache and not force_replan:
            cache_stats["hits"] += 1
            agent_targets = plan_cache[state_key]
        else:
            cache_stats["misses"] += 1
            domain_path, problem_path = generate_pddl_for_env(env, str(pddl_dir))
            plan = quiet_call(solve_pddl, domain_path, problem_path)

            if not plan or len(plan.actions) == 0:
                solved = env._all_boxes_on_goals()
                if solved:
                    plan_cache[state_key] = {}
                    return total_steps, True
                # Planner failed on a non-goal state; do not hard-fail immediately.
                cache_stats["planner_empty"] = cache_stats.get("planner_empty", 0) + 1
                plan_cache.pop(state_key, None)
                if bump_stuck():
                    return max_steps, False
                continue

            agent_targets = extract_target_pos(plan.actions[0])
            plan_cache[state_key] = agent_targets

        if not agent_targets:
            solved = env._all_boxes_on_goals()
            if solved:
                return total_steps, True
            cache_stats["empty_targets"] = cache_stats.get("empty_targets", 0) + 1
            plan_cache.pop(state_key, None)
            if bump_stuck():
                return max_steps, False
            continue

        agents_in_action = list(agent_targets.keys())
        action_queues = {}
        for agent in agents_in_action:
            try:
                action_queues[agent] = get_required_actions(
                    env, agent, agent_targets[agent]
                )
            except ValueError:
                # Invalid primitive conversion for this planned step; replan.
                cache_stats["invalid_targets"] = cache_stats.get("invalid_targets", 0) + 1
                plan_cache.pop(state_key, None)
                action_queues = {}
                break

        if not action_queues:
            if bump_stuck():
                return max_steps, False
            continue

        stuck_without_env_step = 0
        max_queue_len = max(len(queue) for queue in action_queues.values())
        for agent in agents_in_action:
            pad_len = max_queue_len - len(action_queues[agent])
            action_queues[agent] = [None] * pad_len + action_queues[agent]

        while any(action_queues.values()):
            if total_steps >= max_steps:
                return max_steps, False

            step_actions = {}
            for agent in agents_in_action:
                if action_queues[agent]:
                    action = action_queues[agent].pop(0)
                    if action is not None:
                        step_actions[agent] = action

            obs, rewards, terms, truncs, _ = env.step(step_actions)
            del obs, rewards
            total_steps += 1

            if any(terms.values()):
                return total_steps, True
            if any(truncs.values()):
                return max_steps, False

    return max_steps, False


def evaluate_online(
    runs: int,
    max_replans: int,
    max_steps: int,
    seed: int,
    log: Tee,
    max_stuck_replans: int = 25,
):
    steps = []
    solved = 0
    start = time.time()
    # Reuse cache across runs to reduce repeated planner calls during benchmark.
    plan_cache = {}
    cache_stats = {
        "hits": 0,
        "misses": 0,
        "forced_replans": 0,
        "planner_empty": 0,
        "empty_targets": 0,
        "invalid_targets": 0,
        "stuck_abort": 0,
    }
    tmp_root = Path(f"{TMP_ROOT_BASE}_{os.getpid()}")

    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    for run_idx in range(runs):
        np.random.seed(seed + run_idx)
        env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=max_steps)
        pddl_dir = tmp_root / f"online_run_{run_idx:03d}"
        pddl_dir.mkdir(parents=True, exist_ok=True)

        # Use the FAST RPG-based online planner from solution_ex2.py instead
        # of the PDDL/Fast-Downward path (which can hang for many minutes).
        from exercises.ex2.solution_ex2 import run_online_planning as rpg_online
        run_steps = rpg_online(env, max_replans=max_replans)
        run_solved = env._all_boxes_on_goals()
        steps.append(run_steps)
        solved += int(run_solved)

        if (run_idx + 1) % 10 == 0 or run_idx == 0 or run_idx + 1 == runs:
            log.line(
                f"  online run {run_idx + 1:3d}/{runs}: "
                f"steps={run_steps}, solved={run_solved}"
            )

    shutil.rmtree(tmp_root, ignore_errors=True)

    elapsed = time.time() - start
    log.line(
        f"  online planner cache: {cache_stats['hits']} hits, "
        f"{cache_stats['misses']} misses, "
        f"forced_replans={cache_stats.get('forced_replans', 0)}, "
        f"planner_empty={cache_stats.get('planner_empty', 0)}, "
        f"empty_targets={cache_stats.get('empty_targets', 0)}, "
        f"invalid_targets={cache_stats.get('invalid_targets', 0)}, "
        f"stuck_abort={cache_stats.get('stuck_abort', 0)}"
    )
    return float(np.mean(steps)), float(np.std(steps)), solved, elapsed


def evaluate_mpi(runs: int, max_steps: int, seed: int, k: int, log: Tee):
    start = time.time()
    build_env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=max_steps)
    policy, _ = modified_policy_iteration(build_env, k=k)
    build_elapsed = time.time() - start
    log.line(f"  MPI policy built in {build_elapsed:.1f}s")

    mpi_missing = {"count": 0}

    def mpi_policy_fn(env, obs):
        del obs
        state = get_mpi_state(env)
        agents = env.possible_agents
        if state not in policy:
            mpi_missing["count"] += 1
            out = {agents[0]: 2}
            if len(agents) >= 2:
                out[agents[1]] = 2
            return out
        joint_action = policy[state]
        # joint_action may be a tuple (2 agents) or an int (1 agent).
        if isinstance(joint_action, (tuple, list)):
            out = {agents[0]: joint_action[0]}
            if len(agents) >= 2 and len(joint_action) >= 2:
                out[agents[1]] = joint_action[1]
        else:
            out = {agents[0]: joint_action}
        return out

    np.random.seed(seed)
    eval_env = StochasticMultiAgentBoxPushEnv(ascii_map=ASCII_MAP, max_steps=max_steps)
    mean_steps, std_steps, solved = evaluate_policy_with_solved(
        mpi_policy_fn, eval_env, n_runs=runs, max_steps=max_steps
    )
    elapsed = time.time() - start
    log.line(
        f"  MPI eval: missing_state_fallback_steps={mpi_missing['count']} "
        f"(policy size={len(policy)} states)"
    )
    return mean_steps, std_steps, solved, elapsed


def evaluate_policy_with_solved(policy_fn, env, n_runs: int, max_steps: int):
    """Same shape as the skeleton evaluator, plus a solved episode count."""

    steps_per_run = []
    solved = 0

    for _ in range(n_runs):
        obs, _ = env.reset()
        steps = 0
        done = False

        while not done and steps < max_steps:
            actions = policy_fn(env, obs)
            obs, rewards, terms, truncs, _ = env.step(actions)
            del rewards
            steps += 1
            done = any(terms.values()) or any(truncs.values())

        solved += int(env._all_boxes_on_goals())
        steps_per_run.append(steps)

    return float(np.mean(steps_per_run)), float(np.std(steps_per_run)), solved


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-replans", type=int, default=300)
    parser.add_argument(
        "--max-stuck-replans",
        type=int,
        default=25,
        metavar="N",
        help=(
            "Abort an online episode after N consecutive replans that never call "
            "env.step (e.g. empty planner output). Prevents huge FD batches."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mpi-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument("--skip-mpi", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log = Tee(args.output)
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        log.line("=" * 60)
        log.line("Assignment 2 Results")
        log.line("=" * 60)
        log.line(f"started: {started}")
        log.line(f"runs per algorithm: {args.runs}")
        log.line(f"max_steps: {args.max_steps}")
        log.line(f"seed: {args.seed}")
        log.line(f"max_stuck_replans (online abort): {args.max_stuck_replans}")
        log.line("")

        summary = []

        if not args.skip_online:
            log.line("Part 1 - Online Planning")
            mean_ol, std_ol, solved_ol, elapsed_ol = evaluate_online(
                runs=args.runs,
                max_replans=args.max_replans,
                max_steps=args.max_steps,
                seed=args.seed,
                log=log,
                max_stuck_replans=args.max_stuck_replans,
            )
            summary.append(("Online Planning", mean_ol, std_ol, solved_ol, elapsed_ol))
            log.line(
                f"Online Planning  mean = {mean_ol:.2f}, "
                f"std = {std_ol:.2f}, solved = {solved_ol}/{args.runs}, "
                f"elapsed = {elapsed_ol:.1f}s"
            )
            log.line("")

        if not args.skip_mpi:
            log.line("Part 2 - Modified Policy Iteration")
            mean_mpi, std_mpi, solved_mpi, elapsed_mpi = evaluate_mpi(
                runs=args.runs,
                max_steps=args.max_steps,
                seed=args.seed + 100000,
                k=args.mpi_k,
                log=log,
            )
            summary.append(("MPI", mean_mpi, std_mpi, solved_mpi, elapsed_mpi))
            log.line(
                f"MPI              mean = {mean_mpi:.2f}, "
                f"std = {std_mpi:.2f}, solved = {solved_mpi}/{args.runs}, "
                f"elapsed = {elapsed_mpi:.1f}s"
            )
            log.line("")

        log.line("=" * 60)
        log.line("SUMMARY")
        log.line("=" * 60)
        log.line(f"{'Algorithm':<25} {'Mean steps':>12} {'Std steps':>12} {'Solved':>10}")
        log.line("-" * 65)
        for name, mean, std, solved, _elapsed in summary:
            log.line(f"{name:<25} {mean:>12.2f} {std:>12.2f} {solved:>4}/{args.runs:<5}")

    finally:
        log.close()


if __name__ == "__main__":
    main()
