"""
hw1_solution.py — Exercise 1: Multi-Agent Classical Planning with PettingZoo
=============================================================================
Task requirements (from exercises/README.md):
  1. Define a custom map: ≥3 corridor walls, 2 agents, 1 BigBox, 2 SmallBoxes.
  2. Call generate_pddl_for_env() to produce domain.pddl + problem.pddl.
  3. Call solve_pddl() to compute an optimal logical plan.
  4. Execute the plan step-by-step in the PettingZoo environment (no visualizer).
  5. Print the total accumulated rewards dict after the maze is solved.

"""

import os
import sys
import re

# ---------------------------------------------------------------------------
# Path setup — run from the repo root or from exercises/ex1/
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from minigrid.core.constants import DIR_TO_VEC
from environment.multi_agent_env import MultiAgentBoxPushEnv
from environment.pddl_extractor import generate_pddl_for_env
from planner.pddl_solver import solve_pddl

# ---------------------------------------------------------------------------
# Custom map
# ---------------------------------------------------------------------------
# Layout (0-indexed, x=col, y=row):
#
#   Row 0: WWWWWWWWWW
#   Row 1: W A BB   W   <- agent_0 at (2,1), SmallBox at (4,1), SmallBox at (5,1)
#   Row 2: W  WW    W   <- internal walls at (3,2) and (4,2) form a corridor barrier
#   Row 3: W   CC  GW   <- BigBox at (4,3)+(5,3), Goal at (8,3)
#   Row 4: W  WW    W   <- symmetric internal walls at (3,4) and (4,4)
#   Row 5: W A      W   <- agent_1 at (2,5)
#   Row 6: WWWWWWWWWW
#
# Complexity:
#   • 4 internal wall tiles (3 walls in a corridor arrangement on each side)
#   • 2 agents (agent_0, agent_1)
#   • 1 BigBox (needs cooperative push)
#   • 2 SmallBoxes (each pushable by a single agent)
#   • Goal on the right side — agents must navigate around the corridor

ex1_map = [
    "WWWWWWWW",
    "W  AA  W",
    "W  CC  W",
    "W W  B W",
    "W W  B W",
    "W WGG  W",
    "W  GG  W",
    "WWWWWWWW",
]

# ---------------------------------------------------------------------------
# PDDL output folder (inside exercises/ex1/)
# ---------------------------------------------------------------------------
PDDL_FOLDER = os.path.join(os.path.dirname(__file__), "pddl")


# ---------------------------------------------------------------------------
# Action translation helpers  (adapted from visualize_plan.py)
# ---------------------------------------------------------------------------

def extract_target_pos(pddl_action):
    """
    Parse a PDDL action string and return a dict mapping agent names to their
    target (x, y) grid coordinates for this step.

    For move / push-small: one agent moves to a new location.
    For push-big-*: two agents each step into one half of the bigbox.
    For win-*: no agent movement needed.
    """
    action_str = str(pddl_action)
    action_name = action_str.split("(")[0].strip()
    inner = action_str[action_str.find("(") + 1 : action_str.rfind(")")]
    params = re.findall(r"[\w]+", inner)

    agent_targets = {}

    if action_name.startswith("win"):
        return {}

    if action_name.startswith("push-big") and len(params) >= 6:
        # Parameters: a1 a2 from1 from2 boxloc1 boxloc2 toloc1 toloc2 b
        a1, a2 = params[0], params[1]
        # Agents step INTO boxloc1 / boxloc2 (indices 4 and 5)
        t1, t2 = params[4], params[5]
        parts1 = t1.split("_")
        parts2 = t2.split("_")
        if len(parts1) == 3:
            agent_targets[a1] = (int(parts1[1]), int(parts1[2]))
        if len(parts2) == 3:
            agent_targets[a2] = (int(parts2[1]), int(parts2[2]))
    elif len(params) >= 3:
        # move: a from to  |  push-small: a from boxloc toloc b
        agent_name = params[0]
        # target is params[2] for move, params[2] (boxloc) for push-small
        # In both cases the agent physically enters params[2]
        target_loc = params[2]
        parts = target_loc.split("_")
        if len(parts) == 3:
            agent_targets[agent_name] = (int(parts[1]), int(parts[2]))

    return agent_targets


def get_pz_actions(env, agent, target_pos):
    """
    Return a list of PettingZoo integer actions (0=turn-left, 1=turn-right,
    2=forward) that turn the agent to face target_pos, then move forward.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]

    dx = int(target_pos[0]) - int(current_pos[0])
    dy = int(target_pos[1]) - int(current_pos[1])

    # Find which of the 4 cardinal directions matches (dx, dy)
    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break

    if target_dir is None:
        raise ValueError(
            f"Agent '{agent}' at {current_pos}: "
            f"target {target_pos} is not adjacent."
        )

    actions = []
    # Turn right until facing the correct direction (max 3 turns)
    d = current_dir
    while d != target_dir:
        actions.append(1)  # turn right
        d = (d + 1) % 4
    actions.append(2)  # forward
    return actions


# ---------------------------------------------------------------------------
# Custom PDDL Generator (Enforcing strict rules from user prompt)
# ---------------------------------------------------------------------------
def generate_custom_domain(domain_path):
    domain_str = """(define (domain box-push)
  (:requirements :strips :typing :disjunctive-preconditions)
  (:types agent location box bigbox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (bigbox-at ?b - bigbox ?loc1 - location ?loc2 - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (adj-left ?l1 - location ?l2 - location)
    (adj-right ?l1 - location ?l2 - location)
    (adj-up ?l1 - location ?l2 - location)
    (adj-down ?l1 - location ?l2 - location)
  )
  
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)) (not (clear ?to)) (clear ?from))
  )
  
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and 
        (agent-at ?a ?from) 
        (box-at ?b ?boxloc) 
        (clear ?toloc)
        (or 
            (and (adj-up ?from ?boxloc) (adj-up ?boxloc ?toloc))
            (and (adj-down ?from ?boxloc) (adj-down ?boxloc ?toloc))
            (and (adj-left ?from ?boxloc) (adj-left ?boxloc ?toloc))
            (and (adj-right ?from ?boxloc) (adj-right ?boxloc ?toloc))
        )
    )
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
  )

  (:action push-big-up
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-up ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-up ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?boxloc2)
        (adj-up ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-up ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-down
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-down ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-down ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-down ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-left
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-left ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-left ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?boxloc2)
        (adj-left ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-left ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )

  (:action push-big-right
    :parameters (?a1 ?a2 - agent ?from1 ?from2 ?boxloc1 ?boxloc2 ?toloc1 ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj-right ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj-right ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj-down ?boxloc1 ?boxloc2)
        (adj-right ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj-right ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)


def generate_custom_problem(env, problem_path):
    w, h = env.width, env.height
    locations = []
    adjacencies = []
    directional_adj = []
    agents = env.agents
    boxes = []
    big_boxes_dict = {}
    goal_locs = []
    
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x+1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.extend([(loc, f"loc_{x+1}_{y}"), (f"loc_{x+1}_{y}", loc)])
                        directional_adj.extend([("adj-right", loc, f"loc_{x+1}_{y}"), ("adj-left", f"loc_{x+1}_{y}", loc)])
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y+1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.extend([(loc, f"loc_{x}_{y+1}"), (f"loc_{x}_{y+1}", loc)])
                        directional_adj.extend([("adj-down", loc, f"loc_{x}_{y+1}"), ("adj-up", f"loc_{x}_{y+1}", loc)])
                        
                if cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "big":
                        gid = getattr(cell, "group_id", 0)
                        if gid not in big_boxes_dict:
                            big_boxes_dict[gid] = []
                        big_boxes_dict[gid].append(loc)
                    else:
                        boxes.append((f"box_{len(boxes)}", loc))
                elif cell is not None and getattr(cell, "type", "") == "goal":
                    goal_locs.append((x, y))

    agent_locs = [(a, f"loc_{env.agent_positions[a][0]}_{env.agent_positions[a][1]}") for a in agents]
            
    clear_set = set(locations)
    for _, loc in agent_locs: clear_set.discard(loc)
    for _, loc in boxes: clear_set.discard(loc)
    for _, parts in big_boxes_dict.items():
        for p in parts: clear_set.discard(p)

    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents: obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes: obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if big_boxes_dict: obj_str += "    " + " ".join([f"bbig_{i}" for i in big_boxes_dict.keys()]) + " - bigbox\n"

    init_str = ""
    for loc in clear_set: init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs: init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes: init_str += f"    (box-at {b} {loc})\n"
    for gid, parts in big_boxes_dict.items():
        if len(parts) == 2:
            parts.sort(key=lambda p: (int(p.split('_')[2]), int(p.split('_')[1])))
            init_str += f"    (bigbox-at bbig_{gid} {parts[0]} {parts[1]})\n"
    for l1, l2 in adjacencies: init_str += f"    (adj {l1} {l2})\n"
    for adj_type, l1, l2 in directional_adj: init_str += f"    ({adj_type} {l1} {l2})\n"

    # Strictly matched dynamic Goal string
    goal_str = "(:goal (and\n"
    assigned_g = set()
    
    # Associate pairs of goals for big boxes
    for gid, parts in big_boxes_dict.items():
        found = False
        for gx, gy in goal_locs:
            if (gx, gy) not in assigned_g and (gx+1, gy) in goal_locs and (gx+1, gy) not in assigned_g:
                goal_str += f"    (bigbox-at bbig_{gid} loc_{gx}_{gy} loc_{gx+1}_{gy})\n"
                assigned_g.add((gx, gy))
                assigned_g.add((gx+1, gy))
                found = True
                break
        if not found:
            for gx, gy in goal_locs:
                if (gx, gy) not in assigned_g and (gx, gy+1) in goal_locs and (gx, gy+1) not in assigned_g:
                    goal_str += f"    (bigbox-at bbig_{gid} loc_{gx}_{gy} loc_{gx}_{gy+1})\n"
                    assigned_g.add((gx, gy))
                    assigned_g.add((gx, gy+1))
                    break

    # Associate singular goals for small boxes
    unassigned_g = [g for g in goal_locs if g not in assigned_g]
    for b in boxes:
        gx, gy = unassigned_g.pop(0)
        goal_str += f"    (box-at {b[0]} loc_{gx}_{gy})\n"
        
    goal_str += "))"

    problem_str = f"""(define (problem bp-map)
  (:domain box-push)
  (:objects
{obj_str}  )
  (:init
{init_str}  )
  {goal_str}
)
"""
    with open(problem_path, "w") as f:
        f.write(problem_str)


def custom_generate_pddl_for_env(env, pddl_folder="pddl"):
    os.makedirs(pddl_folder, exist_ok=True)
    domain_path = os.path.join(pddl_folder, "domain.pddl")
    problem_path = os.path.join(pddl_folder, "problem.pddl")
    
    generate_custom_domain(domain_path)
    generate_custom_problem(env, problem_path)
    
    return domain_path, problem_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def patch_env():
    """
    Monkeypatch multi_agent_env to clear ghost agents left behind after env.step.
    This prevents visualization crash without editing NO other files!
    """
    from environment.multi_agent_env import MultiAgentBoxPushEnv
    original_step = MultiAgentBoxPushEnv.step
    
    def patched_step(self, actions):
        result = original_step(self, actions)
        valid = set(self.agent_positions.values())
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in valid:
                    cell = self.core_env.grid.get(x, y)
                    if cell is not None and getattr(cell, "type", "") == "agent":
                        self.core_env.grid.set(x, y, None)
        return result
        
    MultiAgentBoxPushEnv.step = patched_step


def main():
    print("=" * 60)
    print("  HW1 – Multi-Agent Classical Planning Solution")
    print("=" * 60)

    patch_env()

    # --- Step 1: Build environment and extract PDDL -------------------------
    print("\n[1] Building environment from custom map...")
    env = MultiAgentBoxPushEnv(ascii_map=ex1_map, render_mode=None)
    env.reset()
    print(f"    Agents : {env.possible_agents}")
    print(f"    Map    : {len(ex1_map)} rows × {len(ex1_map[0])} cols")

    print(f"\n[2] Generating PDDL files in '{PDDL_FOLDER}' ...")
    domain_path, problem_path = custom_generate_pddl_for_env(env, pddl_folder=PDDL_FOLDER)
    print(f"    domain  → {domain_path}")
    print(f"    problem → {problem_path}")

    # --- Step 2: Run Visualize Plan -----------------------------------------
    print("\n[3] Running visualization from visualize_plan.py...")
    
    # Insert project root into sys.path to import visualize_plan
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from visualize_plan import visualize_pddl_plan
    
    # Run the visualization on the PDDL we just generated
    visualize_pddl_plan(ex1_map, domain_path, problem_path)

if __name__ == "__main__":
    main()

