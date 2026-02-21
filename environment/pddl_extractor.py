import os

def generate_domain(domain_path):
    domain_str = """(define (domain box-push)
  (:requirements :strips :typing)
  (:types agent location box bigbox)
  (:predicates
    (agent-at ?a - agent ?loc - location)
    (box-at ?b - box ?loc - location)
    (bigbox-at ?b - bigbox ?loc1 - location ?loc2 - location)
    (clear ?loc - location)
    (adj ?l1 - location ?l2 - location)
    (goal ?loc - location)
    (won )
  )
  
  (:action move
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (agent-at ?a ?from) (adj ?from ?to) (clear ?to))
    :effect (and (agent-at ?a ?to) (not (agent-at ?a ?from)) (not (clear ?to)) (clear ?from))
  )
  
  (:action push-small
    :parameters (?a - agent ?from - location ?boxloc - location ?toloc - location ?b - box)
    :precondition (and (agent-at ?a ?from) (adj ?from ?boxloc) (box-at ?b ?boxloc) (adj ?boxloc ?toloc) (clear ?toloc))
    :effect (and (agent-at ?a ?boxloc) (not (agent-at ?a ?from)) (clear ?from) (box-at ?b ?toloc) (not (box-at ?b ?boxloc)) (not (clear ?toloc)))
  )

  (:action push-big
    :parameters (?a1 - agent ?a2 - agent ?from1 - location ?from2 - location 
                 ?boxloc1 - location ?boxloc2 - location 
                 ?toloc1 - location ?toloc2 - location ?b - bigbox)
    :precondition (and 
        (agent-at ?a1 ?from1) (adj ?from1 ?boxloc1)
        (agent-at ?a2 ?from2) (adj ?from2 ?boxloc2)
        (bigbox-at ?b ?boxloc1 ?boxloc2)
        (adj ?boxloc1 ?toloc1) (clear ?toloc1)
        (adj ?boxloc2 ?toloc2) (clear ?toloc2)
    )
    :effect (and 
        (agent-at ?a1 ?boxloc1) (not (agent-at ?a1 ?from1)) (clear ?from1)
        (agent-at ?a2 ?boxloc2) (not (agent-at ?a2 ?from2)) (clear ?from2)
        (bigbox-at ?b ?toloc1 ?toloc2) (not (bigbox-at ?b ?boxloc1 ?boxloc2))
        (not (clear ?toloc1)) (not (clear ?toloc2))
    )
  )
  
  (:action win-small
    :parameters (?b - box ?loc - location)
    :precondition (and (box-at ?b ?loc) (goal ?loc))
    :effect (won)
  )

  (:action win-big-1
    :parameters (?b - bigbox ?loc1 - location ?loc2 - location)
    :precondition (and (bigbox-at ?b ?loc1 ?loc2) (goal ?loc1))
    :effect (won)
  )
  
  (:action win-big-2
    :parameters (?b - bigbox ?loc1 - location ?loc2 - location)
    :precondition (and (bigbox-at ?b ?loc1 ?loc2) (goal ?loc2))
    :effect (won)
  )
)
"""
    with open(domain_path, "w") as f:
        f.write(domain_str)

def generate_problem(env, problem_path):
    # Extract locations, agents, boxes from env grid
    w, h = env.width, env.height
    
    locations = []
    adjacencies = []
    clear_locs = []
    agents = env.agents
    boxes = []
    big_boxes_dict = {}
    goals = []
    
    # Analyze the static grid
    for y in range(h):
        for x in range(w):
            cell = env.core_env.grid.get(x, y)
            if cell is None or cell.type != 'wall':
                loc = f"loc_{x}_{y}"
                locations.append(loc)
                
                # Check adjacencies
                if x < w - 1:
                    r_cell = env.core_env.grid.get(x+1, y)
                    if r_cell is None or r_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x+1}_{y}"))
                        adjacencies.append((f"loc_{x+1}_{y}", loc))
                if y < h - 1:
                    d_cell = env.core_env.grid.get(x, y+1)
                    if d_cell is None or d_cell.type != 'wall':
                        adjacencies.append((loc, f"loc_{x}_{y+1}"))
                        adjacencies.append((f"loc_{x}_{y+1}", loc))
                        
                # Determine what is here
                if cell is not None and cell.type == "goal":
                    goals.append(loc)
                elif cell is not None and cell.type == "box":
                    if getattr(cell, "box_size", "") == "big":
                        gid = getattr(cell, "group_id", 0)
                        if gid not in big_boxes_dict:
                            big_boxes_dict[gid] = []
                        big_boxes_dict[gid].append(loc)
                    else:
                        boxes.append((f"box_{len(boxes)}", loc))

    agent_locs = []
    for a in agents:
        px, py = env.agent_positions[a]
        agent_locs.append((a, f"loc_{px}_{py}"))
            
    clear_set = set(locations)
    for _, loc in agent_locs:
        clear_set.discard(loc)
    for _, loc in boxes:
        clear_set.discard(loc)
    for _, parts in big_boxes_dict.items():
        for p in parts:
            clear_set.discard(p)

    obj_str = "    " + " ".join(locations) + " - location\n"
    if agents:
        obj_str += "    " + " ".join(agents) + " - agent\n"
    if boxes:
        obj_str += "    " + " ".join([b[0] for b in boxes]) + " - box\n"
    if big_boxes_dict:
        obj_str += "    " + " ".join([f"bbig_{i}" for i in big_boxes_dict.keys()]) + " - bigbox\n"

    init_str = ""
    for loc in clear_set:
        init_str += f"    (clear {loc})\n"
    for a, loc in agent_locs:
        init_str += f"    (agent-at {a} {loc})\n"
    for b, loc in boxes:
        init_str += f"    (box-at {b} {loc})\n"
    for gid, parts in big_boxes_dict.items():
        if len(parts) == 2:
            init_str += f"    (bigbox-at bbig_{gid} {parts[0]} {parts[1]})\n"
    for l1, l2 in adjacencies:
        init_str += f"    (adj {l1} {l2})\n"
    for g in goals:
        init_str += f"    (goal {g})\n"
        
    goal_str = "(won)"

    problem_str = f"""(define (problem bp-map)
  (:domain box-push)
  (:objects
{obj_str}  )
  (:init
{init_str}  )
  (:goal
    {goal_str}
  )
)
"""
    with open(problem_path, "w") as f:
        f.write(problem_str)


def generate_pddl_for_env(env, pddl_folder="pddl"):
    os.makedirs(pddl_folder, exist_ok=True)
    domain_path = os.path.join(pddl_folder, "domain.pddl")
    problem_path = os.path.join(pddl_folder, "problem.pddl")
    
    generate_domain(domain_path)
    generate_problem(env, problem_path)
    
    return domain_path, problem_path
