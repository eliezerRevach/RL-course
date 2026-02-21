import pygame
import time
import re
from environment.multi_agent_env import MultiAgentBoxPushEnv
from planner.pddl_solver import solve_pddl
from minigrid.core.constants import DIR_TO_VEC

def extract_target_pos(pddl_action):
    """
    Returns a dictionary mapping agent names to their target coordinates.
    Handles single-agent actions (move, push-small) and dual-agent (push-big).
    """
    action_str = str(pddl_action)
    action_name = action_str.split('(')[0]
    params = re.findall(r'[\w_]+', action_str[action_str.find("(")+1:action_str.find(")")])
    
    agent_targets = {}
    
    if action_name.startswith('win'):
        return {}
    elif action_name == 'push-big' and len(params) >= 6:
        a1, a2 = params[0], params[1]
        t1, t2 = params[4], params[5] # boxloc1, boxloc2 are the targets for the agents
        
        parts1 = t1.split('_')
        parts2 = t2.split('_')
        if len(parts1) == 3:
            agent_targets[a1] = (int(parts1[1]), int(parts1[2]))
        if len(parts2) == 3:
            agent_targets[a2] = (int(parts2[1]), int(parts2[2]))
    elif len(params) >= 3:
        # move or push-small
        agent_name = params[0]
        target_loc = params[2] 
        parts = target_loc.split('_')
        if len(parts) == 3:
            agent_targets[agent_name] = (int(parts[1]), int(parts[2]))
            
    return agent_targets

def get_required_actions(env, agent, target_pos):
    """
    Returns a list of PettingZoo actions (0=left, 1=right, 2=forward) 
    needed to move the agent into the target adjacent position.
    """
    current_pos = env.agent_positions[agent]
    current_dir = env.agent_dirs[agent]
    
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    target_dir = None
    for d, vec in enumerate(DIR_TO_VEC):
        if vec[0] == dx and vec[1] == dy:
            target_dir = d
            break
            
    if target_dir is None:
        raise ValueError(f"Target {target_pos} is not adjacent to {current_pos}")
        
    actions = []
    while current_dir != target_dir:
        actions.append(1) # 1 = turn right
        current_dir = (current_dir + 1) % 4
    actions.append(2) # 2 = forward
    
    return actions

def visualize_pddl_plan(ascii_map, domain_file, problem_file):
    print("🧠 Solving PDDL problem...")
    plan = solve_pddl(domain_file, problem_file)
    
    if not plan:
        print("❌ No plan could be found. Cannot visualize.")
        return
        
    print(f"✅ Plan found with {len(plan.actions)} steps. Booting visualizer...")
    
    env = MultiAgentBoxPushEnv(ascii_map=ascii_map, render_mode="human")
    env.reset()
    env.core_env.render()
    
    time.sleep(1) 
    
    for pddl_action in plan.actions:
        print(f"Executing: {pddl_action}")
        
        agent_targets = extract_target_pos(pddl_action)
        
        # We need to execute the rotations for all involved agents first
        # then execute their forward movement simultaneously.
        agents = list(agent_targets.keys())
        agent_action_queues = {a: get_required_actions(env, a, agent_targets[a]) for a in agents}
        
        # Iterate until all agents have exhausted their rotation/forward actions
        while any(len(q) > 0 for q in agent_action_queues.values()):
            step_actions = {}
            for a in agents:
                if len(agent_action_queues[a]) > 0:
                    step_actions[a] = agent_action_queues[a].pop(0)
            
            env.step(step_actions)
            env.core_env.render()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            time.sleep(0.4) 
                
    print("🎉 Plan execution complete! Closing in 3 seconds.")
    time.sleep(3)
    pygame.quit()

if __name__ == "__main__":
    # We test visualizer on the Example 2 map (Joint Push Co-op)
    ex2_map = [
        "WWWWWW",
        "W AA W",
        "W CC W",
        "W G  W",
        "WWWWWW"
    ]
    visualize_pddl_plan(ex2_map, "pddl/ex2_domain.pddl", "pddl/ex2_problem.pddl")
