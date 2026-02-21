# Exercise 1: Multi-Agent Classical Planning (PDDL) with PettingZoo
Welcome to the first exercise of the Reinforcement Learning & Planning course!

## 🎯 Objective
In this assignment, you will bridge the gap between abstract Logical Planning (PDDL) and a concrete Multi-Agent Reinforcement Learning (MARL) environment (PettingZoo/MiniGrid). You will explore a custom GridWorld where multiple agents must cooperate to solve physical puzzles.

## 📦 The Environment
The environment (`environment/multi_agent_env.py`) provides a 2D tile-based grid containing:
* **Agents (`A`)**: Red and Green triangles that can rotate and move. Multiple agents can overlap on the same tile.
* **Small Boxes (`B`)**: Yellow boxes. They can be pushed by a **single agent** moving forward into them.
* **Big Boxes (`C`)**: Purple boxes (occupying two grid cells). They can **only** be pushed if **two agents** stand behind the two halves and push forward **simultaneously in the exact same direction**.
* **Goal (`G`)**: The objective tile. To win, an agent or a box must reach the Goal.

### Part 1: Exploration
Run the provided `simulate.py` script.
```bash
python3 simulate.py
```
This script initializes the environment and selects random actions for the agents using the standard PettingZoo `ParallelEnv` API (`env.step(actions_dict)`). Notice that random actions almost never solve the BigBox physical constraints!

### Part 2: PDDL Translation Layer
Open `environment/pddl_extractor.py`. This script is responsible for reading the true current state of the 2D array and translating it into mathematical predicates:
* `(agent-at agent_0 loc_1_1)`
* `(box-at box_0 loc_2_2)`
* `(bigbox-at bbig_1 loc_2_3 loc_3_3)`
* `(clear loc_1_2)`

**Question 1:** Read the generated `pddl/domain.pddl` file. How does the `(push-big)` strict action constraint enforce that exactly two adjacent agents push the large box? What happens if three agents try to push it in the RL environment vs the PDDL domain?

### Part 3: Solving & Execution
We have provided a planner utilizing the `pyperplan` PDDL engine to generate an optimal logical action plan. The script `visualize_plan.py` acts as the bridge: it calls the planner, translates the symbolic output `push-big(agent_0, agent_1, loc...)` back into PettingZoo integer actions (`0=Left, 1=Right, 2=Forward`), and executes them instantly in the engine.

Run:
```bash
python3 visualize_plan.py
```

**Your Programming Task:** 
1. Create a new custom map (A Python list of strings, just like `ex2_map`). It should be uniquely complex, containing at least 3 walls forming a corridor, 2 agents, 1 BigBox, and 1 SmallBox.
2. Write a script `hw1_solution.py` that imports your new map, calls `generate_pddl_for_env()` to automatically create the PDDL text files, and then calls `solve_pddl()`.
3. Instead of simply visualizing it, write a loop that steps the `PettingZoo` environment exactly according to the logical plan, printing the total accumulated `rewards` dictionary after the maze is solved.

## 📤 Submission
Submit your `hw1_solution.py` file, along with a short text or PDF file answering **Question 1**, and containing a paste of your terminal output proving your agents successfully pushed the BigBox and reached the Goal.
