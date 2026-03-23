# RL Course - Multi-Agent Box Pushing Environment

Welcome to the Reinforcement Learning (RL) course repository!

This repository serves as the central hub for our multi-agent box-pushing RL environment course. Here you will find the course environment, exercises, and assignment instructions.

## 🚀 Setting Up the Environment

Before you can run the simulations, you need to install the required Python dependencies. The environment relies on `pygame` for graphics, `minigrid` / `pettingzoo` for RL interfaces, and `unified-planning` with `fast-downward` for solving logical PDDL puzzles.

Since modern operating systems (like macOS) protect the system Python environment, you must create a Virtual Environment before installing the packages:

```bash
# 1. Create a virtual environment named 'venv'
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required dependencies safely
pip install -r requirements.txt
```

## 🎮 Running the Visual Simulation

We have included a full end-to-end visualizer that automatically builds a PDDL domain out of a 2D grid, sends it to the Fast Downward AI planner, and then plays the optimal solution visually on your screen.

To run the large multi-agent simulation where two agents cooperate to push a Big Box:

```bash
python3 visualize_plan.py
```

This script will:
1. Load a hardcoded 8x8 ASCII map.
2. Generate `domain.pddl` and `problem.pddl` in the `pddl/` folder.
3. Call the classical planner to find the shortest list of actions.
4. Launch a `pygame` window and execute the actions step-by-step.

## 📁 Project Architecture & Files

Here is an overview of the core files in this repository and what they do:

### Core Environment
*   **`environment/multi_agent_env.py`**
    The heart of the simulation! This defines the `MultiAgentBoxPushEnv` class, inheriting from PettingZoo's `ParallelEnv`. It handles the core physics: small box pushes, two-agent joint Big Box pushes, grid overlaps, and generating visual frames for the agents.
*   **`environment/box_push_env.py`**
    A simpler, single-agent Gym environment (used for basic training and earlier exercises before moving to multi-agent).
*   **`environment/objects.py`**
    Defines the visual rendering rules for our custom grid objects (`AgentObj`, `SmallBox`, `BigBox`) using standard PyGame polygon rendering metrics.
*   **`environment/wrappers.py`**
    Contains advanced RL wrappers to increase difficulty:
    *   `StochasticActionWrapper`: Adds a chance for agent actions to fail.
    *   `NoisyObservationWrapper`: Adds visual static/noise to the agent's observation matrix.

### Automated Planning (PDDL)
*   **`environment/pddl_extractor.py`**
    Acts as a bridge between the Python grid and classical planning. It parses the live environment state and writes valid mathematically-constrained `domain` and `problem` files. 
*   **`planner/pddl_solver.py`**
    Connects to the `unified-planning` library and pipes the generated PDDL files into the `fast-downward` engine, returning a parsed list of steps if a valid solution exists.

### Main Entry Points
*   **`visualize_plan.py`**
    The main testing script. It strings together the Environment, the Extractor, and the Solver, and then renders the output visually.

### Student Assignments
*   **`exercises/README.md`**
    Contains the homework assignments for the students taking this course (e.g. creating custom maps, integrating wrappers, adding constraints).

## Branch Naming Convention

**IMPORTANT:** When working on your assignments, you must create a new branch for each exercise. Your branch name **must** follow this format:

`student-{firstname}-{lastname}-{exercise}`

For example:
- `student-yossi-cohen-ex1`
- `student-sarah-levi-ex2`

Please ensure you adhere to this naming convention, as it will be used for grading and tracking your progress.
