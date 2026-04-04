# Assignment 1 — Deterministic Box Pushing with Classical Planning

Welcome to the first programming assignment in the Reinforcement Learning & Planning course!

## 🎯 Assignment Description

Your task is to use an **LLM of your choice** to generate a PDDL file that describes the Box Pushing problem.

The world contains **two agents** moving on a 2D grid. Each agent can move in any of the four directions, as long as the target cell is not an obstacle (agents may share the same cell). Agents can also **push a box** in any direction when standing on the appropriate side of it, provided the destination cell is free. After a push, both the agent and the box move one cell in that direction.

The problem contains **3 boxes**: two regular boxes and one heavy box.
- To push a **regular box** — one agent is enough.
- To push the **heavy box** — both agents must push simultaneously from the **same cell** in the **same direction**.

**In summary:** A grid world (possibly with obstacles) containing 2 regular boxes and 1 heavy box. The world has a `move` action (per agent), a `push-small` action (single agent, per direction), and a `push-heavy` action (two agents, per direction). The goal condition refers **only to the final positions of the boxes**, not the agents.

You must generate both a **Domain file** and a **Problem file** by describing the world to your LLM. You may provide a drawing of the initial state or a natural language description. The goal condition should refer **only to box locations**.

You must install and run the classical planner and verify that the resulting plan actually solves the problem by supplying it to the simulator and confirming it reaches the goal state without errors.

---

## 🚀 Environment Setup

### Step 1: Clone the Repository

Open a terminal, navigate to your course folder, and run:

```bash
git clone https://github.com/ronberg-bgu/RL-course.git
cd RL-course
```

### Step 2: Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Step 3: Verify the Environment — Run the Visual Simulation

To make sure everything works, run:

```bash
python3 visualize_plan.py
```

This script loads a predefined map, generates PDDL files, invokes the Fast Downward planner, and displays the solution visually in a pygame window.

### Step 4: Write a PDDL → ASCII Map Translator

As part of the assignment, you must write a **Python script** (`pddl_to_map.py`) that reads the `domain.pddl` and `problem.pddl` files you generated (via the LLM) and translates them back into an ASCII map in the format the simulator accepts.

The ASCII map uses the following characters:

| Character | Meaning |
|-----------|---------|
| `W` | Wall |
| `A` | Agent |
| `B` | Small box |
| `C` | Heavy box |
| `G` | Goal cell |
| ` ` | Empty cell |

Your script should produce a list of strings like:

```python
ascii_map = [
    "WWWWWWWW",
    "W  AA  W",
    "W B C  W",
    "W      W",
    "W   B  W",
    "W G G GW",
    "WWWWWWWW"
]
```

### Step 5: Write `llm_pipeline.py` and Run End-to-End

Your `llm_pipeline.py` must tie everything together. At minimum, when run, it should:
1. Write (or embed) the LLM-generated `pddl/domain.pddl` and `pddl/problem.pddl`
2. Call `pddl_to_map.py` to reconstruct the ASCII map
3. Call the visualizer to solve and display the plan

Here is the pattern to follow in your script:

```python
from visualize_plan import visualize_pddl_plan
from pddl_to_map import parse_pddl_to_map   # your function name may differ

ascii_map = parse_pddl_to_map("pddl/domain.pddl", "pddl/problem.pddl")
visualize_pddl_plan(ascii_map, "pddl/domain.pddl", "pddl/problem.pddl")
```

To capture the full terminal log for submission:

```bash
python3 llm_pipeline.py 2>&1 | tee planner_output.txt
```

### Step 6: Create a Branch for This Assignment

**Important:** You must create a branch in the following format:

```bash
git checkout -b student-{firstname}-{lastname}-ex1
```

For example: `student-yossi-cohen-ex1`, `student-sarah-levi-ex1`

Branch names are used for tracking and grading.

### Step 7: Submit

When finished, push your branch and open a Pull Request at:
[https://github.com/ronberg-bgu/RL-course](https://github.com/ronberg-bgu/RL-course)

---

## 📬 Submission Requirements

Your submission consists of **two parts**:

### Part A — Pull Request on GitHub

Your branch must include all of the following files:

| File | Description |
|------|-------------|
| `llm_pipeline.py` (or similar name) | The pipeline code that queries the LLM, builds the PDDL files, and runs the planner |
| `pddl/domain.pddl` | The generated PDDL domain file |
| `pddl/problem.pddl` | The generated PDDL problem file |
| `pddl_to_map.py` (or similar name) | Script that parses your PDDL files and translates them back into an ASCII map for the visualizer |
| `planner_output.txt` | The **full terminal log** of the planner run (Fast Downward output) |

> **How to save the terminal log:**
> ```bash
> python3 llm_pipeline.py 2>&1 | tee planner_output.txt
> ```
> This command runs your pipeline, prints to the terminal **and** saves everything to `planner_output.txt` at the same time.

### Part B — Live Demo (In-Person Presentation)

In addition to the Pull Request, **you will present your work live in front of the course instructor.**

During the demo you are expected to:
1. Run your full pipeline end-to-end from the terminal.
2. Show the planner finding a valid plan.
3. Run the visual simulator and demonstrate the agents reaching the goal state on **your** map.
4. Explain your prompting strategy — how you described the world to the LLM and what choices you made.

> **No submission is considered complete without the live demo.**

---

## 📋 PDDL Specification — Exact Names for Simulator Compatibility

For your PDDL files to work directly with the simulator, you **must** use exactly the following names and formats.

### Types

```
agent   location   box   heavybox
```

### Predicates

| Predicate | Meaning |
|-----------|---------|
| `(agent-at ?a - agent ?loc - location)` | Agent `?a` is at location `?loc` |
| `(box-at ?b - box ?loc - location)` | Small box `?b` is at `?loc` |
| `(heavybox-at ?h - heavybox ?loc - location)` | Heavy box `?h` is at `?loc` |
| `(clear ?loc - location)` | Location `?loc` has no box on it |
| `(adj ?l1 - location ?l2 - location)` | `?l1` and `?l2` are adjacent (bidirectional) |

### Actions

| Action | Parameters | Who moves |
|--------|-----------|-----------|
| `move` | `?a ?from ?to` | One agent moves to an adjacent empty cell |
| `push-small` | `?a ?from ?boxloc ?toloc ?b` | One agent pushes a small box one cell |
| `push-heavy` | `?a1 ?a2 ?from ?boxloc ?toloc ?h` | Two agents push the heavy box together |

### Object Naming Convention

| Object | Name format | Example |
|--------|------------|---------|
| Locations | `loc_X_Y` | `loc_3_2` |
| Agents | `agent_0`, `agent_1` | — |
| Small boxes | `box_0`, `box_1` | — |
| Heavy box | `hbx_0` | — |

### Important Note on `push-heavy`

The heavy box is 1×1. Both agents must be at the **same cell** (`?from`) adjacent to the box, pushing in the same direction simultaneously.

```lisp
(:action push-heavy
  :parameters (?a1 - agent ?a2 - agent ?from - location ?boxloc - location ?toloc - location ?h - heavybox)
  :precondition (and
    (not (= ?a1 ?a2))
    (agent-at ?a1 ?from) (agent-at ?a2 ?from)
    (adj ?from ?boxloc) (heavybox-at ?h ?boxloc)
    (adj ?boxloc ?toloc) (clear ?toloc))
  :effect (and
    (agent-at ?a1 ?boxloc) (agent-at ?a2 ?boxloc)
    (not (agent-at ?a1 ?from)) (not (agent-at ?a2 ?from))
    (clear ?from)
    (heavybox-at ?h ?toloc) (not (heavybox-at ?h ?boxloc))
    (not (clear ?toloc)))
)
```

### Goal Condition

The goal condition specifies the final required positions of **all** boxes. For example:

```lisp
(:goal (and
    (box-at box_0 loc_2_5)
    (box-at box_1 loc_4_5)
    (heavybox-at hbx_0 loc_6_5)
))
```

The goal condition refers **only to box and heavy box locations** — not to agent positions.
