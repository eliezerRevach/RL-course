# Instructions for AI Agents

## Project Overview
This repository hosts a Reinforcement Learning (RL) course focused on a multi-agent box-pushing environment. The ultimate goal of the course is for students to implement, train, and evaluate RL agents that cooperate to push boxes to target locations in a grid-like world.

## Project Structure
- `README.md`: The main course description, branch naming instructions, and general guide for students.
- `CLAUDE.md`: This file, providing instructions and context to AI agents assisting with the repository.
- `.gitignore`: Standard Python and Mac OS git ignores.
- `environment/`: This directory contains the implementation of the RL environment (e.g., states, actions, rewards, dynamic transitions for the multi-agent box-pushing domain).
- `exercises/`: This directory serves as the space where student exercises will be added.

## Guidelines for AI Agents
- When generating code for the environment, ensure it aligns with standard RL interfaces (e.g., OpenAI Gym / Gymnasium API).
- Support multi-agent scenarios natively, ensuring actions, observations, and rewards are handled uniquely per agent or as joint structures appropriately.
- Only make changes to the `exercises/` folder if explicitly tasked to design or fix an exercise.
- Keep `environment/__init__.py` clean unless making environment modules importable at the top level.
