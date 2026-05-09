# rl-learning-lab

Notes and small experiments for learning reinforcement learning, simulation, and intelligent agent systems.

## Goal

This repository records my learning journey in reinforcement learning, multi-agent systems, simulation, and intelligent agent design.

My long-term interest is to connect AI, simulation, game systems, robotics, and adaptive decision-making.

## Learning Progress

| Day | Topic | Notes | Code |
|---|---|---|---|
| Day 1 | Reinforcement Learning Overview | Agent, environment, state, action, reward, policy | Coming soon |
| Day 2 | Value Functions and Bellman Equation | State-value function, action-value function, return, Bellman equation | Notes |
| Day 3 | Monte Carlo and Temporal Difference Learning | MC learns after a full episode; TD learns at each step using bootstrapping | Notes |
| Day 4 | Q-Learning Foundations | Q-table, epsilon-greedy, off-policy learning, Q-learning update rule | Hands-on started |
| Day 5 | Q-Learning Hands-on: FrozenLake and Taxi | Gymnasium environments, Q-table initialization, greedy policy, epsilon-greedy policy, training loop, evaluation, Hugging Face Hub upload | Notebook + Models |

````md
## Current Focus

I have completed the first hands-on pass of the Unit 2 Q-learning lab from the Hugging Face Deep RL Course.

Current progress:

- Set up the Colab environment
- Fixed compatibility issues with newer Python / Colab versions
- Created the FrozenLake-v1 environment
- Studied `render_mode="rgb_array"`
- Studied FrozenLake map symbols: `S`, `F`, `H`, `G`
- Studied observation space and state representation
- Connected `(row, col)` positions to integer state IDs
- Created and initialized the Q-table using `np.zeros`
- Implemented greedy policy
- Implemented epsilon-greedy policy
- Defined hyperparameters
- Implemented the Q-learning training loop
- Trained and evaluated the FrozenLake agent
- Extended the same Q-learning structure to Taxi-v4
- Uploaded trained Q-learning models to Hugging Face Hub

Hugging Face model repos:

- FrozenLake: `Mantry/q-FrozenLake-v1-4x4-noSlippery`
- Taxi: `Mantry/q-Taxi-v4`

Next step:

- Clean up notes from the notebook
- Add a separate markdown learning note file
- Start Unit 3: Deep Q-Learning

## Repository Structure

```text
rl-learning-lab/
│
├── README.md
├── LICENSE
├── .gitignore
└── notebooks/
    ├── .gitkeep
    └── unit2_q_learning_frozenlake_taxi.ipynb
```
