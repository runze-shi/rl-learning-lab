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
| Day 5 | FrozenLake Environment Setup | Gymnasium environment, RGB array rendering, observation space, state representation, Q-table initialization | Notebook + Notes |

## Current Focus

I am currently working on the Unit 2 Q-learning hands-on lab from the Hugging Face Deep RL Course.

Current progress:

- Set up the Colab environment
- Fixed compatibility issues with newer Python / Colab versions
- Created the FrozenLake-v1 environment
- Studied `render_mode="rgb_array"`
- Studied FrozenLake map symbols: `S`, `F`, `H`, `G`
- Studied observation space and state representation
- Connected `(row, col)` positions to integer state IDs
- Created and initialized the Q-table using `np.zeros`

Next step:

- Implement greedy policy
- Implement epsilon-greedy policy
- Define hyperparameters
- Implement the Q-learning training loop
- Evaluate the trained agent
- Extend the same Q-learning structure to Taxi-v3

## Topics I Plan to Learn

- Reinforcement Learning fundamentals
- Markov Decision Processes
- Q-learning
- Deep Q-learning
- Policy Gradient methods
- Actor-Critic methods
- Multi-Agent Reinforcement Learning
- Simulation environments
- Robotics simulation
- Intelligent manufacturing systems

## Repository Structure

```text
rl-learning-lab/
│
├── README.md
├── notebooks/
│   └── unit2_q_learning_frozenlake_taxi.ipynb
├── notes/
│   └── unit2_q_learning_notes.md
└── requirements.txt
