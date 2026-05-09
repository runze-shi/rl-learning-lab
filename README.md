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
| Day 6 | Deep Q-Learning Foundations | From Q-table to neural network approximation, Atari observation space, stacked frames, Q-target, TD error, bootstrapping, epsilon decay, greedy evaluation | In progress |

## Current Focus

I have completed the first hands-on pass of the Unit 2 Q-learning lab from the Hugging Face Deep RL Course and started Unit 3: Deep Q-Learning.

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
- Started studying Deep Q-Learning / DQN
- Understood why Q-tables do not scale to large observation spaces such as Atari
- Learned that DQN replaces the Q-table with a neural network that approximates Q-values
- Studied stacked frames as state input for capturing motion and temporal information
- Learned the difference between the network prediction `Q(s, a)` and the Q-target
- Understood that the Q-target is computed from real reward plus discounted future value
- Connected TD error, bootstrapping, and gradient descent to DQN training
- Reviewed how epsilon-greedy exploration gradually shifts toward exploitation through epsilon decay
- Understood that evaluation usually uses a greedy policy after training

Hugging Face model repos:

- FrozenLake: `Mantry/q-FrozenLake-v1-4x4-noSlippery`
- Taxi: `Mantry/q-Taxi-v4`

Next step:

- Finish the full DQN algorithm walkthrough
- Study experience replay and target network
- Clean up notes from the Q-learning notebook
- Add a separate markdown learning note file
- Start a DQN notebook or concept note for Unit 3
