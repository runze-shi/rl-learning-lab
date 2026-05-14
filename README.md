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
| Day 7 | DQN Algorithm and Stabilization | Experience replay, replay memory, fixed Q-target, current network vs target network, Double DQN, theta as learned parameters, phi as processed state input | Concept notes |

### Day 7 Progress: DQN Algorithm and Stabilization

Continued studying Deep Q-Learning / DQN with a focus on the full training algorithm and the stability techniques used in DQN.

New understanding:

- DQN replaces the Q-table with a neural network that predicts Q-values.
- The Q-network uses trainable parameters, usually written as theta.
- Theta represents all learned weights and biases of the neural network.
- The state input may be written as phi, which means a processed state representation.
- The Q-target is the learning target for the current Q-value prediction.
- TD error measures the difference between the Q-target and the current Q-value prediction.
- Experience replay stores past experience tuples in replay memory.
- Each experience tuple contains state, action, reward, and next state.
- DQN randomly samples mini-batches from replay memory instead of only learning from the latest experience.
- Experience replay helps reuse experiences, reduce correlation between sequential samples, prevent forgetting, and stabilize learning.
- DQN uses two networks:
  - Current Q-network: the main network being trained.
  - Target Q-network: a delayed copy used to compute more stable Q-targets.
- Fixed Q-target means the target network is updated only every few steps, so the learning target does not move all the time.
- Double DQN separates action selection and action evaluation.
- In Double DQN, the current network selects the best next action, and the target network evaluates that selected action.
- This helps reduce Q-value overestimation caused by the max operation.

Hugging Face model repos:

- FrozenLake: `Mantry/q-FrozenLake-v1-4x4-noSlippery`
- Taxi: `Mantry/q-Taxi-v4`

Next step:

- Review the full DQN algorithm from start to finish
- Organize Day 6 and Day 7 into a separate Unit 3 DQN concept note
- Prepare a short meeting explanation connecting DQN to smart manufacturing
- Start DQN hands-on only after the core algorithm feels clearer
- Later connect DQN ideas to multi-agent reinforcement learning and robotics simulation
