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
| Day 8 | Policy Gradient Foundations | Policy-based vs value-based methods, stochastic policy, parameterized policy, perceptual aliasing, gradient ascent, advantages and disadvantages | Concept notes |
| Day 9 | Policy Gradient Objective and REINFORCE | Objective function J(θ), trajectory probability, environment dynamics, log trick, Policy Gradient Theorem, REINFORCE algorithm | Unit 4 notes |
| Day 10 | From REINFORCE to Actor-Critic and Advantage | Variance in REINFORCE, Actor-Critic structure, Critic as value estimator, TD error, Advantage function, A2C intuition | Concept notes |


## Day 10 Progress: From REINFORCE to Actor-Critic and Advantage

Continued Unit 6: Actor-Critic Methods.

Today focused on understanding why REINFORCE has high variance and how Actor-Critic methods improve policy learning by adding a value estimator.

New understanding:

- REINFORCE updates the policy using full-episode returns.
- Because full-episode returns can be strongly affected by randomness, REINFORCE has high variance.
- High variance means the learning signal is unstable across episodes, even if the estimator is not necessarily biased.
- Actor-Critic methods use two function approximators:
  - Actor: learns the policy and chooses actions.
  - Critic: learns a value estimate and evaluates states or state-action pairs.
- The Actor updates its policy using feedback from the Critic.
- The Critic updates its value prediction using TD error.
- TD error compares a new target with the Critic’s current prediction:

  TD error = TD target - current prediction

- If TD error is positive, the Critic underestimated the value.
- If TD error is negative, the Critic overestimated the value.
- Advantage measures whether an action is better or worse than expected in the current state:

  A(s, a) = Q(s, a) - V(s)

- Advantage is not a probability difference. It is a value / expected return difference.
- A positive advantage means the action is better than the state’s average expected value.
- A negative advantage means the action is worse than the state’s average expected value.
- A2C uses advantage to make policy updates more stable and more informative than using raw returns or raw Q-values.

Key summary:

REINFORCE asks: “Did this full episode get a good return?”

Actor-Critic asks: “Was this action good according to the Critic’s value estimate?”

Advantage Actor-Critic asks: “Was this action better or worse than expected in this state?”

This makes the learning signal more stable and helps reduce the high variance problem in REINFORCE.

Current weak points to revisit:

- TD error sign:
  - target > prediction → positive TD error → Critic underestimated.
  - target < prediction → negative TD error → Critic overestimated.
- Advantage is a value difference, not a probability difference.
- V(s) is a state-value baseline, not simply a historical average.

Next step:

Do a small hands-on project before moving further:
- Implement or study a simple Actor-Critic / A2C example.
- Track Actor output, Critic value prediction, TD target, TD error, and Advantage.
- Compare REINFORCE-style full-return learning with Actor-Critic step-by-step learning.
- After hands-on, continue toward PPO and later multi-agent reinforcement learning.
