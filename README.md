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


## Day 9 Progress: Policy Gradient Objective and REINFORCE

Continued Unit 4: Policy Gradient with PyTorch.

Today focused on understanding the objective function, trajectory probability, the Policy Gradient Theorem, and the REINFORCE algorithm.

New understanding:

- A trajectory τ represents one complete episode or rollout history, including states, actions, rewards, and transitions.
- The return R(τ) is the cumulative reward from one trajectory.
- The objective function J(θ) represents the expected return of the current policy πθ.
- J(θ) can be written as the weighted average of all possible trajectories:

  J(θ) = Στ P(τ; θ) R(τ)

- P(τ; θ) is the probability that the current policy generates trajectory τ.
- A trajectory probability is a product of two main parts at each time step:
  - Environment dynamics: P(s_{t+1} | s_t, a_t)
  - Policy action probability: πθ(a_t | s_t)
- Environment dynamics describe how the environment changes after an action.
- Policy action probability describes how likely the agent is to choose a specific action in a given state.
- A trajectory is sequential: the current action affects the next state, and the next state affects future actions.
- This connects to sequential decision-making, similar to badminton strategy, where a bad shot may be caused by the previous shot or previous positioning.
- The true policy gradient is hard to compute directly because it would require considering all possible trajectories.
- It is also difficult because the environment dynamics may be unknown or not differentiable.
- The Policy Gradient Theorem helps avoid directly differentiating the environment dynamics.
- The log trick, or likelihood ratio trick, transforms the gradient of trajectory probability into a gradient of log probability.
- A high-level understanding of the log trick:
  - log turns products into sums.
  - ∇ log f(x) = ∇f(x) / f(x).
  - This allows the gradient to focus on the policy probability terms.
- Since θ controls the policy but not the environment, the environment dynamics disappear when taking the gradient with respect to θ.
- The final policy-gradient idea is to increase the probability of actions that appeared in high-return trajectories.
- REINFORCE, also called Monte Carlo Policy Gradient, uses the estimated return from a complete episode to update the policy.
- For one trajectory, REINFORCE estimates the gradient by summing the log-probability gradients of selected actions and weighting them by the trajectory return.
- For multiple trajectories, REINFORCE averages the gradient estimates across sampled episodes.
- High-return trajectories push up the probabilities of their state-action combinations.
- Low-return trajectories provide weaker encouragement, and later methods such as baseline, advantage, and Actor-Critic can help reduce variance and improve credit assignment.

Key summary:

Policy Gradient does not need to know the full environment transition formula.  
Instead, it samples trajectories using the current policy, observes their returns, and updates θ so that actions from high-return trajectories become more likely in the future.

REINFORCE is the basic Monte Carlo version of Policy Gradient:  
collect full episodes, calculate returns, estimate the policy gradient, and update the policy.

Important formulas reviewed:

J(θ) = E_{τ ~ πθ}[R(τ)]

J(θ) = Στ P(τ; θ) R(τ)

P(τ; θ) = Π_t P(s_{t+1} | s_t, a_t) πθ(a_t | s_t)

∇θ J(θ) = E_{πθ}[∇θ log πθ(a_t | s_t) R(τ)]

Next step:

- Review the remaining Unit 4 materials at a high level.
- Complete the Unit 4 glossary and quiz.
- Revisit the optional Policy Gradient Theorem derivation after understanding the full REINFORCE workflow.
- Compare REINFORCE with value-based methods such as Q-learning and DQN.
- Later connect Policy Gradient to Actor-Critic, Advantage, PPO, and multi-agent reinforcement learning.
