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

## Day 8 Progress: Policy Gradient Foundations

Restarted RL study after a one-week break and began Unit 4: Policy Gradient with PyTorch.

New understanding:

- The main goal of reinforcement learning is to find an optimal policy π* that maximizes the expected cumulative reward.
- Value-based methods learn a value function first, such as Q(s, a), and then derive a policy from the learned values.
- In Q-learning and DQN, the policy is usually implicit because actions are selected based on the highest Q-value.
- Policy-based methods directly learn or search for an optimal policy instead of learning a value function first.
- A parameterized stochastic policy πθ(a|s) outputs a probability distribution over actions given a state.
- θ represents the trainable parameters of the policy, such as the weights and biases of a neural network.
- The objective function J(θ) represents the expected cumulative reward of the policy.
- The goal is to find the value of θ that maximizes J(θ).
- Policy-gradient methods are a subclass of policy-based methods.
- Policy-gradient methods directly optimize θ by using gradient ascent on J(θ).
- Gradient ascent is used because the objective is to maximize expected return.
- In implementation, this can also be written as minimizing a negative loss using gradient descent.
- Policy-based methods are often on-policy, meaning they use trajectories collected by the most recent version of the current policy.
- A stochastic policy can help in perceptual aliasing situations, where different true states look the same to the agent.
- In the dust-cleaning robot example, two red states have the same observation: walls above and below.
- These two red states require different actions, but a deterministic policy may always choose the same direction and get stuck.
- A stochastic policy can choose left or right with some probability, making it more likely to eventually reach the dust.
- Policy-gradient methods are useful for high-dimensional and continuous action spaces because they do not need to output a Q-value for every possible action.
- Policy-gradient methods can change action preferences more smoothly over time than value-based greedy policies.
- Disadvantages of policy-gradient methods include local maximum convergence, slower training, and high variance.
- Actor-Critic methods will later help reduce variance by combining policy learning with value estimation.

Key summary:

Policy-based methods directly learn how to act.  
Policy-gradient methods optimize a parameterized policy πθ using gradient ascent to maximize expected cumulative reward J(θ).  
Compared with value-based methods, policy-gradient methods are more flexible for stochastic policies and continuous action spaces, but they can be slower and less stable.

Next step:

- Continue Unit 4 by reviewing the policy-gradient objective function and gradient ascent.
- Read the optional Policy Gradient Theorem section at a high level.
- Complete the Unit 4 glossary and quiz.
- Prepare a short explanation comparing DQN and Policy Gradient.
- Later connect Policy Gradient to Actor-Critic, PPO, and multi-agent reinforcement learning.
