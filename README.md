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
| Day 11 | Curiosity-Driven Exploration and ICM | State-dependent rewards, intrinsic reward, feature representation, inverse model, forward model, prediction error as curiosity | Concept notes |


## Day 11 Progress: Curiosity-Driven Exploration and ICM

Continued Unit 5 and curiosity-driven RL reading.

Today focused on connecting the pyramid / gold brick example from Unit 5 with curiosity-driven exploration and the Intrinsic Curiosity Module (ICM).

New understanding:

- Reward is not only object-dependent. The same object can have different reward depending on the current state and previous actions.
- In the pyramid example, a gold brick is meaningful only after the correct state has been reached, such as pressing the button and generating the pyramid.
- This means the agent should not only recognize objects. It also needs to understand state, action, and environment dynamics.
- Sparse extrinsic reward is difficult because the agent may receive mostly zero reward before finding the meaningful goal.
- Curiosity-driven exploration gives the agent an intrinsic reward when the result of an action is hard to predict.
- In ICM, feature representation Φ(s) means the useful encoded features of a state, not the full raw observation.
- The inverse model predicts the action from Φ(s_t) and Φ(s_{t+1}).
- The inverse model loss L_I trains the encoder to keep action-relevant features and ignore irrelevant environmental changes.
- The forward model predicts Φ(s_{t+1}) from Φ(s_t) and a_t.
- The forward model loss L_F is the squared L2 prediction error between predicted next features and real next features.
- The forward prediction error becomes intrinsic reward:

  r_i = η / 2 * || Φ_hat(s_{t+1}) - Φ(s_{t+1}) ||_2^2

- If prediction error is large, the state is novel or not yet understood, so intrinsic reward is high.
- If prediction error becomes small, the agent has learned that area, so the curiosity reward decreases.
- The policy is encouraged to seek high intrinsic reward, while the forward model is trained to reduce prediction error.
- This creates a learning loop:

  unknown area → high prediction error → curiosity reward → exploration → learning → lower error → move to the next unknown area

Key summary:

The pyramid example shows that reward is state-dependent, not simply object-dependent.

ICM extends this idea by allowing the agent to create its own intrinsic reward from prediction error.

The inverse model helps learn what information is action-relevant. The forward model learns how the environment changes after an action. The policy is then encouraged to explore places where the forward model is still surprised.

Current weak points to revisit:

- The negative sign in the policy objective:
  - RL wants to maximize expected reward.
  - The combined objective is written as minimization.
  - Therefore the reward term appears as negative expected return.
- L_F has two roles:
  - For the policy, larger prediction error means larger intrinsic reward.
  - For the forward model, larger prediction error means larger loss that should be minimized.
- L_I does not directly create curiosity reward. It mainly helps learn action-relevant feature representation.

Next step:

- Review the original ICM paper or a clean implementation.
- Draw the ICM computation graph from raw state to Φ(s), inverse loss, forward loss, and intrinsic reward.
- After this conceptual pass, move toward PPO and compare curiosity-driven exploration with standard extrinsic-reward optimization.
