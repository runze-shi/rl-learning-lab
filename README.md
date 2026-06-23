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
| Day 12 | Multi-Agent Reinforcement Learning Foundations | MARL motivation, cooperative vs competitive environments, decentralized learning, centralized learning, non-stationarity, perceptual aliasing, shared policy and global reward | Concept notes |
| Day 13 | Self-Play and ELO Evaluation in MARL | Self-play, opponent pool, historical policy copies, opponent difficulty, ML-Agents self-play hyperparameters, ELO rating, K-factor, relative skill evaluation | Concept notes |
| Day 14 | Proximal Policy Optimization (PPO) Foundations | Policy optimization, on-policy learning, actor-critic structure, probability ratio, clipped surrogate objective, advantage estimates, multiple epochs, stability vs sample efficiency | Concept notes |

## Day 14 Progress: Proximal Policy Optimization (PPO)

Continued policy-gradient and actor-critic learning and focused on Proximal Policy Optimization, usually called PPO.

Today focused on understanding why PPO became one of the most commonly used deep reinforcement learning algorithms, especially for continuous control, robotics-style environments, simulation, and agent training.

New understanding:

* PPO is a policy-gradient and actor-critic method.
* The actor learns a policy that decides which action to take.
* The critic learns a value function that estimates how good the current state is.
* PPO is designed to make policy updates more stable than basic policy-gradient methods.
* In vanilla policy gradient, one large update can change the policy too much and make the agent suddenly perform worse.
* PPO tries to improve the policy while preventing the new policy from moving too far away from the old policy.
* The key idea of PPO is not simply to maximize reward as fast as possible, but to make controlled, safe, and stable updates.

Core problem PPO tries to solve:

* In policy-gradient methods, the agent directly changes its policy.
* If the policy changes too aggressively, the agent may forget useful behavior.
* A bad update can make previously good actions much less likely.
* This can cause unstable learning or performance collapse.
* Earlier methods such as TRPO tried to solve this by limiting how far the new policy can move from the old policy.
* PPO keeps a similar idea but uses a simpler clipped objective instead of a complicated constrained optimization method.

Important idea: old policy vs new policy

* PPO collects data using the current policy, then treats that policy as the old policy during the update.
* The old policy is the policy that generated the training data.
* The new policy is the policy currently being updated.
* PPO compares the probability of taking the same action under the new policy and the old policy.
* This comparison tells us whether the update is making an action more or less likely than before.

Probability ratio:

```text
r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
```

Where:

* `π_θ(a_t | s_t)` is the probability of taking action `a_t` in state `s_t` under the new policy.
* `π_old(a_t | s_t)` is the probability of taking the same action under the old policy.
* `r_t(θ)` is the probability ratio between the new and old policy.

Understanding the probability ratio:

* If `r_t(θ) = 1`, the new policy gives the action the same probability as the old policy.
* If `r_t(θ) > 1`, the new policy makes the action more likely.
* If `r_t(θ) < 1`, the new policy makes the action less likely.
* PPO uses this ratio to measure how much the policy has changed for the sampled action.

Connection to advantage:

* PPO uses the advantage value to decide whether increasing or decreasing an action’s probability is good.
* Advantage measures whether an action was better or worse than expected.
* A positive advantage means the action performed better than expected.
* A negative advantage means the action performed worse than expected.

Intuition:

* If the advantage is positive, PPO wants to increase the probability of that action.
* If the advantage is negative, PPO wants to decrease the probability of that action.
* But PPO does not want to change the probability too much in one update.
* This is why PPO uses clipping.

Clipped surrogate objective:

```text
L_CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1 - ε, 1 + ε)A_t)]
```

Where:

* `A_t` is the advantage estimate.
* `r_t(θ)` is the probability ratio.
* `ε` is the clipping range, often something like `0.1` or `0.2`.
* `clip(r_t(θ), 1 - ε, 1 + ε)` limits the probability ratio to a safe range.
* The `min` operation chooses the more conservative objective.

Understanding clipping:

* PPO does not allow the new policy to gain unlimited benefit from making an action much more likely.
* If the probability ratio becomes too large, it is clipped.
* If the probability ratio becomes too small, it is also clipped.
* This discourages overly aggressive policy updates.
* The policy can still improve, but the improvement is constrained.
* This makes PPO more stable than basic policy-gradient methods.

Positive advantage case:

* If an action has positive advantage, PPO wants to increase its probability.
* However, once the probability ratio goes above `1 + ε`, the objective is clipped.
* This means PPO stops rewarding the policy for increasing that action’s probability too much.
* The agent learns: “This action was good, but do not over-update too aggressively.”

Negative advantage case:

* If an action has negative advantage, PPO wants to reduce its probability.
* However, once the probability ratio goes below `1 - ε`, the objective is clipped.
* This prevents the policy from reducing the action probability too much in a single update.
* The agent learns: “This action was bad, but do not erase it too aggressively.”

Why PPO is still on-policy:

* PPO is considered an on-policy algorithm.
* The data is collected using the current or recent version of the policy.
* PPO cannot reuse old experience indefinitely like DQN experience replay.
* This is because the probability ratio depends on the old policy that generated the data.
* If the data becomes too old, it no longer represents the current policy well.
* However, PPO can usually reuse the same batch for multiple training epochs.
* This makes PPO more sample-efficient than simple REINFORCE or basic A2C.

Actor-Critic structure in PPO:

* The actor outputs the policy, usually as an action distribution.
* The critic estimates the value of the current state.
* The critic helps compute the advantage.
* The actor uses the advantage to update the policy.
* The critic is trained to make better value predictions.
* PPO often combines policy loss, value loss, and entropy bonus in the full training objective.

Typical PPO loss components:

* Policy loss: improves the actor using the clipped surrogate objective.
* Value loss: trains the critic to predict returns or value targets.
* Entropy bonus: encourages exploration by preventing the policy from becoming too deterministic too early.

Entropy in PPO:

* Entropy measures how uncertain or spread out the policy distribution is.
* Higher entropy means the policy explores more.
* Lower entropy means the policy is more confident and deterministic.
* An entropy bonus encourages the agent to keep exploring.
* This is useful early in training because the agent should not commit too quickly to one behavior.

Why PPO is popular:

* PPO is relatively simple to implement compared with TRPO.
* PPO is usually more stable than vanilla policy gradient.
* PPO works well in many continuous-control environments.
* PPO can be used with neural-network policies.
* PPO balances stability and learning speed.
* PPO is a strong default algorithm for many reinforcement learning tasks.

Comparison with earlier algorithms:

* Compared with REINFORCE, PPO has lower variance because it usually uses a critic and advantage estimates.
* Compared with A2C, PPO can often use multiple epochs of updates on the same batch.
* Compared with DQN, PPO directly learns a policy instead of learning a Q-value table or Q-network.
* Compared with DQN, PPO is more naturally suited for continuous action spaces.
* Compared with TRPO, PPO has a simpler objective and is easier to implement.

Important limitation:

* PPO is not a magic algorithm.
* It still requires careful hyperparameter tuning.
* It can still be sample-inefficient compared with human learning.
* It can still fail if the reward design is poor.
* It can still overfit to a specific environment setup.
* PPO is stable, but not guaranteed to find the best possible policy.

Human-learning analogy:

PPO feels like learning a badminton skill without changing my whole playstyle too violently.

If I discover that a certain shot works well, I should use it more. But if I suddenly force that shot every rally, I may become predictable or break my overall rhythm.

If one shot fails, I should use it less. But I should not completely delete it from my skill set after one bad result.

PPO is like a cautious coach saying:

```text
Improve in the direction that worked,
but do not change too much at once.
```

This connects to badminton training:

* A good update is not the biggest possible update.
* A good update is stable, understandable, and repeatable.
* If I change my grip, footwork, or swing too aggressively, my whole system may collapse.
* Small controlled changes are easier to integrate.
* PPO’s clipped update is similar to controlled technical correction: improve, but stay close enough to the previous stable version.

Key idea:

PPO improves a policy by comparing the new policy with the old policy and clipping the update when the change becomes too large.

The agent should become better, but not by making unstable jumps.

PPO is important because it gives policy-gradient learning a practical balance between exploration, improvement, and stability.

Personal connection:

PPO feels closely related to my own learning process.

When I study reinforcement learning, badminton, or programming, I should not completely rewrite my habits every day. I should update my behavior in a controlled way.

If something works, increase it slightly.
If something does not work, reduce it slightly.
But do not destroy the whole system after one result.

This makes PPO feel like a “safe self-improvement algorithm.”

Current weak points to revisit:

* How advantage is estimated in PPO.
* How Generalized Advantage Estimation works.
* The exact difference between PPO and A2C in implementation.
* Why PPO can update for multiple epochs while still being considered on-policy.
* How the value loss and policy loss are combined.
* How entropy bonus affects exploration.
* How PPO handles continuous action spaces.
* How PPO is implemented in libraries such as Stable-Baselines3, CleanRL, or ML-Agents.
* How PPO relates to RLHF and large language model alignment.

Next step:

* Study PPO implementation details.
* Understand Generalized Advantage Estimation.
* Compare PPO with A2C using the actor-critic training loop.
* Later, connect PPO to RLHF and understand why PPO became important in human-feedback-based language model training.
