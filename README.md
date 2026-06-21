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

## Day 12 Progress: Multi-Agent Reinforcement Learning Foundations

Started learning Multi-Agent Reinforcement Learning (MARL).

Today focused on understanding why MARL is different from single-agent RL, and how multiple agents can interact in the same environment.

New understanding:

* In single-agent RL, one agent interacts with the environment and learns a policy that maximizes expected return.
* In MARL, multiple agents interact with the environment and also interact with each other.
* This means the learning problem is not only about one agent choosing good actions. It is also about coordination, competition, communication, and shared decision-making.
* Multi-agent environments can be cooperative, competitive, or mixed.
* In a cooperative environment, agents work together to maximize a common benefit.
* For example, warehouse robots may cooperate to load, unload, and move packages efficiently.
* In a competitive or adversarial environment, agents try to maximize their own benefit while reducing the opponent’s benefit.
* For example, in a tennis game, each player tries to beat the other player.
* Some environments are mixed, where agents cooperate with teammates while competing against opponents, such as team sports or multi-agent games.

Today also focused on decentralized and centralized multi-agent system design.

In a decentralized approach:

* Each agent is trained independently.
* Each agent treats other agents as part of the environment dynamics, not as learning agents.
* This makes the system easier to design because each agent can be trained similarly to a single-agent RL problem.
* However, this creates a non-stationary environment.
* The environment becomes non-stationary because other agents are also learning and changing their policies over time.
* From the perspective of one agent, the same state and action may lead to different outcomes because other agents may behave differently.
* This makes learning harder and can prevent stable convergence.
* A useful analogy is collaborative writing: if one agent writes the first paragraph, another agent writes the second paragraph, and a third agent writes the third paragraph without sharing context, each paragraph may be locally reasonable but the whole essay may not be logically coherent.

Another important concept was perceptual aliasing in MARL:

* Different global states can look identical from one agent’s local observation.
* If agents are identical and receive the same observation, they may take the same action.
* However, in a multi-agent task, identical agents may need to take different actions depending on their global positions, roles, or teammates’ states.
* This means local observation alone may be insufficient for coordination.
* The agent may think it is facing the same problem, but from the global perspective it is actually a different situation.
* This connects to the earlier idea of perceptual aliasing: different real states are compressed into the same observation.

In a centralized approach:

* The system uses a higher-level process to collect agents’ experiences.
* Agents may share an experience buffer.
* A common policy can be learned from collective experience.
* The policy can take global state information as input, such as the coverage map and the positions of all agents.
* The policy outputs joint actions for all agents.
* The reward is global, meaning the system evaluates how well the whole team performs instead of only evaluating each individual agent.
* This can make the environment more stationary because all agents are treated as one larger entity.
* The centralized view is useful for coordination because the system can reason about the whole team instead of only local behavior.

Key summary:

MARL extends reinforcement learning from one agent to multiple interacting agents.

The main difficulty is that other agents are not just passive parts of the environment. They may also be learning, adapting, competing, or cooperating. This makes the environment non-stationary from each individual agent’s perspective.

Decentralized learning is simple but can suffer from non-stationarity, partial observability, and poor coordination.

Centralized learning can use shared experience, global state, joint actions, and global reward to learn a more coordinated team policy.

A useful way to remember this:

* Decentralized MARL = many small agents learning independently.
* Centralized MARL = many agents treated as one larger system with shared experience and global coordination.
* The ideal future direction may be centralized training with decentralized execution, where agents learn with global information but act locally during deployment.

Personal connection:

This topic feels closely related to intelligent agent systems and future AI companion design. A multi-agent system is not only about making one agent smarter. It is about designing how many agents share memory, communicate, divide roles, coordinate actions, and optimize a shared goal.

Current weak points to revisit:

* The exact difference between centralized learning and centralized training with decentralized execution.
* How parameter sharing works when agents are identical.
* How to avoid identical agents taking identical actions when they receive similar observations.
* How global reward can create credit assignment problems.
* How MARL algorithms handle non-stationarity in practice.

Next step:

* Continue the MARL unit with self-play.
* Understand how agents can improve by playing against themselves or historical versions of themselves.
* Connect self-play with competitive environments, team games, and AI-vs-AI training.
* Later, compare centralized MARL with practical algorithms used in robotics, simulation, and smart manufacturing.
