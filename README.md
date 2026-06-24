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
| Day 15 | Human-Guided RL and RL Taxonomy | RLHF, reward model, KL penalty, LMs in RL, imitation learning, model-based vs model-free, on-policy vs off-policy | Concept notes |
| Day 16 | Offline RL, Decision Transformers, and Curriculum Learning | Online vs offline RL, counterfactual queries, Decision Transformer, return-conditioned sequence modeling, Curriculum Learning, Automatic Curriculum Learning, Domain Randomization | Concept notes |

## Day 16 Progress: Offline RL, Decision Transformers, and Curriculum Learning

Continued advanced reinforcement learning topics and focused on how agents can learn from fixed datasets, sequence models, and staged training difficulty.

Today focused on three main ideas:

* Offline vs online reinforcement learning.
* Decision Transformers as sequence modeling for offline RL.
* Curriculum Learning and Automatic Curriculum Learning.
* Domain Randomization for robustness and sim-to-real transfer.

This day helped connect reinforcement learning with offline datasets, Transformers, imitation learning, robotics, and human-style staged learning.

New understanding:

* Online RL means the agent can keep interacting with the environment during training.
* Offline RL means the agent learns only from a fixed dataset collected beforehand.
* Offline vs online is different from on-policy vs off-policy.
* PPO is usually online and on-policy.
* DQN is usually online and off-policy.
* Offline RL is usually offline and off-policy.
* Offline RL is useful when real-world trial and error is expensive, unsafe, or impossible.
* The main challenge of offline RL is the counterfactual queries problem.
* Decision Transformer treats reinforcement learning as a conditional sequence modeling problem.
* Instead of explicitly learning a value function or using policy-gradient optimization, Decision Transformer predicts actions from trajectory history and desired return.
* Decision Transformer connects offline RL, imitation learning, and Transformer-style sequence modeling.
* A desired return is a conditioning signal, not a guarantee that the generated trajectory will actually achieve that return.
* Multiple trajectories may correspond to the same desired return.
* Curriculum Learning trains agents by gradually increasing task difficulty.
* Automatic Curriculum Learning tries to automatically select useful learning situations based on the agent’s current ability.
* Domain Randomization randomizes environment parameters during training so that the agent learns a more robust policy.

Important idea: Online Reinforcement Learning

In online reinforcement learning, the agent collects new experience by interacting with the environment during training.

Process:

```text
agent interacts with environment
→ collects experience
→ updates policy
→ interacts again
→ updates again
```

This is the setting used by most algorithms learned earlier in the course.

Examples:

* Q-learning
* DQN
* A2C
* PPO
* Self-play

Online RL is flexible because the agent can continuously collect new experience from its current behavior.

However, online RL requires either a real environment or a simulator.

This can be difficult because:

* Real-world interaction may be expensive.
* Real-world interaction may be unsafe.
* A realistic simulator may be hard to build.
* If the simulator has flaws, the agent may exploit those flaws.
* Training directly in a real system can be risky.

Badminton analogy:

Online RL feels like learning by playing new rallies.

I try a shot, observe the result, adjust my strategy, and then play again.

Important idea: Offline Reinforcement Learning

In offline reinforcement learning, the agent does not interact with the environment during training.

Instead, it learns from a fixed dataset collected beforehand.

Process:

```text
fixed dataset
→ train policy from dataset
→ no new environment interaction during training
```

The dataset may come from:

* human demonstrations
* expert policies
* older agents
* random exploration
* logged real-world behavior
* historical system data

Offline RL is useful when direct trial and error is expensive, unsafe, or impractical.

Examples:

* robotics
* autonomous driving
* healthcare
* smart manufacturing
* production scheduling
* large language model alignment

Badminton analogy:

Offline RL feels like learning only from recorded match videos.

I cannot play new rallies during training. I can only study past trajectories and try to learn a good policy from them.

Core problem in offline RL: counterfactual queries

Offline RL has a major challenge called the counterfactual queries problem.

A counterfactual question asks:

```text
What would have happened if the agent had taken a different action?
```

For example, if the dataset contains many trajectories where a car turns left at an intersection, but no examples where it turns right, the agent may still ask:

```text
What would happen if I turned right?
```

The dataset may not contain enough information to answer this.

This makes it difficult to estimate the value of unseen actions.

This problem is related to:

* distribution shift
* out-of-distribution actions
* extrapolation error
* value overestimation

If the learned policy chooses actions that are not well represented in the dataset, the value function may make unreliable predictions.

This is why offline RL often needs to be conservative.

Connection to on-policy and off-policy:

Offline vs online and on-policy vs off-policy are different classifications.

```text
Online vs Offline:
Can the agent collect new data from the environment during training?

On-policy vs Off-policy:
Does the training data come from the current policy?
```

Examples:

* PPO is usually online and on-policy.
* A2C is usually online and on-policy.
* DQN is usually online and off-policy.
* Q-learning is usually online and off-policy.
* Offline RL is offline and usually off-policy.
* Behavior cloning is offline-style imitation learning from fixed demonstrations.

Important idea: Decision Transformer

Decision Transformer treats reinforcement learning as a conditional sequence modeling problem.

Instead of explicitly training a value function, Q-function, or policy-gradient objective, it uses a Transformer model to predict future actions from trajectory history.

The model is conditioned on:

* desired return
* past states
* past actions

The input can be understood as a sequence:

```text
return-to-go, state, action, return-to-go, state, action, ...
```

The model then autoregressively predicts the next action.

This is similar to how a language model predicts the next token from previous tokens, but here the sequence contains returns, states, and actions.

Traditional RL asks:

```text
What action should I take to maximize return?
```

Decision Transformer asks:

```text
Given this desired return and past trajectory history,
what action should come next?
```

This is a shift in the RL paradigm.

Decision Transformer does not directly maximize return through a conventional RL algorithm.

Instead, it learns from offline trajectories and generates actions that are likely to be consistent with the desired return.

Connection to offline RL:

* Decision Transformer is an offline RL method.
* It learns from a fixed dataset of trajectories.
* It does not require online environment interaction during training.
* It uses sequence modeling rather than conventional value-based or policy-gradient RL.

Connection to imitation learning:

Decision Transformer is related to imitation learning because it learns from existing trajectories.

However, it is not only copying expert actions.

It is conditioned on the desired return, which means it can generate different behaviors depending on the return it is asked to achieve.

This makes it feel like return-conditioned imitation from offline trajectories.

Important detail: desired return is not a guarantee

The desired return is a conditioning signal.

It tells the model:

```text
Generate actions that look like trajectories associated with this return.
```

But it does not guarantee that the generated trajectory will actually achieve that return.

Reasons it may fail:

* The dataset may not contain enough high-return trajectories.
* The current state may not support the desired return.
* The model may sample a weaker action and drift away from a successful trajectory.
* The test environment may differ from the training data.
* The model only learns patterns from past data and cannot guarantee future success.

Multiple trajectories may achieve the same desired return.

For example, in badminton, a high-quality rally could come from:

* stable defense and counterattack
* aggressive attacking pressure
* net control
* deep clears and patience
* opponent movement control

Decision Transformer may generate different trajectories depending on the current state, past history, desired return, and sampling behavior.

Badminton analogy:

Decision Transformer feels like learning from many recorded matches.

If I ask for a high-return rally, the model tries to generate actions that resemble successful high-return rallies in the dataset.

But if my current state is already bad, or if the dataset does not contain enough examples, it cannot guarantee that the trajectory will actually reach the desired return.

Short summary:

```text
Decision Transformer is not a planner that guarantees success.
It is a sequence model that generates actions conditioned on desired return and past trajectory history.
```

Important idea: Curriculum Learning

Curriculum Learning trains an agent by gradually increasing task difficulty.

Instead of giving the agent the hardest task from the beginning, the training process is organized so that the agent can acquire skills step by step.

Examples:

* A bipedal agent may first learn to stand, then walk, then jump.
* A robot may first learn reaching, then grasping, then object manipulation.
* A game agent may first learn simple levels, then harder levels.
* A badminton beginner may first learn basic footwork, then high clears, then moving shots, then real rallies.

Curriculum Learning is useful when:

* the final task is too hard at the beginning
* rewards are sparse
* the agent needs to acquire skills incrementally
* the environment has many variations
* robustness is important

Human-learning analogy:

Curriculum Learning feels like learning badminton step by step.

I should not start from full-speed match play before learning basic footwork, grip, swing, and timing.

A good curriculum allows skills to build on each other.

Important idea: Automatic Curriculum Learning

Designing a curriculum manually can be difficult.

Automatic Curriculum Learning tries to automatically adjust the distribution of training tasks based on the agent’s current ability.

The goal is to select learning situations that are useful for the agent.

A good task should not be too easy or too hard.

```text
Too easy:
the agent learns little.

Too hard:
the agent receives no useful feedback.

Appropriately challenging:
the agent can improve.
```

Automatic Curriculum Learning tries to adapt the learning environment so that the agent receives useful training situations.

Connection to self-play:

Self-play can create an automatic curriculum.

As the agent improves, its opponents can also become stronger.

This naturally increases task difficulty.

Connection to curiosity:

Curiosity encourages the agent to explore novel or difficult-to-predict states.

Curriculum Learning organizes the learning process by selecting useful tasks or environments.

Both help the agent learn more efficiently when simple reward maximization is not enough.

Important idea: Domain Randomization

Domain Randomization means randomly changing environment parameters during training.

The goal is to make the agent robust to variations and reduce overfitting to one fixed simulator.

Examples of randomized factors:

* object positions
* object mass
* friction
* lighting
* camera angle
* terrain
* initial states
* sensor noise
* control delay
* environment disturbances

This is especially important in robotics because a policy trained in a clean simulator may fail in the real world.

The simulator and the real world are never perfectly the same.

This gap is called the sim-to-real gap.

Domain Randomization helps by exposing the agent to many variations during training.

This makes the learned policy less fragile.

Analogy to supervised learning:

Domain Randomization is like data augmentation for reinforcement learning environments.

In supervised learning, we may change image brightness, crop images, or add noise to make a vision model more robust.

In reinforcement learning, we randomize environment properties so that the agent does not overfit to one fixed setup.

Badminton analogy:

Training only in one stable condition can make a player fragile.

A more robust player practices with:

* different shuttle speeds
* different opponents
* different court conditions
* different rhythms
* different levels of pressure
* different tactical situations

This helps the player adapt to real matches.

Connection to Curriculum Learning:

Curriculum Learning controls task difficulty.

Domain Randomization controls environment variation.

They can be combined.

For example, the training process may begin with small random variations and gradually increase the range of randomization.

This creates a curriculum over environment diversity.

Connection to smart manufacturing:

Domain Randomization can be useful in manufacturing simulation.

A smart manufacturing RL agent may need to handle variations in:

* order arrival times
* machine processing speed
* machine failure probability
* robot travel speed
* sensor noise
* resource availability
* task priority
* layout congestion

A policy trained only in one fixed factory setup may not generalize well.

Randomizing the simulation environment can help the agent learn a more robust strategy.

Big picture:

Today’s topics expand the reinforcement learning map beyond online trial-and-error.

Offline RL shows that agents can learn from fixed datasets when real-world interaction is expensive or unsafe.

Decision Transformer shows that offline RL can be approached through sequence modeling.

Curriculum Learning shows that the order and difficulty of training tasks matter.

Domain Randomization shows that environment diversity can improve robustness and sim-to-real transfer.

Together, these ideas are important for real-world AI systems, robotics, smart manufacturing, and language-model-based agents.

Human-learning analogy:

Today’s topics feel close to how humans learn.

Offline RL is like learning from past videos and records.

Decision Transformer is like learning patterns from successful trajectories and trying to reproduce actions that match a target outcome.

Curriculum Learning is like learning skills in the right order.

Automatic Curriculum Learning is like having a coach who adjusts training difficulty based on current ability.

Domain Randomization is like practicing under many different conditions so that the skill becomes robust.

Key idea:

Modern reinforcement learning is not only about an agent repeatedly interacting with one fixed environment.

Agents may also learn from:

```text
fixed datasets
past trajectories
desired returns
progressive task difficulty
randomized environments
expert demonstrations
human feedback
language-model knowledge
```

This makes RL more practical for complex real-world systems.

Personal connection:

Today helped me connect reinforcement learning with Transformers, offline datasets, staged skill learning, and robustness.

Decision Transformer is especially interesting because it connects RL with the sequence modeling style used by language models.

Curriculum Learning feels closely related to human learning and badminton training: a skill should be built step by step.

Domain Randomization also feels intuitive because real-world performance requires adapting to different conditions instead of overfitting to one perfect training setup.

These ideas are important for my long-term interests in AI agents, robotics, simulation, game systems, smart manufacturing, and adaptive decision-making.

Current weak points to revisit:

* How return-to-go is represented in Decision Transformer.
* How Decision Transformer differs from behavior cloning in implementation.
* How Decision Transformer compares with Q-learning-based offline RL.
* What kinds of datasets are suitable for Decision Transformer.
* Why desired return conditioning does not guarantee actual achieved return.
* How Automatic Curriculum Learning selects tasks.
* How curriculum learning can be combined with self-play.
* How domain randomization helps sim-to-real transfer in robotics.
* How to choose the right range of randomization.
* How these ideas apply to smart manufacturing and robot coordination.

Next step:

* Add Day 16 to the README learning progress table.
* Organize Day 15 and Day 16 into the final advanced RL summary.
* Prepare a Day 1–16 knowledge map for future paper reading.
* Start moving from concept learning toward paper reading, small experiments, and written analysis.
* Later, explore papers or projects related to PPO, ICM, self-play, RLHF, imitation learning, offline RL, Decision Transformer, and curriculum learning.

