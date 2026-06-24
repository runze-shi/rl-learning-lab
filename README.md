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

## Day 15 Progress: Human-Guided RL, Language Models, and Imitation Learning

Continued from PPO and connected policy optimization to more recent topics in modern reinforcement learning and language-model-based agents.

Today focused on understanding how reinforcement learning can be guided by human feedback, language-model knowledge, expert demonstrations, and different learning frameworks.

The main topics were:

* Reinforcement Learning from Human Feedback, usually called RLHF.
* Reward models and human preference learning.
* KL divergence and KL penalty in RLHF.
* The connection between PPO clipping and KL penalty.
* Language models in reinforcement learning.
* Two routes for combining LMs and RL.
* Imitation learning, behavior cloning, distribution shift, and DAgger.
* Model-based vs model-free reinforcement learning.
* On-policy vs off-policy learning.
* The relationship between model-based RL and game-theoretic thinking.

Today felt like a transition from learning individual RL algorithms to understanding how different learning signals can be combined in modern AI systems.

New understanding:

* RLHF is not a single algorithm, but a multi-stage training process.
* RLHF usually starts from a pretrained language model.
* Human preference data is used to train a reward model.
* The reward model turns human preferences into a scalar reward signal.
* PPO can then be used to fine-tune the language model with respect to the reward model.
* In RLHF, the language model can be viewed as a policy.
* Text generation can be formulated as a sequential decision-making problem, where each token is an action.
* KL penalty prevents the fine-tuned language model from moving too far away from the original reference model.
* PPO clipping and KL penalty are similar in spirit because both limit overly aggressive policy changes.
* Language models can provide useful world knowledge and task understanding for RL agents.
* RL can ground and correct language-model knowledge through environment interaction.
* Imitation learning allows an agent to learn from expert demonstrations instead of pure trial and error.
* Behavior cloning is the simplest form of imitation learning.
* DAgger addresses distribution shift by asking experts to label states that the learner actually visits.
* Model-free RL learns policies or values directly from experience.
* Model-based RL learns or uses a world model to predict how the environment changes.
* On-policy algorithms learn from data generated by the current or recent policy.
* Off-policy algorithms can learn from data generated by old policies, other agents, replay buffers, or demonstrations.
* Model-based RL and game theory both involve thinking about future consequences, but game theory specifically considers other strategic agents.

Core problem RLHF tries to solve:

* Traditional language models are trained with next-token prediction.
* Next-token prediction helps a model learn language patterns, but it does not fully define what makes an answer good.
* Human preferences are difficult to write as a simple reward function.
* A good answer may need to be helpful, truthful, harmless, clear, concise, creative, or aligned with the user’s intent.
* Automatic metrics such as BLEU or ROUGE can be limited because they compare generated text with reference text using surface-level patterns.
* RLHF tries to use human feedback as data for optimization.
* Instead of manually writing the perfect reward function, RLHF learns a reward model from human preferences.

Important idea: RLHF pipeline

The basic RLHF pipeline can be understood as:

```text
Pretrained Language Model
→ Human preference data
→ Reward Model
→ RL fine-tuning, often with PPO
→ More human-aligned Language Model
```

The pretrained language model already knows how to generate coherent text.

The reward model learns which outputs humans prefer.

The RL stage updates the language model to generate outputs that receive higher reward from the reward model.

Understanding the reward model:

* The reward model is also called a preference model.
* It takes text as input and outputs a scalar reward.
* This scalar reward numerically represents how preferable the text is expected to be to humans.
* Human annotators usually compare or rank multiple model outputs.
* Ranking or pairwise comparison is often easier and less noisy than asking humans to give exact numerical scores.
* The reward model is not a perfect judge.
* It is an approximation of human preference learned from data.
* If the human feedback data is biased, noisy, or inconsistent, the reward model can inherit those problems.

Example:

```text
Prompt:
Explain PPO to a beginner.

Answer A:
PPO is a policy gradient method with a clipped surrogate objective.

Answer B:
PPO is like a cautious coach that lets an agent improve without changing its strategy too aggressively.

Human preference:
B is better for a beginner.
```

The reward model learns from many comparisons like this.

Language modeling as an RL problem:

In RLHF, a language model can be formulated as a policy.

```text
Policy: language model
State: prompt + generated tokens so far
Action: choosing the next token
Reward: reward model score minus KL penalty
```

For a normal RL agent, the policy chooses actions in an environment.

For a language model, the policy chooses tokens from the vocabulary.

The generated answer is like an episode.

The reward is usually given after the model produces a response.

Important idea: KL divergence and KL penalty

In RLHF, the final reward can be written as:

```text
r = r_θ - λr_KL
```

Where:

* `r_θ` is the reward model score.
* `r_KL` is the KL divergence penalty.
* `λ` controls how strong the penalty is.

KL divergence measures how different two probability distributions are.

In RLHF, it compares the current fine-tuned policy with the original reference model.

The KL penalty discourages the model from moving too far away from the original language model.

Why KL penalty is needed:

* The reward model is not perfect.
* If the language model only maximizes reward model score, it may exploit the reward model.
* The model may generate strange or meaningless text that receives high reward.
* This is a form of reward hacking.
* KL penalty keeps the fine-tuned model close to the original model’s language distribution.
* This helps preserve coherence and prevents unstable or unnatural behavior.

Connection between PPO clipping and KL penalty:

PPO clipping and KL penalty both limit how much the policy changes, but they work at different levels.

```text
PPO clipping:
limits how much the policy changes during one update.

KL penalty:
limits how far the current model drifts from the original reference model.
```

PPO clipping is like a local update safety mechanism.

KL penalty is like a global reference-model safety mechanism.

Badminton analogy:

```text
PPO clipping:
Do not change today’s technique too much in one training session.

KL penalty:
Do not abandon the whole stable playing style just to chase one reward signal.
```

RLHF tools and limitations:

Open-source RLHF tools include:

* TRL: a Hugging Face ecosystem library for fine-tuning pretrained language models with PPO.
* TRLX: an expanded RLHF framework for larger-scale online and offline training.
* RL4LMs: a research-oriented toolkit supporting multiple RL algorithms such as PPO, A2C, TRPO, and NLPO.

Limitations of RLHF:

* Human feedback data is expensive.
* Human annotators can disagree.
* Human preferences do not always have a single ground truth.
* Reward models are imperfect approximations.
* Models may exploit the reward model.
* RLHF does not fully solve factuality, safety, or uncertainty.
* PPO is widely used, but it may not be the final best optimizer for RLHF.

Future directions:

* Better reward models.
* Better RL optimizers beyond PPO.
* Offline RL methods.
* ILQL, or Implicit Language Q-Learning.
* Better methods for reducing reward hacking.
* Better understanding of exploration and exploitation in language-model RL.

Important idea: Language Models in RL

Language models can be useful for reinforcement learning agents because they encode knowledge from large text corpora.

A language model may know common facts such as:

```text
A key can open a door.
A cup can hold water.
A fragile object should be handled carefully.
If an agent is hungry, food may be useful.
```

This knowledge can help RL agents avoid learning everything from scratch.

Traditional RL often starts from a tabula-rasa setup, where the agent has no prior knowledge.

This can be sample-inefficient and may lead to strange behaviors.

Language models can provide prior knowledge, while RL can ground and correct that knowledge through interaction with the environment.

The core synergy is:

```text
Language Model:
world knowledge, task understanding, semantic guidance

Reinforcement Learning:
environment interaction, grounding, adaptation, correction
```

Two routes for combining LMs and RL:

Route A: Language model as the policy

* The language model itself acts as the policy.
* It observes a state, often represented as text.
* It outputs an action, command, or token sequence.
* RL updates the language model based on environment feedback.
* PPO can be used to adapt the LM to a textual or interactive environment.
* This route treats the LM as an agent that learns through RL.

Process:

```text
LM prior knowledge
→ LM as policy
→ interact with environment
→ receive reward
→ PPO or RL updates the LM
```

Route B: Frozen LM guides an RL agent

* The language model is kept mostly frozen.
* It does not directly learn from the environment.
* It provides high-level suggestions, goals, or useful exploration directions.
* A separate RL agent interacts with the environment and learns the actual policy.
* The LM acts more like a planner, teacher, or guide.

Process:

```text
Environment observation
→ convert observation to text
→ frozen LM suggests useful goals or actions
→ RL agent explores under this guidance
→ environment gives reward
→ RL policy learns
```

Short comparison:

```text
Route A:
The LM goes onto the field and learns through RL.

Route B:
The LM sits beside the field as a coach while the RL agent practices and learns.
```

Affordance model and value function:

In robotics, a language model may suggest a meaningful action, but another model is needed to check whether the action is actually possible in the current environment.

An affordance model or value function estimates whether a skill is feasible.

Example task:

```text
Put the apple on the table.
```

The language model may suggest:

```text
Find an apple.
Pick up the apple.
Go to the table.
Place the apple.
```

The affordance model or value function checks:

* Can the robot see the apple?
* Is the robot close enough to pick it up?
* Is the apple already in the gripper?
* Is the table reachable?
* Is this action feasible right now?

Summary:

```text
Language model:
Does this action make semantic sense?

Value function / affordance model:
Can this action actually work in the current environment?
```

Important idea: Imitation Learning

Imitation learning allows an agent to learn from expert demonstrations.

Instead of learning only through reward and trial and error, the agent first learns what an expert would do.

Basic format:

```text
state → expert action
```

This is useful when exploration is expensive, unsafe, or inefficient.

Examples:

* A robot learning from human demonstrations.
* A self-driving system learning from expert drivers.
* A scheduling agent learning from expert human decisions.
* A badminton beginner learning from a coach.

Behavior cloning:

Behavior cloning is the simplest form of imitation learning.

It treats imitation as supervised learning.

```text
Input: state
Label: expert action
Goal: predict the expert action
```

If actions are discrete, behavior cloning can use cross entropy loss.

If actions are continuous, it can use regression loss such as mean squared error.

Strengths of behavior cloning:

* Simple to understand.
* Efficient compared with pure trial and error.
* Does not require designing a reward function.
* Can provide a strong starting policy.

Weakness of behavior cloning:

* It only learns from expert states.
* If the learner makes mistakes, it may enter states not covered by expert demonstrations.
* In these unfamiliar states, the model may not know how to recover.
* This can lead to compounding error.

Distribution shift and compounding error:

Distribution shift happens when the learner visits states that are different from the expert demonstration data.

Compounding error happens when one small mistake leads to unfamiliar states, which then cause more mistakes.

Badminton analogy:

The coach demonstrates a high clear from a perfect position.

But in a real match, I may be late, off-balance, and behind the shuttle.

The expert demonstration did not include this bad state, so simple imitation may fail.

DAgger:

DAgger stands for Dataset Aggregation.

The key idea is to let the learner act, collect the states the learner actually visits, and ask the expert to label the correct actions for those states.

Process:

```text
1. Train an initial policy from expert demonstrations.
2. Let the learner act in the environment.
3. Collect the states visited by the learner.
4. Ask the expert for the correct actions in those states.
5. Add the new data to the dataset.
6. Retrain the policy.
7. Repeat.
```

DAgger is stronger than basic behavior cloning because it teaches the agent how to recover from its own mistakes.

Badminton analogy:

```text
Behavior cloning:
Watch the coach’s perfect demonstration.

DAgger:
Play a real rally, expose my actual mistakes, and let the coach correct the states where I really fail.
```

Connection between imitation learning and RLHF:

SFT in RLHF is similar to behavior cloning.

In imitation learning:

```text
state → expert action
```

In supervised fine-tuning for language models:

```text
prompt → human-written answer
```

The model first imitates human demonstrations.

Then RLHF continues with reward model training and PPO optimization.

Simple summary:

```text
SFT is like imitation learning.
Reward model + PPO is like reinforcement learning from human preference.
```

Important idea: Model-Based vs Model-Free RL

Model-based and model-free RL ask whether the agent uses an explicit model of the environment.

Here, “model” means a world model or environment model, not just a neural network.

A world model tries to predict:

```text
P(s' | s, a): what next state may happen after action a
R(s, a): what reward may be received
```

Model-free RL:

* Does not explicitly learn or use a world model.
* Directly learns action values or policies from experience.
* Asks: “What action should I take?” or “How valuable is this action?”
* Examples include Q-learning, DQN, DDQN, REINFORCE, A2C, and PPO.

Badminton analogy:

```text
I do not fully simulate the future.
I learn from experience that this shot usually works in this situation.
```

Model-based RL:

* Learns or uses a model of how the environment changes.
* Can simulate possible future outcomes.
* Can plan before acting.
* Asks: “If I do this, what will happen next?”

Badminton analogy:

```text
If I play a drop shot, the opponent may move forward.
If the opponent moves forward, I may lift to the back court.
If they lift weakly, I may get an attacking chance.
```

Short summary:

```text
Model-free:
learn from experience directly.

Model-based:
learn or use a world model, then plan with it.
```

Important idea: On-Policy vs Off-Policy RL

On-policy and off-policy learning ask where the training data comes from.

On-policy:

* Learns from data generated by the current or recent policy.
* The policy that generates the data is the policy being improved.
* Examples include REINFORCE, A2C, PPO, and SARSA.
* PPO is on-policy because it uses data collected by the current or recent version of the policy.
* PPO can reuse a batch for multiple epochs, but it cannot reuse very old data indefinitely like DQN.

Badminton analogy:

```text
I train myself using videos of my current playing style.
```

Off-policy:

* Can learn from data generated by a different policy.
* The data may come from old versions of the agent, random exploration, expert demonstrations, replay buffers, or offline datasets.
* Examples include Q-learning, DQN, DDQN, and many offline RL methods.

Badminton analogy:

```text
I can learn from old videos, coach videos, expert matches, or my past mistakes.
```

Short summary:

```text
On-policy:
learn from the current version of myself.

Off-policy:
learn from old versions, other agents, replay buffers, or demonstrations.
```

Important idea: Model-Based RL vs Game Theory

Model-based RL and game theory are related because both consider future consequences.

Model-based RL asks:

```text
If I take this action, how will the environment change?
```

Game theory asks:

```text
If I take this action, how will other intelligent players respond?
```

The main difference is that model-based RL may model the environment, but the environment does not always have strategic intent.

Game theory and multi-agent reinforcement learning involve other agents that also have goals, strategies, and learning behavior.

Badminton analogy:

```text
Model-based RL:
I simulate the shuttle trajectory and possible next rally states.

Game theory:
I simulate the opponent’s mind and how they may adapt to my habits.
```

Short summary:

```text
Model-based RL predicts the world.
Game theory predicts other decision-makers.
```

Big picture:

Today’s topics show that modern reinforcement learning can use many different learning signals.

Classic RL learns from reward.

Curiosity-driven exploration learns from novelty and prediction error.

Self-play learns from evolving opponents.

RLHF learns from human preferences.

Language models provide prior knowledge and task understanding.

Imitation learning learns from expert demonstrations.

Model-based RL learns or uses a world model for planning.

On-policy and off-policy learning describe where the training data comes from.

Together, these ideas show that future agents may combine reward, imitation, language knowledge, planning, human feedback, exploration, and multi-agent interaction.

Human-learning analogy:

Today’s topics feel like different ways a human can learn.

RL is like learning from results.

Imitation learning is like learning from a coach.

RLHF is like learning from human preference and feedback.

Language-model-guided RL is like using prior knowledge before practicing.

Model-based RL is like mentally simulating possible outcomes.

On-policy learning is like learning from my current version.

Off-policy learning is like learning from old videos, expert demonstrations, or stored experience.

Game-theoretic thinking is like not only predicting the world, but also predicting another player’s mind.

This connects to badminton training:

* I can learn from match outcomes.
* I can imitate a coach.
* I can receive preference-like feedback about which choice is better.
* I can use tactical knowledge before trying something.
* I can mentally simulate the next few shots.
* I can learn from current games, old videos, and expert matches.
* I can also think about how the opponent adapts to me.

Key idea:

Modern reinforcement learning is not only about trial and error.

It can combine many learning signals:

```text
reward
human feedback
expert demonstrations
language knowledge
world models
stored experience
curiosity
multi-agent interaction
```

The more complex the agent and environment become, the more important it is to understand how these learning signals fit together.

Personal connection:

Today helped me see reinforcement learning as a broader research map instead of just a list of algorithms.

PPO connects to RLHF.

RLHF connects to human preferences and language-model alignment.

Language models in RL connect to agents, robotics, and world knowledge.

Imitation learning connects to expert demonstrations and learning from teachers.

Model-based RL connects to planning and mental simulation.

On-policy and off-policy learning explain why some algorithms can reuse old experience and others cannot.

This feels important for my long-term interest in AI agents, simulation, robotics, game systems, and adaptive decision-making.

It also connects to the idea of building agents that can learn from humans, adapt through feedback, use prior knowledge, and improve over time.

Current weak points to revisit:

* The detailed training process of reward models in RLHF.
* The exact mathematical meaning of KL divergence.
* How KL penalty is implemented in PPO-based RLHF.
* The difference between PPO clipping and KL regularization in implementation.
* How ILQL works and why it is useful for offline RLHF.
* More concrete examples of LMs being used to guide RL agents.
* How affordance models are trained in robotics.
* The implementation difference between behavior cloning and DAgger.
* More examples of model-based RL algorithms.
* The connection between multi-agent RL and game theory.
* How these ideas can be applied to smart manufacturing or robot coordination.

Next step:

* Organize today’s content into the Day 15 PPT section after PPO.
* Add RLHF pipeline, KL penalty, LMs in RL, imitation learning, and RL taxonomy to the learning report.
* Later, read selected papers on PPO, RLHF, ICM, self-play, and imitation learning.
* Start moving from concept learning toward paper reading, small experiments, and written analysis.
* Explore how these concepts connect to multi-agent reinforcement learning, robotics simulation, and intelligent manufacturing systems.

