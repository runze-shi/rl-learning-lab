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

## Day 13 Progress: Self-Play and ELO Evaluation in MARL

Continued the MARL unit and focused on self-play and ELO-based evaluation.

Today focused on understanding how agents can improve in adversarial environments when it is difficult to choose a suitable opponent.

New understanding:

* In adversarial games, training an agent correctly can be difficult because the opponent’s skill level strongly affects learning.
* If the opponent is too weak, the agent may overfit to useless strategies that only work against weak opponents.
* If the opponent is too strong, the agent may lose constantly and fail to receive useful learning signals.
* The ideal opponent should be close to the agent’s current level and should gradually become stronger as the agent improves.
* Self-play solves this problem by using copies of the agent’s own policy as opponents.
* The agent can start by playing against a copy of itself, then later play against newer or older versions of itself.
* This allows the opponent difficulty to increase progressively without needing an external expert opponent.
* Self-play can be understood as bootstrapping an opponent from the agent’s own learning process.

A useful human-learning analogy:

* If I am a level 3 badminton player, playing only against level 1 players is too easy and may teach bad habits.
* Playing only against level 5 players is too hard and may give too little useful feedback.
* The best training pool is around level 2 to level 4, with occasional exposure to level 5.
* Training should include a mixture of stable opponents, similar-level opponents, and slightly stronger opponents.
* This is similar to self-play: the best opponent is not only the strongest version, but a pool of different historical versions.

Today also studied self-play hyperparameters in ML-Agents.

New understanding of hyperparameters:

* `swap_steps` controls how often the agent changes opponents.
* If opponents change too quickly, training may become unstable.
* If opponents change too slowly, the agent may overfit to a single opponent.
* `window` controls how many historical opponents are saved in the opponent pool.
* A larger window means more opponent diversity because the agent can train against policies from different stages of training.
* `play_against_latest_model_ratio` controls how often the agent plays against the latest version of itself instead of a sampled historical opponent.
* A higher ratio means the agent plays more often against its most recent version, which can create stronger pressure to improve but may reduce stability.
* `save_steps` controls how often a new version of the policy is saved into the opponent pool.
* If policies are saved too frequently, many saved opponents may be too similar.
* If policies are saved too rarely, the opponent pool may miss useful intermediate difficulty levels.
* `team_change` can be understood as how often teams or opponents are changed in team-based settings.

Key idea:

Self-play is not simply “playing against yourself.” It is a way to construct an adaptive opponent curriculum. The agent learns from opponents that are similar to itself, older than itself, or slightly stronger than itself.

Today also studied ELO as a way to evaluate agents.

New understanding:

* In adversarial games, cumulative reward is not always a meaningful metric because performance depends heavily on opponent strength.
* ELO is a relative rating system that estimates the skill level of players or agents based on match outcomes.
* ELO is especially useful in zero-sum games, where one side’s gain corresponds to the other side’s loss.
* The expected score is calculated from the rating difference between two players.
* If a high-rated player beats a low-rated player, the high-rated player gains only a few points because the result was expected.
* If a low-rated player beats a high-rated player, the low-rated player gains many points because the result was unexpected.
* If there is a draw, the lower-rated player may gain some points because drawing against a stronger opponent is better than expected.

ELO update formula:

```text
R'_A = R_A + K(S_A - E_A)
```

Where:

* `R_A` is the current rating of player A.
* `R'_A` is the updated rating.
* `E_A` is the expected score.
* `S_A` is the actual score, where win = 1, draw = 0.5, and loss = 0.
* `K` is the K-factor, which controls how much the rating changes after one game.

Understanding K-factor:

* K-factor is like the learning rate of the rating system.
* A larger K means the rating changes more after each game.
* A smaller K means the rating changes more slowly and remains more stable.
* New or uncertain players usually have a larger K because the system is less certain about their true skill.
* Experienced or high-level players usually have a smaller K because their rating has already been tested over many games.
* In the same match, different players may use different K values.
* This means a new account and an old account can have similar ratings and be matched together, but their rating changes after the match may be different.
* If both sides use the same K, rating changes are more symmetric.
* If players use different K values, rating changes may not be perfectly zero-sum.

Advantages of ELO:

* Points are balanced when using the same K-factor, because the winner takes points from the loser.
* It is self-correcting: beating weak opponents gives only a few points, while beating strong opponents gives more points.
* It works reasonably well for team games by using the average rating of each team.

Disadvantages of ELO:

* ELO does not directly measure each individual’s contribution in a team game.
* In team settings, all players may gain or lose rating together even if their individual contributions were very different.
* This connects to the credit assignment problem in MARL.
* Rating deflation can occur because a good rating requires maintaining skill over time.
* Ratings from different historical periods cannot always be compared directly because the player population, strategies, environment, or game version may have changed.

Key summary:

Self-play helps an agent improve by training it against adaptive opponents generated from its own past and current policies.

The opponent pool should contain a useful range of difficulties and behaviors. Too easy leads to overfitting. Too hard leads to poor learning. A diverse pool helps the agent learn a more general and robust policy.

ELO is useful for evaluating agents in adversarial environments because it measures relative skill instead of only tracking reward. However, ELO depends on the opponent population and does not directly explain individual contribution or specific skill improvement.

Personal connection:

Self-play feels closely related to human learning. The best opponent is not always the strongest person. A good learning system needs a pool of opponents around my current level, slightly below my level for stability, and slightly above my level for growth.

This also connects to badminton training: if I am currently around level 3, I should mostly train with level 2 to level 4 players, occasionally play against level 5 players, and avoid only playing against level 1 players. Learning works best when the challenge is difficult but still understandable.

Current weak points to revisit:

* How self-play is implemented in code in ML-Agents.
* How opponent sampling works from the saved policy window.
* The exact difference between latest model ratio and opponent window.
* How self-play avoids overfitting to a single historical opponent.
* How ELO is computed for teams in practical MARL environments.
* How ELO relates to modern rating systems such as MMR, Glicko, or TrueSkill.
* How to evaluate individual contribution in team-based MARL beyond ELO.

Next step:

* Continue the MARL unit with the soccer team AI-vs-AI example.
* Understand how self-play is applied in a concrete team environment.
* Connect self-play, ELO, team reward, and opponent pools to practical MARL training.
* Later, compare this with smart manufacturing scenarios where agents may cooperate, compete for resources, or coordinate under shared global objectives.
