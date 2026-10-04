"""
PA1 — algorithms.

Fill in every function marked TODO. Keep the signatures: train.py and
evaluate.py call them as written, and the grader assumes the conventions in
the docstrings. Read PA1_SPEC.html section 2 before writing a line.

Do NOT import stable-baselines3, cleanrl, or any library that implements
these algorithms for you.
"""
from __future__ import annotations

import numpy as np

from pa1_envs import TabularEnv


# ============================================================ dynamic programming
#
# All three functions take the model env.P directly. Conventions that decide
# the graded numbers:
#   - SYNCHRONOUS sweeps: read V_k, write into a fresh V_{k+1}. Not in-place.
#   - V starts at 0 everywhere. Terminal states stay 0.
#   - A sweep is one full pass over all states; the first pass is sweep 1.
#   - argmax ties break toward the LOWEST action index. np.argmax does this.


def policy_evaluation(env: TabularEnv, policy: np.ndarray,
                      sweeps: int | None = None, theta: float = 1e-12
                      ) -> tuple[np.ndarray, int]:
    """Iterative policy evaluation.

    policy[s, a] is the probability of action a in state s.
    If `sweeps` is given, run exactly that many and stop.
    Otherwise run until max|V_{k+1} - V_k| < theta.
    Returns (V, number_of_sweeps_performed).
    """
    V = np.zeros(env.n_states)      # V = 0 everywhere, including terminal states
    k = 0                           # sweeps performed; the first pass is sweep 1

    while True:
        V_old = V.copy()            # synchronous: read only from V_k
        delta = 0.0                 # largest change in this sweep

        for s in range(env.n_states):
            v = 0.0
            for a in range(env.n_actions):
                prob = policy[s, a]                 # pi(a|s)
                for prob_next, next_state, reward, terminated in env.P[s][a]:
                    v += prob * prob_next * (reward + env.gamma * V_old[next_state] * (1 - terminated))
            delta = max(delta, abs(v - V_old[s]))
            V[s] = v

        k += 1

        if sweeps is not None:      # fixed number of sweeps
            if k >= sweeps:
                break
        elif delta < theta:         # run to convergence
            break

    return V, k


def greedy_policy(env: TabularEnv, V: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy w.r.t. V. Returns an int array of length n_states.

    Ties break toward the lowest action index.
    """
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        # one-step lookahead: score each action
        q = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for prob_next, next_state, reward, terminated in env.P[s][a]:
                q[a] += prob_next * (reward + env.gamma * V[next_state] * (1 - terminated))

        # np.argmax returns the first maximum, so ties go to the lowest action index
        policy[s] = np.argmax(q)

    return policy


def policy_iteration(env: TabularEnv, theta: float = 1e-12
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Policy iteration.

    Initial policy: action 0 (up) in every state.
    One round = policy evaluation to convergence + one greedy improvement.
    Count the final round, in which the policy does not change.
    Returns (V, policy, number_of_rounds).
    """
    policy = np.zeros(env.n_states, dtype=int)  # initial policy: up in every state
    rounds = 0

    while True:
        rounds += 1                 # counted first, so the final round is included

        # 1. policy evaluation: one-hot pi[s, a] from the action list
        pi = np.eye(env.n_actions)[policy]
        V, _ = policy_evaluation(env, pi, theta=theta)

        # 2. policy improvement: act greedily w.r.t. V
        new_policy = greedy_policy(env, V)

        # 3. stop when the policy no longer changes
        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return V, policy, rounds


def value_iteration(env: TabularEnv, theta: float = 1e-12
                    ) -> tuple[np.ndarray, int]:
    """Value iteration, synchronous. Returns (V, number_of_sweeps)."""
    V = np.zeros(env.n_states)      # V = 0 everywhere, including terminal states
    sweeps = 0

    while True:
        sweeps += 1
        delta = 0.0
        v_old = V.copy()            # synchronous: read only from V_k

        for s in range(env.n_states):
            # one-step lookahead: score each action
            q = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for prob_next, next_state, reward, terminated in env.P[s][a]:
                    q[a] += prob_next * (reward + env.gamma * v_old[next_state] * (1 - terminated))

            v_new = np.max(q)       # best action's value (Bellman optimality)
            delta = max(delta, abs(v_new - v_old[s]))
            V[s] = v_new

        if delta < theta:
            break

    return V, sweeps


# ================================================================== TD control
#
# Conventions:
#   - Q starts at 0 everywhere.
#   - epsilon-greedy: with probability epsilon choose uniformly among all
#     actions; otherwise argmax Q[s] (lowest index on ties).
#   - Training returns are UNDISCOUNTED sums of rewards per episode.
#   - Use the provided rng for every random draw so results reproduce.


def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float,
                   rng: np.random.Generator) -> int:
    """Return an action for state s."""
    if rng.random() < epsilon:
        # explore: any of the actions, uniformly at random
        return int(rng.integers(Q.shape[1]))

    # exploit: best action; np.argmax breaks ties toward the lowest index
    return int(np.argmax(Q[s]))
    


def sarsa(env: TabularEnv, alpha: float, epsilon: float, episodes: int,
          max_steps: int, rng: np.random.Generator
          ) -> tuple[np.ndarray, list[float]]:
    """On-policy TD control.

    Returns (Q, per_episode_training_returns).
    The target uses the action actually taken next: Q[s', a'].
    """
    
    Q = np.zeros((env.n_states, env.n_actions))  # Q = 0 everywhere
    returns = []  # per-episode undiscounted returns

    for _ in range(episodes):
        s, _ = env.reset()  # start state
        a = epsilon_greedy(Q, s, epsilon, rng)  # choose initial action
        G = 0.0  # undiscounted return for this episode

        for _ in range(max_steps):
            s_next, reward, terminated, _, _= env.step(a)  # take action a
            G += reward  # accumulate undiscounted return

            # choose next action using epsilon-greedy policy
            a_next = epsilon_greedy(Q, s_next, epsilon, rng)

            # update Q using the SARSA update rule
            target = reward if terminated else reward + env.gamma * Q[s_next, a_next]
            Q[s, a] += alpha * (target - Q[s, a])

            s, a = s_next, a_next  # move to the next state and action

            if terminated:
                break

        returns.append(G)  # record the return for this episode

    return Q, returns            


def q_learning(env: TabularEnv, alpha: float, epsilon: float, episodes: int,
               max_steps: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, list[float]]:
    """Off-policy TD control.

    Returns (Q, per_episode_training_returns).
    The target uses the best next action: max_a' Q[s', a'].
    """
    Q = np.zeros((env.n_states, env.n_actions))  # Q = 0 everywhere
    returns = []  # per-episode undiscounted returns

    for _ in range(episodes):
        s, _ = env.reset()  # start state
        G = 0.0  # undiscounted return for this episode

        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)  # choose action using epsilon-greedy policy
            s_next, reward, terminated, _, _ = env.step(a)  # take action a
            G += reward  # accumulate undiscounted return

            # update Q using the Q-learning update rule
            target = reward if terminated else reward + env.gamma * np.max(Q[s_next])
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_next  # move to the next state
            if terminated:
                break

        returns.append(G)  # record the return for this episode
        
    return Q, returns            
