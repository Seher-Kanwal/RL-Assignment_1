"""
PA1 — training entry point.

    python train.py                       # everything, all seeds
    python train.py --config configs/pa1.yaml

Runs the dynamic-programming computations on RoomsGridWorld and the TD
control experiments on CliffWalk, writes raw per-seed output to results/,
and fills in results.json.

The structure below is a suggestion. Rearrange it if you prefer, but keep
the three things the grader depends on: results.json in the required
format, one raw file per seed in results/, and reproducibility from the
seeds in the config.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import yaml

from pa1_envs import RoomsGridWorld, CliffWalk, PROBE_STATES, ACTION_NAMES
import agent

# helper functions
def probe_values(V: np.ndarray, probes: dict) -> dict:
    """V at each probe state, rounded to 3 decimals: {"A": ..., "B": ..., ...}."""
    return {label: round(float(V[s]), 3) for label, s in probes.items()}


def probe_actions(policy: np.ndarray, probes: dict) -> dict:
    """Action name at each probe state: {"A": "up", ...}."""
    return {label: ACTION_NAMES[int(policy[s])] for label, s in probes.items()}


def rollout(env, policy: np.ndarray, max_steps: int = 1000) -> tuple[float, list[int]]:
    """Follow a deterministic policy from the start state using the model env.P.
    Returns (undiscounted_return, path_of_states)."""
    s = env.start_state
    total = 0.0
    path = [s]
    for _ in range(max_steps):
        _, s, reward, terminated = env.P[s][int(policy[s])][0]
        total += reward
        path.append(s)
        if terminated:
            break
    return total, path



def run_dynamic_programming(cfg: dict) -> dict:
    """Return the 'gridworld' block of results.json."""
    env = RoomsGridWorld()
    probes = PROBE_STATES["gridworld"]
    theta = float(cfg["dynamic_programming"]["theta"])
    uniform = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

    n_sweeps = int(cfg["dynamic_programming"]["policy_eval_sweeps"])

    # 1. random policy evaluation, fixed number of sweeps
    V_rand_10, _ = agent.policy_evaluation(env, uniform, sweeps=n_sweeps)
    V_rand_conv, _ = agent.policy_evaluation(env, uniform, theta=theta)

    # 2. optimal: policy iteration and value iteration
    V_pi, policy_pi, pi_rounds = agent.policy_iteration(env, theta=theta)
    V_star, vi_sweeps = agent.value_iteration(env, theta=theta)
    policy_star = agent.greedy_policy(env, V_star)

    # sanity check from the spec: both methods must agree
    if not np.array_equal(policy_pi, policy_star):
        print("WARNING: policy iteration and value iteration disagree")

    # 3. follow the optimal policy from S
    ret, path = rollout(env, policy_star)

    return {
        "V_random_10_sweeps": probe_values(V_rand_10, probes),
        "V_random_converged": probe_values(V_rand_conv, probes),
        "V_optimal": probe_values(V_star, probes),
        "optimal_action": probe_actions(policy_star, probes),
        "policy_iteration_rounds": int(pi_rounds),
        "value_iteration_sweeps": int(vi_sweeps),
        "optimal_undiscounted_return_from_start": round(float(ret), 3),
    }



def run_cliff_dp(cfg: dict) -> dict:
    """Value-iteration items for the 'cliff' block."""
    env = CliffWalk()
    probes = PROBE_STATES["cliff"]
    theta = float(cfg["dynamic_programming"]["theta"])

     # 1. optimal values and optimal policy
    V_star, vi_sweeps = agent.value_iteration(env, theta=theta)
    policy_star = agent.greedy_policy(env, V_star)

    # 2. follow the optimal policy from S
    ret, path = rollout(env, policy_star)

    # sanity check from the spec: the optimal policy must reach the goal
    if path[-1] != env.goal_state:
        print("WARNING: optimal policy did not reach the goal")

    # 3. assemble the cliff DP entries for results.json
    return {
        "V_optimal": probe_values(V_star, probes),
        "optimal_action": probe_actions(policy_star, probes),
        "value_iteration_sweeps": int(vi_sweeps),
        "optimal_undiscounted_return_from_start": round(float(ret), 3),
        "optimal_path_length": len(path) - 1,
    }



def run_td_control(cfg: dict, seeds: list[int], raw_dir: pathlib.Path) -> dict:
    """SARSA and Q-learning over all seeds. Returns the six reported TD items."""
    td = cfg["td_control"]
    env = CliffWalk()

    # hyperparameters from configs/pa1.yaml
    alpha = float(td["alpha"])
    epsilon = float(td["epsilon"])
    episodes = int(td["episodes"])
    max_steps = int(td["max_steps_per_episode"])
    window = int(td["training_return_window"])

    algorithms = {"sarsa": agent.sarsa, "qlearning": agent.q_learning}
    summary = {name: {"training": [], "greedy": [], "highest_row": []}
               for name in algorithms}

    for seed in seeds:
        for name, train_fn in algorithms.items():

            # 1. train: a fresh rng per (algorithm, seed) so every run reproduces
            rng = np.random.default_rng(seed)
            Q, returns = train_fn(env, alpha, epsilon, episodes, max_steps, rng)
            returns = np.asarray(returns, dtype=float)

            # 2. greedy evaluation: argmax Q, no epsilon, no learning
            policy = np.argmax(Q, axis=1)
            greedy_return, path = rollout(env, policy, max_steps=max_steps)
            highest_row = min(s // env.n_cols for s in path)
            if path[-1] != env.goal_state:
                print(f"WARNING: {name} seed {seed}: greedy policy did not reach the goal")

            # 3. raw per-seed output
            np.savez(raw_dir / f"{name}_seed{seed}.npz",
                     Q=Q, returns=returns, greedy_path=np.array(path),
                     greedy_return=greedy_return, highest_row=highest_row, seed=seed)

            summary[name]["training"].append(returns[-window:].mean())
            summary[name]["greedy"].append(greedy_return)
            summary[name]["highest_row"].append(highest_row)
            print(f"{name:9s} seed {seed}: training {returns[-window:].mean():8.3f}  "
                  f"greedy {greedy_return:6.1f}  highest_row {highest_row}")

    # 4. average over seeds, in the same key order as the results.json template
    out = {}
    for name in algorithms:
        out[f"{name}_training_return"] = round(float(np.mean(summary[name]["training"])), 3)
    for name in algorithms:
        out[f"{name}_greedy_return"] = round(float(np.mean(summary[name]["greedy"])), 3)
    for name in algorithms:
        out[f"{name}_highest_row"] = round(float(np.mean(summary[name]["highest_row"])), 3)
    return out

    




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pa1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    raw_dir = pathlib.Path(cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "gridworld": run_dynamic_programming(cfg),
        "cliff": {**run_cliff_dp(cfg),
                  **run_td_control(cfg, cfg["seeds"], raw_dir)},
    }
    out = pathlib.Path(cfg["output"]["results_json"])
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
