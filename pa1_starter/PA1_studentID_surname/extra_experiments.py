"""
PA1 — supplementary experiments for the report (not part of the graded results.json).

    python extra_experiments.py

Every experiment uses the same environments, the same update rules and the same
agent.epsilon_greedy as the graded runs. Training is re-implemented here only so
that quantities agent.sarsa / agent.q_learning do not return (per-episode greedy
snapshots, a per-episode epsilon schedule) can be recorded; check_equivalence()
asserts that this loop reproduces agent.sarsa / agent.q_learning bit for bit.

    A  policy iteration vs value iteration convergence trace      (report Q1)
    B  learned value maps max_a Q(s, a), SARSA vs Q-learning      (report Q2)
    C  epsilon sweep with 20 seeds + greedy-policy stability      (report Q3)
    D  SARSA at epsilon = 0.01 for 20000 episodes                 (report Q3)
    E  cross-evaluation: learned policies executed at various ε   (report Q4)
    F  decaying epsilon (0.1 -> 0) for SARSA and Q-learning       (report Q4)

Outputs: results_extra/*.json and figures/extra_*.png
"""
from __future__ import annotations

import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import agent
from pa1_envs import RoomsGridWorld, CliffWalk

FIG = pathlib.Path("figures")
OUT = pathlib.Path("results_extra")
RAW = pathlib.Path("results")

ALPHA, EPISODES, MAX_STEPS, WINDOW = 0.5, 3000, 400, 500
SWEEP_EPS = [0.01, 0.05, 0.1, 0.2, 0.3]
SWEEP_SEEDS = list(range(20))
ALGOS = {"sarsa": "SARSA", "qlearning": "Q-learning"}
COLORS = {"sarsa": "#2a78d6", "qlearning": "#eb6834", "optimal": "#52514e"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ARROWS = ["↑", "→", "↓", "←"]
_CACHE = {}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ------------------------------------------------------------------ helpers
def greedy_path(env, Q, max_steps=MAX_STEPS):
    """Greedy rollout using the model (no learning). Returns (return, path, reached)."""
    s, total, path = env.start_state, 0.0, [env.start_state]
    for _ in range(max_steps):
        _, s, r, done = env.P[s][int(np.argmax(Q[s]))][0]
        total += r
        path.append(s)
        if done:
            return total, path, True
    return total, path, False


def train(env, algo, eps_schedule, episodes, seed, snapshot_from=None):
    """SARSA / Q-learning exactly as in agent.py, with a per-episode epsilon.

    Returns dict with Q, returns, and (if snapshot_from is set) for every episode
    >= snapshot_from whether the greedy policy at that moment reaches G.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns, snaps, rows = [], [], []
    for ep in range(episodes):
        eps = eps_schedule(ep)
        s, _ = env.reset()
        G = 0.0
        if algo == "sarsa":
            a = agent.epsilon_greedy(Q, s, eps, rng)
            for _ in range(MAX_STEPS):
                s2, r, term, _, _ = env.step(a)
                G += r
                a2 = agent.epsilon_greedy(Q, s2, eps, rng)
                target = r if term else r + env.gamma * Q[s2, a2]
                Q[s, a] += ALPHA * (target - Q[s, a])
                s, a = s2, a2
                if term:
                    break
        else:
            for _ in range(MAX_STEPS):
                a = agent.epsilon_greedy(Q, s, eps, rng)
                s2, r, term, _, _ = env.step(a)
                G += r
                target = r if term else r + env.gamma * np.max(Q[s2])
                Q[s, a] += ALPHA * (target - Q[s, a])
                s = s2
                if term:
                    break
        returns.append(G)
        if snapshot_from is not None and ep >= snapshot_from:
            _, path, ok = greedy_path(env, Q)
            snaps.append(ok)
            rows.append(min(x // env.n_cols for x in path))
    return {"Q": Q, "returns": np.array(returns), "snap_ok": np.array(snaps),
            "snap_row": np.array(rows)}


def check_equivalence():
    env = CliffWalk()
    for seed in range(5):
        for algo, fn in (("sarsa", agent.sarsa), ("qlearning", agent.q_learning)):
            Q_ref, R_ref = fn(env, ALPHA, 0.1, EPISODES, MAX_STEPS, np.random.default_rng(seed))
            mine = train(env, algo, lambda ep: 0.1, EPISODES, seed)
            assert np.array_equal(mine["Q"], Q_ref) and np.array_equal(mine["returns"], R_ref), \
                f"instrumented {algo} differs from agent.py (seed {seed})"
    print("equivalence check: instrumented training == agent.sarsa / agent.q_learning (5 seeds, both)")


def smooth(x, w=50):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - w)
    return (c[idx] - c[lo]) / (idx - lo)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)


# ------------------------------------------------------------------ A: PI vs VI
def experiment_A():
    env = RoomsGridWorld()
    NS = np.array([[env.P[s][a][0][1] for a in range(4)] for s in range(env.n_states)])
    R = np.array([[env.P[s][a][0][2] for a in range(4)] for s in range(env.n_states)])
    D = np.array([[env.P[s][a][0][3] for a in range(4)] for s in range(env.n_states)], float)

    # value iteration trace (same synchronous backup as agent.value_iteration)
    V, vi_deltas = np.zeros(env.n_states), []
    while True:
        Vn = (R + env.gamma * V[NS] * (1 - D)).max(1)
        vi_deltas.append(float(np.abs(Vn - V).max()))
        V = Vn
        if vi_deltas[-1] < 1e-12:
            break
    V_vi, vi_sweeps = agent.value_iteration(env)
    assert vi_sweeps == len(vi_deltas)

    # policy iteration trace, using agent.policy_evaluation / greedy_policy per round
    policy, rounds = np.zeros(env.n_states, int), []
    pe_deltas = []                    # concatenated evaluation deltas across rounds
    while True:
        pi = np.eye(4)[policy]
        V, k = agent.policy_evaluation(env, pi)
        # delta trace for this round's evaluation (same backup, for the plot)
        W, d = np.zeros(env.n_states), []
        for _ in range(k):
            Wn = (pi * (R + env.gamma * W[NS] * (1 - D))).sum(1)
            d.append(float(np.abs(Wn - W).max()))
            W = Wn
        pe_deltas.append(d)
        new = agent.greedy_policy(env, V)
        changed = int((new != policy).sum())
        rounds.append({"round": len(rounds) + 1, "eval_sweeps": k, "V_start": float(V[35]),
                       "states_changed": changed})
        if changed == 0:
            break
        policy = new
    V_pi, p_pi, n_rounds = agent.policy_iteration(env)
    assert n_rounds == len(rounds)

    total_pi = sum(r["eval_sweeps"] for r in rounds)
    res = {"vi_sweeps": vi_sweeps, "vi_deltas": vi_deltas, "pi_rounds": rounds,
           "pi_total_eval_sweeps": total_pi, "pi_total_backups": total_pi + len(rounds),
           "same_policy": bool(np.array_equal(p_pi, agent.greedy_policy(env, V_vi))),
           "max_V_diff": float(np.abs(V_pi - V_vi).max())}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(pe_deltas)))
    for i, d in enumerate(pe_deltas):
        ax.plot(np.arange(1, len(d) + 1), np.maximum(d, 1e-16), color=shades[i], linewidth=1.4,
                label="PI evaluation, rounds 1–9 (light → dark)" if i == 0 else None)
    ax.plot(np.arange(1, len(vi_deltas) + 1), np.maximum(vi_deltas, 1e-16), color=COLORS["qlearning"],
            linewidth=2.4, marker="o", markersize=3.5, label=f"value iteration ({vi_sweeps} sweeps)")
    ax.axhline(1e-12, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(1.1, 2.5e-12, "θ = 1e-12", color=INK_MUTED, fontsize=8)
    ax.annotate("rounds 1–6: 540 sweeps each\n(error shrinks only ×0.95 per sweep)", xy=(450, 1e-9),
                xytext=(14, 1e-14), fontsize=8, color=INK_MUTED,
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e2)
    ax.set_xlabel("sweep number within one evaluation (log scale)")
    ax.set_ylabel("max |V_{k+1} − V_k|")
    ax.set_title(f"(a) PI: {total_pi:,} evaluation sweeps in total; VI: {vi_sweeps}", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    style_axis(ax)

    ax = axes[1]
    r_idx = [r["round"] for r in rounds]
    bars = ax.bar(r_idx, [r["eval_sweeps"] for r in rounds], color=COLORS["sarsa"], width=0.6)
    for b, r in zip(bars, rounds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 12, f"{r['eval_sweeps']}",
                ha="center", fontsize=8, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, -75, f"{r['states_changed']}", ha="center",
                fontsize=8, color=INK_MUTED)
    ax.text(0.2, -75, "Δπ:", ha="right", fontsize=8, color=INK_MUTED)
    ax.set_ylim(-110, 640)
    ax.set_xticks(r_idx)
    ax.set_xlabel("PI round  (Δπ = states whose action changed)")
    ax.set_ylabel("evaluation sweeps in the round")
    ax.set_title(f"(b) {len(rounds)} rounds; V*(S) reached in round "
                 f"{next(r['round'] for r in rounds if abs(r['V_start'] - V_vi[35]) < 1e-9)}",
                 loc="left", fontsize=10.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG / "extra_A_pi_vs_vi.png", dpi=200)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ B: value maps
def experiment_B():
    env = CliffWalk()
    V_dp, _ = agent.value_iteration(env)
    maps = {}
    for algo in ALGOS:
        Qs = [np.load(RAW / f"{algo}_seed{s}.npz")["Q"] for s in range(5)]
        maps[algo] = np.mean([Q.max(1) for Q in Qs], axis=0)
        maps[algo + "_policy"] = Qs[0].argmax(1)
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
    lo = min(maps["sarsa"].min(), maps["qlearning"].min(), V_dp.min())
    fig, axes = plt.subplots(1, 3, figsize=(15, 2.9))
    panels = [("sarsa", "(a) SARSA: max_a Q(s,a)", maps["sarsa"], maps["sarsa_policy"]),
              ("qlearning", "(b) Q-learning: max_a Q(s,a)", maps["qlearning"], maps["qlearning_policy"]),
              ("dp", "(c) DP optimum V*(s)", V_dp, agent.greedy_policy(env, V_dp))]
    for ax, (key, title, V, pol) in zip(axes, panels):
        grid = V.reshape(env.n_rows, env.n_cols).copy()
        mask = np.zeros_like(grid, bool)
        mask[4, 1:] = True                      # cliff + goal: not meaningful values
        im = ax.imshow(np.ma.masked_where(mask, grid), cmap=cmap, vmin=lo, vmax=0)
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                s = env.to_s(r, c)
                if r == 4 and 0 < c < 9:
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#f6d0cf"))
                    ax.text(c, r, "cliff", ha="center", va="center", color="#d03b3b", fontsize=7)
                elif s == env.goal_state:
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#0ca30c"))
                    ax.text(c, r, "G", ha="center", va="center", color="white", fontweight="bold")
                else:
                    t = "white" if (V[s] - lo) / (0 - lo) > 0.55 else INK
                    ax.text(c, r - 0.18, ARROWS[pol[s]], ha="center", va="center", color=t, fontsize=10)
                    ax.text(c, r + 0.25, f"{V[s]:.0f}", ha="center", va="center", color=t, fontsize=7)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("value")
    cb.outline.set_visible(False)
    fig.savefig(FIG / "extra_B_value_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    edge = [env.to_s(3, c) for c in range(1, 9)]           # row 3, above the cliff
    top = [env.to_s(0, c) for c in range(1, 9)]
    return {"mean_value_row3_cols1to8": {a: float(maps[a][edge].mean()) for a in ALGOS} | {"dp": float(V_dp[edge].mean())},
            "mean_value_row0_cols1to8": {a: float(maps[a][top].mean()) for a in ALGOS} | {"dp": float(V_dp[top].mean())},
            "value_at_S": {a: float(maps[a][env.start_state]) for a in ALGOS} | {"dp": float(V_dp[env.start_state])}}


# ------------------------------------------------------------------ C: 20-seed sweep + stability
def experiment_C():
    env = CliffWalk()
    res, curves = {}, {}
    for eps in SWEEP_EPS:
        res[str(eps)] = {}
        for algo in ALGOS:
            rows = []
            for seed in SWEEP_SEEDS:
                out = train(env, algo, lambda ep, e=eps: e, EPISODES, seed, snapshot_from=EPISODES - WINDOW)
                ret, path, ok = greedy_path(env, out["Q"])
                rows.append({
                    "training": float(out["returns"][-WINDOW:].mean()),
                    "falls": float((out["returns"][-WINDOW:] < -60).mean()),
                    "greedy": ret, "reached": ok,
                    "highest_row": min(x // env.n_cols for x in path),
                    "snap_ok": float(out["snap_ok"].mean()),
                    "snap_row_mean": float(out["snap_row"][out["snap_ok"]].mean()) if out["snap_ok"].any() else np.nan,
                })
                curves.setdefault((eps, algo), []).append(smooth(out["returns"]))
            agg = {k: [r[k] for r in rows] for k in rows[0]}
            res[str(eps)][algo] = {
                "training_mean": float(np.mean(agg["training"])), "training_std": float(np.std(agg["training"])),
                "falls_mean": float(np.mean(agg["falls"])),
                "greedy_reached": int(np.sum(agg["reached"])),
                "greedy_median": float(np.median(agg["greedy"])),
                "greedy_mean_reached": float(np.mean([g for g, ok in zip(agg["greedy"], agg["reached"]) if ok])),
                "highest_row_median": float(np.median(agg["highest_row"])),
                "highest_row_mean": float(np.mean(agg["highest_row"])),
                "snap_ok_mean": float(np.mean(agg["snap_ok"])), "snap_ok_std": float(np.std(agg["snap_ok"])),
                "per_seed": rows,
            }
            print(f"  C eps={eps:<5} {algo:9s} training {res[str(eps)][algo]['training_mean']:7.2f}  "
                  f"greedy reached {res[str(eps)][algo]['greedy_reached']}/20  stability {res[str(eps)][algo]['snap_ok_mean']:.2f}")

    # figure C1: four panels vs epsilon
    pos = np.arange(len(SWEEP_EPS))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))
    for algo, label in ALGOS.items():
        r = [res[str(e)][algo] for e in SWEEP_EPS]
        sh = -0.06 if algo == "sarsa" else 0.06
        axes[0].errorbar(pos + sh, [x["training_mean"] for x in r], yerr=[x["training_std"] for x in r],
                         color=COLORS[algo], marker="o", linewidth=2, capsize=3, label=label)
        axes[1].plot(pos + sh, [100 * x["falls_mean"] for x in r], color=COLORS[algo], marker="o", linewidth=2)
        axes[2].plot(pos + sh, [x["greedy_mean_reached"] for x in r], color=COLORS[algo], marker="o", linewidth=2)
        for p, x in zip(pos + sh, r):
            if x["greedy_reached"] < len(SWEEP_SEEDS):
                axes[2].annotate(f"{len(SWEEP_SEEDS) - x['greedy_reached']}/20 fail", (p, x["greedy_mean_reached"]),
                                 textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5,
                                 color=COLORS[algo])
        axes[3].errorbar(pos + sh, [100 * x["snap_ok_mean"] for x in r], yerr=[100 * x["snap_ok_std"] for x in r],
                         color=COLORS[algo], marker="o", linewidth=2, capsize=3)
    axes[2].axhline(-11, color=INK_MUTED, linestyle="--", linewidth=1)
    axes[2].text(pos[0] - 0.2, -11.6, "optimal −11", color=INK_MUTED, fontsize=8, va="top")
    titles = ["(a) Training return, last 500 ep.", "(b) Episodes with a cliff fall (%)",
              "(c) Final greedy return (runs reaching G)", "(d) Greedy policy reaches G (% of last 500)"]
    ylabels = ["undiscounted return", "% of last 500 episodes", "undiscounted return", "% of snapshots"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_xticks(pos); ax.set_xticklabels([f"{e:g}" for e in SWEEP_EPS])
        ax.set_xlabel("ε"); ax.set_ylabel(yl); ax.set_title(t, loc="left", fontsize=10.5)
        style_axis(ax)
    axes[2].set_ylim(-19, -10)
    axes[3].set_ylim(0, 105)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "extra_C_epsilon_sweep_20seeds.png", dpi=200)
    plt.close(fig)

    # figure C2: learning curves per epsilon (small multiples)
    fig, axes = plt.subplots(1, len(SWEEP_EPS), figsize=(16, 3.0), sharey=True)
    for ax, eps in zip(axes, SWEEP_EPS):
        for algo, label in ALGOS.items():
            c = np.stack(curves[(eps, algo)])
            m = c.mean(0)
            ax.plot(np.arange(1, len(m) + 1), m, color=COLORS[algo], linewidth=1.6, label=label)
        ax.axhline(-11, color=INK_MUTED, linestyle="--", linewidth=0.8)
        ax.set_title(f"ε = {eps:g}", loc="left", fontsize=10.5)
        ax.set_ylim(-120, 0); ax.set_xlabel("episode")
        style_axis(ax)
    axes[0].set_ylabel("training return (50-ep. avg)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "extra_C_learning_curves_by_eps.png", dpi=200)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ D: long SARSA run at eps = 0.01
def experiment_D(episodes=20000):
    env = CliffWalk()
    res = {}
    traces = []
    for seed in range(5):
        out = train(env, "sarsa", lambda ep: 0.01, episodes, seed, snapshot_from=0)
        rows = out["snap_row"].astype(float)
        rows[~out["snap_ok"]] = np.nan
        traces.append(rows)
        ret, path, ok = greedy_path(env, out["Q"])
        res[seed] = {"final_greedy": ret, "final_highest_row": min(x // env.n_cols for x in path),
                     "reached": ok, "training_last500": float(out["returns"][-WINDOW:].mean())}
        print(f"  D seed {seed}: after {episodes} episodes greedy {ret}, highest row {res[seed]['final_highest_row']}")
    return {"episodes": episodes, "per_seed": res,
            "trace_rows_every_100": [np.round(t[::100], 2).tolist() for t in traces]}


# ------------------------------------------------------------------ E: cross-evaluation
def run_policy(env, Q, eps, n_episodes, seed):
    """Execute argmax-Q policy with epsilon-random actions, NO learning."""
    rng = np.random.default_rng(seed)
    rets, falls = [], []
    for _ in range(n_episodes):
        s, _ = env.reset(); G = 0.0; f = 0
        for _ in range(MAX_STEPS):
            a = agent.epsilon_greedy(Q, s, eps, rng)
            s, r, term, _, _ = env.step(a)
            G += r; f += r == -75
            if term:
                break
        rets.append(G); falls.append(f)
    return float(np.mean(rets)), float(np.mean(falls))


def experiment_E(n_episodes=2000):
    env = CliffWalk()
    exec_eps = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    V_dp, _ = agent.value_iteration(env)
    NS = np.array([[env.P[s][a][0][1] for a in range(4)] for s in range(env.n_states)])
    R = np.array([[env.P[s][a][0][2] for a in range(4)] for s in range(env.n_states)])
    D = np.array([[env.P[s][a][0][3] for a in range(4)] for s in range(env.n_states)], float)
    Q_dp = R + env.gamma * V_dp[NS] * (1 - D)
    res = {}
    for algo in ALGOS:
        Qs = [np.load(RAW / f"{algo}_seed{s}.npz")["Q"] for s in range(5)]   # trained at eps = 0.1
        res[algo] = {}
        for e in exec_eps:
            vals = [run_policy(env, Q, e, n_episodes, 1000 + i) for i, Q in enumerate(Qs)]
            res[algo][str(e)] = {"return": float(np.mean([v[0] for v in vals])),
                                 "return_std": float(np.std([v[0] for v in vals])),
                                 "falls_per_episode": float(np.mean([v[1] for v in vals]))}
    res["dp_optimal"] = {str(e): dict(zip(["return", "falls_per_episode"],
                                          run_policy(env, Q_dp, e, n_episodes, 999))) for e in exec_eps}
    for algo in list(ALGOS) + ["dp_optimal"]:
        print(f"  E {algo:10s} " + "  ".join(f"ε={e:g}: {res[algo][str(e)]['return']:.1f}" for e in exec_eps))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for algo, label in ALGOS.items():
        y = [res[algo][str(e)]["return"] for e in exec_eps]
        sd = [res[algo][str(e)]["return_std"] for e in exec_eps]
        axes[0].errorbar(exec_eps, y, yerr=sd, color=COLORS[algo], marker="o", linewidth=2, capsize=3,
                         label=f"{label} policy (trained at ε = 0.1)")
        axes[1].plot(exec_eps, [res[algo][str(e)]["falls_per_episode"] for e in exec_eps],
                     color=COLORS[algo], marker="o", linewidth=2, label=label)
    axes[0].plot(exec_eps, [res["dp_optimal"][str(e)]["return"] for e in exec_eps], color=COLORS["optimal"],
                 linestyle="--", linewidth=1.3, label="DP-optimal policy")
    for ax in axes:
        ax.axvline(0.1, color=GRID, linewidth=6, zorder=0)
        ax.text(0.1, ax.get_ylim()[1] if ax is axes[1] else -5, "training ε", ha="center", va="top",
                fontsize=8, color=INK_MUTED)
        ax.set_xlabel("ε used when EXECUTING the learned policy (no learning)")
        style_axis(ax)
    axes[0].set_ylabel("mean undiscounted return"); axes[0].set_ylim(-110, 0)
    axes[1].set_ylabel("cliff falls per episode")
    axes[0].set_title("(a) Return of each learned policy vs execution ε", loc="left", fontsize=10.5)
    axes[1].set_title("(b) Cliff falls per episode", loc="left", fontsize=10.5)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "extra_E_cross_evaluation.png", dpi=200)
    plt.close(fig)
    return res


# ------------------------------------------------------------------ F: decaying epsilon
def experiment_F(episodes=3000):
    env = CliffWalk()
    schedules = {
        "constant 0.1": lambda ep: 0.1,
        "linear 0.1 → 0": lambda ep: 0.1 * max(0.0, 1 - ep / (0.8 * episodes)),
    }
    res, curves = {}, {}
    for name, sched in schedules.items():
        res[name] = {}
        for algo in ALGOS:
            rows = []
            for seed in range(20):
                out = train(env, algo, sched, episodes, seed, snapshot_from=episodes - WINDOW)
                ret, path, ok = greedy_path(env, out["Q"])
                rows.append((ret, min(x // env.n_cols for x in path), ok, out["returns"][-WINDOW:].mean(),
                             out["snap_ok"].mean()))
                curves.setdefault((name, algo), []).append(smooth(out["returns"]))
            r = np.array(rows, dtype=float)
            res[name][algo] = {"greedy_mean": float(r[:, 0].mean()), "highest_row_mean": float(r[:, 1].mean()),
                               "reached": int(r[:, 2].sum()), "training_last500": float(r[:, 3].mean()),
                               "snap_ok_mean": float(r[:, 4].mean()),
                               "highest_row_counts": {int(k): int(v) for k, v in zip(*np.unique(r[:, 1], return_counts=True))},
                               "per_seed_greedy": r[:, 0].tolist(), "per_seed_row": r[:, 1].tolist(),
                               "per_seed_reached": r[:, 2].astype(bool).tolist()}
            print(f"  F {name:15s} {algo:9s} greedy {res[name][algo]['greedy_mean']:6.2f}  highest row "
                  f"{res[name][algo]['highest_row_mean']:.2f}  training(last500) {res[name][algo]['training_last500']:6.2f}"
                  f"  reached {res[name][algo]['reached']}/20  stability {res[name][algo]['snap_ok_mean']:.2f}")

    _CACHE['F_curves'] = curves
    return res


def figure_FD(resF, resD):
    """Can SARSA reach the optimal (cliff-edge) path? Decaying epsilon and a long run."""
    curves = _CACHE["F_curves"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1.3]})
    ax = axes[0]
    styles = {"constant 0.1": "-", "linear 0.1 → 0": (0, (4, 2))}
    for (name, algo), c in curves.items():
        m = np.stack(c).mean(0)
        ax.plot(np.arange(1, len(m) + 1), m, color=COLORS[algo], linestyle=styles[name], linewidth=1.6,
                label=f"{ALGOS[algo]}, ε {name}")
    ax.axhline(-11, color=INK_MUTED, linestyle=":", linewidth=1)
    ax.text(30, -10.2, "optimal −11", fontsize=8, color=INK_MUTED)
    ax.axvline(2400, color=GRID, linewidth=1)
    ax.text(2420, -57, "ε = 0 from here\n(decaying runs)", fontsize=8, color=INK_MUTED)
    ax.set_ylim(-60, -5); ax.set_xlabel("episode"); ax.set_ylabel("training return (50-ep. avg)")
    ax.set_title("(a) Training return, constant vs decaying ε (20 seeds)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    style_axis(ax)

    ax = axes[1]
    names = list(resF)
    for j, name in enumerate(names):
        for i, algo in enumerate(ALGOS):
            r = resF[name][algo]
            x0 = j + (i - 0.5) * 0.36
            g = np.array(r["per_seed_greedy"]); ok = np.array(r["per_seed_reached"])
            jit = np.linspace(-0.12, 0.12, len(g))
            ax.scatter(x0 + jit[ok], g[ok], s=22, color=COLORS[algo], edgecolor="white", linewidth=0.7,
                       zorder=3, label=ALGOS[algo] if j == 0 else None)
            if (~ok).any():
                ax.scatter(x0 + jit[~ok], np.full((~ok).sum(), -19.4), s=40, marker="X",
                           color=COLORS[algo], zorder=3)
                ax.text(x0, -19.0, f"{(~ok).sum()} fail", ha="center", fontsize=7.5, color=COLORS[algo])
            ax.text(x0, -9.3, f"row {r['highest_row_mean']:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(-11, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"ε {n}" for n in names])
    ax.set_ylim(-20, -8.5); ax.set_ylabel("final greedy return")
    ax.set_title("(b) Final greedy return per seed", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style_axis(ax)

    ax = axes[2]
    pal = plt.cm.Blues(np.linspace(0.45, 0.95, len(resD["trace_rows_every_100"])))
    for k, (seed, tr) in enumerate(zip(resD["per_seed"], resD["trace_rows_every_100"])):
        tr = np.array(tr, dtype=float)
        x = np.arange(len(tr)) * 100
        ax.step(x, tr + (k - 2) * 0.05, where="post", color=pal[k], linewidth=1.4, label=f"seed {seed}")
    ax.axhline(3, color=COLORS["qlearning"], linestyle="--", linewidth=1)
    ax.text(200, 2.85, "optimal path / Q-learning (row 3)", fontsize=8, color=COLORS["qlearning"], va="bottom")
    ax.set_ylim(3.4, -0.4); ax.set_yticks(range(4))
    ax.set_xlabel("episode"); ax.set_ylabel("highest row of greedy path")
    ax.set_title(f"(c) SARSA, ε = 0.01, {resD['episodes']:,} episodes (gaps = no path to G)",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG / "extra_F_sarsa_optimal_path.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    check_equivalence()
    results = {}
    for name, fn in [("A", experiment_A), ("B", experiment_B), ("E", experiment_E),
                     ("F", experiment_F), ("D", experiment_D), ("C", experiment_C)]:
        t = time.time()
        print(f"experiment {name} ...")
        results[name] = fn()
        print(f"experiment {name} done in {time.time() - t:.1f}s")
    figure_FD(results["F"], results["D"])
    (OUT / "extra_results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {OUT / 'extra_results.json'} and figures/extra_*.png in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
