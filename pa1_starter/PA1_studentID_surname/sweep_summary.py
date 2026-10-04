"""
PA1 — epsilon sweep summary (report question 3).

    for e in 0.01 0.05 0.1 0.2 0.3:  python train.py --config configs/eps_<e>.yaml
    python sweep_summary.py

Reads results_sweep/eps_*/{sarsa,qlearning}_seed*.npz written by train.py and
produces:

    results_sweep/summary.csv        one row per (epsilon, algorithm): mean ± std over seeds
    figures/epsilon_sweep.png        training return, greedy return, highest row vs epsilon

Only epsilon differs between the runs; every other setting is configs/pa1.yaml.

The .npz files do not record cliff falls, so each run is replayed once with
agent.sarsa / agent.q_learning in a CliffWalk that counts cliff transitions
(reward -75). The replayed returns are asserted to equal the saved ones, so the
counts belong to exactly the saved runs.
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import agent
from pa1_envs import CliffWalk

SWEEP = pathlib.Path("results_sweep")
FIG = pathlib.Path("figures")
ALGOS = {"sarsa": "SARSA", "qlearning": "Q-learning"}
COLORS = {"sarsa": "#2a78d6", "qlearning": "#eb6834"}   # same as evaluate.py
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
WINDOW = 500                                            # training_return_window
FAIL_Y = -20.5                                          # where failed greedy runs are drawn

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

METRICS = [  # (key, panel title, y label)
    ("training", "Training return (last 500 episodes)", "undiscounted return"),
    ("greedy", "Greedy return from S", "undiscounted return"),
    ("highest_row", "Highest row on greedy path", "row (0 = top, 4 = cliff)"),
]


class CountingCliffWalk(CliffWalk):
    """CliffWalk that records the number of cliff transitions in each episode."""

    def __init__(self):
        super().__init__()
        self.falls = []

    def reset(self, *args, **kwargs):
        self.falls.append(0)
        return super().reset(*args, **kwargs)

    def step(self, action):
        out = super().step(action)
        if out[1] == self.cliff_reward:
            self.falls[-1] += 1
        return out


def cliff_falls(run_dir: pathlib.Path, algo: str, seed: int, saved_returns: np.ndarray) -> np.ndarray:
    """Per-episode cliff-fall counts of one saved run, by replaying its training."""
    td = yaml.safe_load((pathlib.Path("configs") / f"{run_dir.name}.yaml").read_text())["td_control"]
    env = CountingCliffWalk()
    train_fn = agent.sarsa if algo == "sarsa" else agent.q_learning
    _, returns = train_fn(env, td["alpha"], td["epsilon"], td["episodes"],
                          td["max_steps_per_episode"], np.random.default_rng(seed))
    if not np.array_equal(np.asarray(returns), saved_returns):
        raise RuntimeError(f"replay of {run_dir.name}/{algo} seed {seed} does not match the saved run")
    return np.array(env.falls)


def load_sweep() -> dict[float, dict[str, dict[str, np.ndarray]]]:
    """{epsilon: {algo: {"training": per-seed array, "greedy": ..., "highest_row": ..., "falls": ...}}}"""
    out = {}
    for d in sorted(SWEEP.glob("eps_*"), key=lambda p: float(p.name[4:])):
        eps = float(d.name[4:])
        out[eps] = {}
        for algo in ALGOS:
            files = sorted(d.glob(f"{algo}_seed*.npz"))
            if not files:
                raise FileNotFoundError(f"no {algo} runs in {d} — run train.py with configs/{d.name}.yaml")
            data = [np.load(f) for f in files]
            falls = [cliff_falls(d, algo, int(z["seed"]), z["returns"]) for z in data]
            out[eps][algo] = {
                "training": np.array([z["returns"][-WINDOW:].mean() for z in data]),
                "greedy": np.array([float(z["greedy_return"]) for z in data]),
                "highest_row": np.array([float(z["highest_row"]) for z in data]),
                # share of the last 500 episodes with at least one actual cliff transition
                "falls": np.array([(f[-WINDOW:] > 0).mean() for f in falls]),
                # greedy path ends at G (state 49); False = greedy policy loops until the cap
                "reached": np.array([int(z["greedy_path"][-1]) == 49 for z in data]),
                "n_seeds": len(files),
            }
    return out


def write_csv(sweep) -> None:
    path = SWEEP / "summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "algorithm", "n_seeds",
                    "training_return_mean", "training_return_std",
                    "greedy_return_mean", "greedy_return_std",
                    "highest_row_mean", "highest_row_std",
                    "frac_last500_episodes_with_cliff_fall",
                    "seeds_greedy_reaches_goal", "greedy_return_per_seed", "highest_row_per_seed"])
        for eps, algos in sweep.items():
            for algo, m in algos.items():
                w.writerow([eps, algo, m["n_seeds"],
                            round(m["training"].mean(), 3), round(m["training"].std(), 3),
                            round(m["greedy"].mean(), 3), round(m["greedy"].std(), 3),
                            round(m["highest_row"].mean(), 3), round(m["highest_row"].std(), 3),
                            round(m["falls"].mean(), 4),
                            f"{int(m['reached'].sum())}/{m['n_seeds']}",
                            " ".join(f"{g:g}" for g in m["greedy"]),
                            " ".join(f"{h:g}" for h in m["highest_row"])])
    print(f"wrote {path}")


def plot(sweep) -> None:
    eps = np.array(list(sweep))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    pos = np.arange(len(eps))                      # evenly spaced categories
    shift = {"sarsa": -0.12, "qlearning": 0.12}    # side by side within each epsilon
    for ax, (key, title, ylabel) in zip(axes, METRICS):
        for algo, label in ALGOS.items():
            x = pos + shift[algo]
            if key == "training":
                # well-behaved: mean ± std over seeds
                mean = np.array([sweep[e][algo][key].mean() for e in eps])
                std = np.array([sweep[e][algo][key].std() for e in eps])
                ax.errorbar(x, mean, yerr=std, color=COLORS[algo], linewidth=2,
                            marker="o", markersize=6, capsize=3, label=label)
                continue
            # greedy metrics: one dot per seed + median line, failed seeds marked
            med = np.array([np.median(sweep[e][algo][key]) for e in eps])
            ax.plot(x, med, color=COLORS[algo], linewidth=2, label=label)
            for xi, e in zip(x, eps):
                vals, ok = sweep[e][algo][key].copy(), sweep[e][algo]["reached"]
                if key == "greedy":
                    vals[~ok] = FAIL_Y              # -400 drawn at the panel's bottom edge
                jit = np.linspace(-0.07, 0.07, len(vals))
                ax.scatter(xi + jit[ok], vals[ok], s=28, color=COLORS[algo],
                           edgecolor="white", linewidth=1, zorder=3)
                ax.scatter(xi + jit[~ok], vals[~ok], s=46, marker="X", color=COLORS[algo],
                           edgecolor="white", linewidth=0.8, zorder=3)
        if key == "greedy":
            ax.axhline(-11, color=INK_MUTED, linewidth=1, linestyle="--")
            ax.text(pos[0] - 0.35, -10.6, "optimal (−11)", color=INK_MUTED, fontsize=8)
            ax.set_ylim(FAIL_Y - 0.8, -8)
            ax.axhline(FAIL_Y + 0.6, color=GRID, linewidth=1, linestyle=":")
            ax.text(pos[0] - 0.35, FAIL_Y + 0.25, "✕ = never reaches G (return −400, not to scale)",
                    color=INK_MUTED, fontsize=8)
        if key == "highest_row":
            ax.set_ylim(4.3, -0.3)                 # row 0 at the top, like the grid
            ax.set_yticks(range(5))
            ax.set_yticklabels(["0 (top)", "1", "2", "3", "4 (cliff)"])
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{e:g}" for e in eps])
        ax.set_xlabel("ε (exploration rate)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower left")
    n = next(iter(sweep.values()))["sarsa"]["n_seeds"]
    fig.suptitle(f"CliffWalk ε sweep, {n} seeds (α = 0.5, 3000 episodes). Left: mean ± 1 std. "
                 "Middle and right: one marker per seed, line = median",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "epsilon_sweep.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / 'epsilon_sweep.png'}")


def main():
    sweep = load_sweep()
    print(f"{'eps':>5} {'algorithm':>10} | {'training':>16} | {'greedy':>14} | {'highest row':>12} | falls/ep")
    for e, algos in sweep.items():
        for algo, m in algos.items():
            print(f"{e:>5g} {ALGOS[algo]:>10} | {m['training'].mean():7.2f} ± {m['training'].std():5.2f} | "
                  f"{m['greedy'].mean():6.1f} ± {m['greedy'].std():4.1f} | "
                  f"{m['highest_row'].mean():5.1f} ± {m['highest_row'].std():3.1f} | {m['falls'].mean():.3f}")
    write_csv(sweep)
    plot(sweep)


if __name__ == "__main__":
    main()
