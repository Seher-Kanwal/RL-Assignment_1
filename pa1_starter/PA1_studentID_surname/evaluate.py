"""
PA1 — evaluation and figures.

    python evaluate.py

Reads the raw per-seed output that train.py wrote to results/ and produces
the three required figures in figures/:

    figures/gridworld_values_policy.png   heatmap of V* with greedy arrows
    figures/cliff_training_curves.png     SARSA vs Q-learning, mean ± std over seeds
    figures/cliff_greedy_paths.png        each algorithm's greedy path on the grid

Evaluation is GREEDY and does not learn: act on argmax Q, never epsilon.
Keep this separate from training — the specification grades that separation.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")           # works on headless machines
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import agent
from pa1_envs import RoomsGridWorld, CliffWalk

FIG = pathlib.Path("figures")
RAW = pathlib.Path("results")
CONFIG = pathlib.Path("configs/pa1.yaml")                # the seeds to load come from here

ALGOS = {"sarsa": "SARSA", "qlearning": "Q-learning"}
COLORS = {"sarsa": "#2a78d6", "qlearning": "#eb6834"}   # blue, orange (CVD-checked)
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ARROWS = ["↑", "→", "↓", "←"]                          # 0=up 1=right 2=down 3=left
SMOOTH = 50                                             # moving-average window (episodes)

# one-hue sequential ramp (light = low value, dark = high value)
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def greedy_rollout(env, Q: np.ndarray, max_steps: int = 400):
    """Follow argmax Q from the start. Returns (undiscounted_return, path_of_states)."""
    s, _ = env.reset()
    total = 0.0
    path = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))                # greedy: no epsilon, no learning
        s, reward, terminated, _, _ = env.step(a)
        total += reward
        path.append(s)
        if terminated:
            break
    return total, path


def plot_gridworld(V: np.ndarray, policy: np.ndarray) -> None:
    """Heatmap of V with an arrow per cell for the greedy action; walls blank."""
    env = RoomsGridWorld()
    grid = V.reshape(env.n_rows, env.n_cols).astype(float)
    special = np.array([[env.grid[r][c] for c in range(env.n_cols)] for r in range(env.n_rows)])
    grid = np.ma.masked_where(np.isin(special, ["#", "G", "H"]), grid)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = VALUE_CMAP.copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(grid, cmap=cmap)

    lo, hi = float(grid.min()), float(grid.max())
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = env.to_s(r, c)
            ch = special[r, c]
            if ch == "#":
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#52514e"))
            elif ch in "GH":
                fill = "#0ca30c" if ch == "G" else "#d03b3b"
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color=fill))
                label = "G\n+10" if ch == "G" else "H\n−8"
                ax.text(c, r, label, ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
            else:
                # dark text on light cells, white text on dark cells
                txt = "white" if (V[s] - lo) / (hi - lo) > 0.55 else INK
                ax.text(c, r - 0.13, ARROWS[policy[s]], ha="center", va="center",
                        color=txt, fontsize=17)
                ax.text(c, r + 0.27, f"{V[s]:.2f}", ha="center", va="center",
                        color=txt, fontsize=8)
            if s == env.start_state:
                ax.text(c - .42, r - .42, "S", ha="left", va="top", color=txt,
                        fontsize=9, fontweight="bold")

    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xticks(np.arange(-.5, env.n_cols), minor=True)
    ax.set_yticks(np.arange(-.5, env.n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("V*(s)")
    cb.outline.set_visible(False)
    ax.set_title("RoomsGridWorld: optimal values V* and greedy policy (γ = 0.95)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "gridworld_values_policy.png", dpi=200)
    plt.close(fig)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w episodes (shorter at the start)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def plot_training_curves(per_seed: dict[str, list[np.ndarray]]) -> None:
    """per_seed["sarsa"] is a list of per-episode return arrays, one per seed.
    Plot mean and a ± std band for each algorithm on the same axes."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for name, label in ALGOS.items():
        runs = np.stack([moving_average(r, SMOOTH) for r in per_seed[name]])
        mean, std = runs.mean(axis=0), runs.std(axis=0)
        ep = np.arange(1, mean.size + 1)
        ax.fill_between(ep, mean - std, mean + std, color=COLORS[name], alpha=0.18, linewidth=0)
        ax.plot(ep, mean, color=COLORS[name], linewidth=2, label=label)
        # same number as *_training_return in results.json (raw, last 500 episodes)
        last500 = np.mean([r[-500:].mean() for r in per_seed[name]])
        ax.text(ep[-1] + 30, mean[-1], f"{label}\nlast 500: {last500:.1f}", color=INK,
                va="center", fontsize=9)

    ax.axhline(-11, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(40, -9.5, "optimal path return (−11)", color=INK_MUTED, fontsize=8)
    ax.set_ylim(-100, 0)
    ax.set_xlim(0, mean.size)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"training return ({SMOOTH}-episode moving average)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    n = len(per_seed["sarsa"])
    ax.set_title(f"CliffWalk training return, ε = 0.1: mean ± 1 std over {n} seeds",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cliff_training_curves.png", dpi=200)
    plt.close(fig)


def plot_greedy_paths(paths: dict[str, list[list[int]]]) -> None:
    """Draw each algorithm's greedy path on the 5 x 10 CliffWalk grid.
    paths["sarsa"] holds one path per seed; seeds are drawn slightly offset."""
    env = CliffWalk()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.4))
    for ax, (name, label) in zip(axes, ALGOS.items()):
        # grid cells
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                ch = env.grid[r][c]
                fill = {"C": "#f6d0cf", "G": "#0ca30c", "S": "#e8e7e3"}.get(ch, "#fcfcfb")
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=fill,
                                       edgecolor=GRID, linewidth=1))
        for c in range(1, env.n_cols - 1):
            ax.text(c, 4, "cliff", ha="center", va="center", color="#d03b3b", fontsize=8)
        ax.text(0, 4, "S", ha="center", va="center", color=INK, fontweight="bold")
        ax.text(env.n_cols - 1, 4, "G", ha="center", va="center", color="white", fontweight="bold")

        # one line per seed, offset so identical paths stay visible
        seed_paths = paths[name]
        offsets = np.linspace(-0.2, 0.2, len(seed_paths))
        for path, off in zip(seed_paths, offsets):
            rows = np.array([s // env.n_cols for s in path], dtype=float) + off
            cols = np.array([s % env.n_cols for s in path], dtype=float) + off
            ax.plot(cols, rows, color=COLORS[name], linewidth=2, alpha=0.85,
                    solid_capstyle="round", solid_joinstyle="round")
        highest = np.mean([min(s // env.n_cols for s in p) for p in seed_paths])
        length = np.mean([len(p) - 1 for p in seed_paths])
        ax.set_title(f"{label}: greedy path, {len(seed_paths)} seeds "
                     f"(mean highest row {highest:.1f}, mean length {length:.1f} steps)",
                     loc="left", fontsize=10.5)
        ax.set_xlim(-.5, env.n_cols - .5)
        ax.set_ylim(env.n_rows - .5, -.5)          # row 0 at the top
        ax.set_aspect("equal")
        ax.set_xticks(range(env.n_cols))
        ax.set_yticks(range(env.n_rows))
        ax.set_ylabel("row")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
    axes[-1].set_xlabel("column")
    fig.tight_layout()
    fig.savefig(FIG / "cliff_greedy_paths.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)

    # Figure 1: recompute V* and the greedy policy from the model (fast, deterministic)
    grid_env = RoomsGridWorld()
    V_star, _ = agent.value_iteration(grid_env)
    plot_gridworld(V_star, agent.greedy_policy(grid_env, V_star))

    # Figures 2 and 3: load the raw per-seed output written by train.py
    cliff = CliffWalk()
    per_seed = {name: [] for name in ALGOS}
    paths = {name: [] for name in ALGOS}
    # load exactly the configured seeds, so stale files from other seed lists are ignored
    seeds = yaml.safe_load(CONFIG.read_text())["seeds"]
    for name in ALGOS:
        files = [RAW / f"{name}_seed{s}.npz" for s in seeds]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"missing {missing} in {RAW}/ — run train.py first")
        for f in files:
            data = np.load(f)
            per_seed[name].append(data["returns"])
            ret, path = greedy_rollout(cliff, data["Q"])
            paths[name].append(path)
            print(f"{f.name}: greedy return {ret:.1f}, highest row "
                  f"{min(s // cliff.n_cols for s in path)}, {len(path) - 1} steps")

    plot_training_curves(per_seed)
    plot_greedy_paths(paths)
    print(f"wrote 3 figures to {FIG}/")


if __name__ == "__main__":
    main()
