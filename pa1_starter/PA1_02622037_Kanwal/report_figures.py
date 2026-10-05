"""
PA1 — the figures used in report.pdf, drawn at their printed size (no retraining).

    python train.py --config configs/pa1.yaml   # results/*.npz
    python extra_experiments.py                 # results_extra/extra_results.json
    python report_figures.py                    # figures/report_fig*.png

Figures 1, 2 and 5 show the same data as the three required figures from
evaluate.py, re-laid out for the page; Figures 3, 4 and 6 redraw supplementary
results from extra_experiments.py.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import agent
from pa1_envs import RoomsGridWorld, CliffWalk

OUT = pathlib.Path("figures")
OUT.mkdir(exist_ok=True)
RAW = pathlib.Path("results")
EXTRA = json.loads(pathlib.Path("results_extra/extra_results.json").read_text(encoding="utf-8"))

ALGOS = {"sarsa": "SARSA", "qlearning": "Q-learning"}
C = {"sarsa": "#2a78d6", "qlearning": "#eb6834", "optimal": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ARROWS = ["↑", "→", "↓", "←"]
BLUES = LinearSegmentedColormap.from_list("b", ["#e6f0fc", "#a9cbf3", "#5b9be6", "#2a6fc4", "#123f7a"])
DPI = 300

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(algo):
    return [np.load(RAW / f"{algo}_seed{s}.npz") for s in range(5)]


def save(fig, name):
    fig.savefig(OUT / f"report_{name}", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# -------------------------------------------------------- Fig 1: RoomsGridWorld
def fig_gridworld():
    env = RoomsGridWorld()
    V, _ = agent.value_iteration(env)
    pol = agent.greedy_policy(env, V)
    fig, ax = plt.subplots(figsize=(3.2, 2.75))
    grid = V.reshape(env.n_rows, env.n_cols)
    im = ax.imshow(grid, cmap=BLUES, vmin=V.min(), vmax=10)
    for s in range(env.n_states):
        r, c = env.to_rc(s)
        ch = env.cell(s)
        if ch == "#":
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#5a5a5a"))
        elif ch == "G":
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#0ca30c"))
            ax.text(c, r, "G\n+10", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        elif ch == "H":
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#d03b3b"))
            ax.text(c, r, "H\n−8", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        else:
            t = "white" if V[s] > 5 else INK
            ax.text(c, r - 0.17, ARROWS[pol[s]], ha="center", va="center", color=t, fontsize=9)
            ax.text(c, r + 0.27, f"{V[s]:.2f}", ha="center", va="center", color=t, fontsize=5.8)
            if ch == "S":
                ax.text(c - .44, r - .44, "S", ha="left", va="top", color=t, fontsize=6, fontweight="bold")
    ax.set_xticks(range(env.n_cols)); ax.set_yticks(range(env.n_rows))
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("V*(s)", fontsize=7)
    save(fig, "fig1_gridworld.png")


# ------------------------------------------------------ CliffWalk drawing helper
def cliff_background(ax, env):
    ax.set_xlim(-.5, env.n_cols - .5); ax.set_ylim(env.n_rows - .5, -.5)
    ax.set_aspect("equal")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=GRID, lw=0.5))
    for c in range(1, env.n_cols - 1):
        ax.add_patch(Rectangle((c - .5, 3.5), 1, 1, color="#f6d0cf"))
    ax.add_patch(Rectangle((-.5, 3.5), 1, 1, color="#e8e8e8"))
    ax.add_patch(Rectangle((env.n_cols - 1.5, 3.5), 1, 1, color="#0ca30c"))
    ax.text(0, 4, "S", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(env.n_cols - 1, 4, "G", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(4.5, 4, "cliff  (−75, back to S)", ha="center", va="center", fontsize=7, color="#b02a2a")
    ax.set_xticks(range(env.n_cols)); ax.set_yticks(range(env.n_rows))
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ------------------------------------------------------ Fig 2: greedy paths
def fig_paths():
    env = CliffWalk()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.85))
    for ax, algo in zip(axes, ALGOS):
        cliff_background(ax, env)
        runs = load(algo)
        offsets = np.linspace(-0.2, 0.2, len(runs))
        for off, z in zip(offsets, runs):
            rc = np.array([env.to_rc(int(s)) for s in z["greedy_path"]], float)
            ax.plot(rc[:, 1] + off, rc[:, 0] + off, color=C[algo], lw=1.1, alpha=0.85)
        rows = [int(z["highest_row"]) for z in runs]
        rets = [float(z["greedy_return"]) for z in runs]
        ret = f"{np.mean(rets):.1f}".replace("-", "−")
        ax.set_title(f"{ALGOS[algo]}  (highest_row {np.mean(rows):.1f}, return {ret})", loc="left")
    save(fig, "fig2_paths.png")


# ------------------------------------------------------ Fig 3: learned value maps
def fig_value_maps():
    env = CliffWalk()
    V_dp, _ = agent.value_iteration(env)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.75))
    maps = {}
    for algo in ALGOS:
        Q = np.mean([z["Q"] for z in load(algo)], axis=0)   # seed-averaged Q
        maps[algo] = (Q.max(1), Q.argmax(1))
    lo = min(m[0].min() for m in maps.values())
    for ax, algo in zip(axes, ALGOS):
        V, pol = maps[algo]
        grid = V.reshape(env.n_rows, env.n_cols).copy()
        mask = np.zeros_like(grid, bool); mask[4, 1:] = True
        im = ax.imshow(np.ma.masked_where(mask, grid), cmap=BLUES.reversed(), vmin=lo, vmax=0)
        for c in range(1, env.n_cols - 1):
            ax.add_patch(Rectangle((c - .5, 3.5), 1, 1, color="#f6d0cf"))
        ax.add_patch(Rectangle((env.n_cols - 1.5, 3.5), 1, 1, color="#0ca30c"))
        ax.text(env.n_cols - 1, 4, "G", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        ax.text(4.5, 4, "cliff", ha="center", va="center", fontsize=7, color="#b02a2a")
        for s in range(env.n_states):
            r, c = env.to_rc(s)
            if r == 4 and c > 0:
                continue
            t = "white" if (V[s] - lo) / (0 - lo) < 0.45 else INK
            ax.text(c, r - 0.2, ARROWS[pol[s]], ha="center", va="center", color=t, fontsize=7)
            ax.text(c, r + 0.25, f"{V[s]:.0f}", ha="center", va="center", color=t, fontsize=5.8)
        ax.set_title(f"{ALGOS[algo]}: max_a Q̄(s,a) and argmax", loc="left")
        bare(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig3_value_maps.png")

    # numbers quoted in the text
    edge = [env.to_s(3, c) for c in range(1, 9)]
    top = [env.to_s(0, c) for c in range(1, 9)]
    for algo in ALGOS:
        V = maps[algo][0]
        print(f"  {algo:9s} row3 {V[edge].mean():6.2f}  row0 {V[top].mean():6.2f}  S {V[env.start_state]:6.2f}")
    vis = [s for s in range(env.n_states) if not (env.to_rc(s)[0] == 4 and env.to_rc(s)[1] > 0)]
    print(f"  max |V_Q - V*| over non-cliff states: {np.abs(maps['qlearning'][0] - V_dp)[vis].max():.2e}")
    print(f"  dp       row3 {V_dp[edge].mean():6.2f}  S {V_dp[env.start_state]:6.2f}")


# ------------------------------------------------------ Fig 4: epsilon sweep (20 seeds)
def fig_eps_sweep():
    res = EXTRA["C"]
    eps = sorted(res, key=float)
    x = np.arange(len(eps))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.85))
    for algo in ALGOS:
        tr = [res[e][algo]["training_mean"] for e in eps]
        trs = [res[e][algo]["training_std"] for e in eps]
        falls = [100 * np.mean([p["falls"] for p in res[e][algo]["per_seed"]]) for e in eps]
        fs = [100 * np.std([p["falls"] for p in res[e][algo]["per_seed"]]) for e in eps]
        ok = [100 * res[e][algo]["snap_ok_mean"] for e in eps]
        oks = [100 * res[e][algo]["snap_ok_std"] for e in eps]
        kw = dict(color=C[algo], marker="o", ms=3, lw=1.3, capsize=2, label=ALGOS[algo])
        axes[0].errorbar(x, tr, trs, **kw)
        axes[1].errorbar(x, falls, fs, **kw)
        axes[2].errorbar(x, ok, oks, **kw)
    for e_i, e in enumerate(eps):
        fail = 20 - res[e]["sarsa"]["greedy_reached"]
        if fail:
            axes[2].annotate(f"{fail}/20\nfinal fail", (e_i, 100 * res[e]["sarsa"]["snap_ok_mean"]),
                             xytext=(0, -20), textcoords="offset points", ha="center",
                             fontsize=6.3, color=C["sarsa"])
    titles = ["(a) Training return (last 500 ep.)", "(b) Episodes with a cliff fall (%)",
              "(c) Greedy policy reaches G (%)"]
    for ax, t in zip(axes, titles):
        ax.set_xticks(x, eps); ax.set_xlabel("ε"); ax.set_title(t, loc="left")
    axes[2].set_ylim(40, 105)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_eps_sweep.png")


# ------------------------------------------------------ Fig 5: training curves (spec settings)
def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for algo in ALGOS:
        R = np.stack([smooth(z["returns"]) for z in load(algo)])
        m, s = R.mean(0), R.std(0)
        ep = np.arange(len(m)) + 50
        ax.fill_between(ep, m - s, m + s, color=C[algo], alpha=0.2, lw=0)
        last = np.mean([z["returns"][-500:].mean() for z in load(algo)])
        ax.plot(ep, m, color=C[algo], lw=1.0, label=f"{ALGOS[algo]} (last 500: {last:.1f})")
    ax.axhline(-11, color=C["optimal"], ls="--", lw=0.8)
    ax.text(3000, -9, "optimal −11", ha="right", va="bottom", fontsize=6.5, color=C["optimal"])
    ax.set_ylim(-100, 0); ax.set_xlim(0, 3000)
    ax.set_xlabel("episode"); ax.set_ylabel("return (50-ep. moving avg.)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig5_training_curves.png")


# ------------------------------------------------------ Fig 6: frozen policies at different eps
def fig_cross_eval():
    res = EXTRA["E"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.9))
    for key, lab, col, ls in [("dp_optimal", "DP-optimal policy", C["optimal"], "--"),
                              ("sarsa", "SARSA policy", C["sarsa"], "-"),
                              ("qlearning", "Q-learning policy", C["qlearning"], "-")]:
        eps = sorted(res[key], key=float)
        xs = [float(e) for e in eps]
        mk = None if key == "dp_optimal" else "o"
        axes[0].plot(xs, [res[key][e]["return"] for e in eps], ls, color=col, marker=mk, ms=3, lw=1.2, label=lab)
        axes[1].plot(xs, [res[key][e]["falls_per_episode"] for e in eps], ls, color=col, marker=mk, ms=3, lw=1.2)
    for ax in axes:
        ax.axvline(0.1, color=GRID, lw=4, zorder=0)
        ax.set_xlabel("ε at execution (frozen policy, no learning)")
    for ax in axes:
        ax.text(0.106, 1.0, "training ε", transform=ax.get_xaxis_transform(), fontsize=6.5,
                color=MUTED, va="top")
    axes[0].set_title("(a) Mean return", loc="left")
    axes[1].set_title("(b) Cliff falls per episode", loc="left")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_cross_eval.png")


if __name__ == "__main__":
    fig_gridworld(); fig_paths(); fig_value_maps()
    fig_eps_sweep(); fig_training_curves(); fig_cross_eval()
    print("figures written to", OUT)
