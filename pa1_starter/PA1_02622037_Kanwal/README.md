# PA1 — Seher Kanwal, <student ID>

From Bellman updates to TD control: dynamic programming on RoomsGridWorld,
SARSA and Q-learning on CliffWalk.

## How to reproduce

Run from this folder with the environment from the Environment section active.

```powershell
# one-time setup (Windows; conda was not available, so a venv is used instead).
# The venv lives OUTSIDE this folder so it is never included in the archive.
python -m venv ..\rl2026
..\rl2026\Scripts\Activate.ps1
pip install -r requirements.txt

# every number in results.json, plus raw per-seed output in results/
python train.py --config configs/pa1.yaml

# the three required figures in figures/ (reads results/, does not retrain)
python evaluate.py
```

| Output | Produced by |
|---|---|
| `results.json`, every `gridworld` entry | `train.py` → `run_dynamic_programming` |
| `results.json`, `cliff` DP entries (`V_optimal` … `optimal_path_length`) | `train.py` → `run_cliff_dp` |
| `results.json`, `cliff` TD entries (`*_training_return`, `*_greedy_return`, `*_highest_row`) | `train.py` → `run_td_control` |
| `results/{sarsa,qlearning}_seed{0..4}.npz` (Q, per-episode returns, greedy path) | `train.py` → `run_td_control` |
| `figures/gridworld_values_policy.png` | `evaluate.py` → `plot_gridworld` |
| `figures/cliff_training_curves.png` | `evaluate.py` → `plot_training_curves` |
| `figures/cliff_greedy_paths.png` | `evaluate.py` → `plot_greedy_paths` |

All hyperparameters come from `configs/pa1.yaml` (α = 0.5, ε = 0.1, 3000 episodes,
400-step cap, θ = 1e-12, seeds 0–4). Re-running `train.py` reproduces
`results.json` byte for byte.

### Supplementary experiments for the report (figures marked [supp.])

```powershell
python extra_experiments.py   # ~70 s; results_extra/extra_results.json + figures/extra_*.png
```

| Experiment | Report | Seeds | Figure |
|---|---|---|---|
| A: PI vs VI convergence trace | Q1 | — (deterministic) | `extra_A_pi_vs_vi.png` |
| B: learned value maps max_a Q (uses `results/*.npz`) | Q2 | 0–4 | `extra_B_value_maps.png` |
| C: ε sweep with greedy-policy stability | Q3 | 0–19 | `extra_C_epsilon_sweep_20seeds.png`, `extra_C_learning_curves_by_eps.png` |
| D: SARSA, ε = 0.01, 20,000 episodes | Q3 | 0–4 | `extra_F_sarsa_optimal_path.png` (c) |
| E: frozen policies executed at several ε, no learning (2,000 episodes each) | Q4 | 0–4 (execution rng 999–1004) | `extra_E_cross_evaluation.png` |
| F: ε decayed linearly 0.1 → 0 | Q4 | 0–19 | `extra_F_sarsa_optimal_path.png` (a, b) |
| G: SARSA greedy failures, using the exact fixed point q*_ε and α ∈ {0.5, 0.1, 0.5/(1+n/50)} | Q2, Q3 | 0–19 | report Table 4 |

The script re-implements the training loop only so that it can record per-episode
greedy snapshots and use a per-episode ε. It uses the same update rules and
`agent.epsilon_greedy`. On start-up it asserts that its Q tables and returns are
bit-identical to `agent.sarsa` / `agent.q_learning` for seeds 0–4. All runs
completed; none were dropped.

**Cliff falls** (report Table 2, Figure 4b; `summary.csv`) are counted as the
share of the last 500 training episodes that contain at least one actual cliff
transition (reward −75). An earlier version used "episode return < −60" as a
proxy, which also counted long SARSA episodes without any fall (e.g. 18.2 %
instead of 15.6 % at ε = 0.3); the proxy is no longer used. `sweep_summary.py`
gets the counts by replaying each saved run with `agent.sarsa` / `agent.q_learning`
in a CliffWalk that counts cliff transitions, and asserts that the replayed
returns equal the saved ones.

The supplementary runs also produced failed greedy policies, i.e. runs whose
final greedy policy does not reach G. All are reported, not dropped:
- C (20 seeds): 1/20 at ε = 0.1, 1/20 at ε = 0.2, 5/20 at ε = 0.3
- D: seed 3

### Report figures

```powershell
python report_figures.py      # figures/report_fig1..6_*.png, the six figures in report.pdf
```

Reads `results/*.npz` and `results_extra/extra_results.json`; nothing is
retrained. Figures 1, 2 and 5 show the same data as the three required figures
from `evaluate.py`, laid out at their printed size; Figures 3, 4 and 6 redraw
experiments B, C and E. The report's tables are filled from the same files.

### ε sweep (report question 3)

```powershell
foreach ($e in '0.01','0.05','0.1','0.2','0.3') { python train.py --config "configs/eps_$e.yaml" }
python sweep_summary.py      # results_sweep/summary.csv + figures/epsilon_sweep.png
```

Each `configs/eps_<ε>.yaml` is identical to `configs/pa1.yaml` except
`td_control.epsilon` and the two output paths (`results_sweep/eps_<ε>/`), so the
main `results.json` and `results/` are never overwritten. The ε = 0.1 sweep run
reproduces the main TD results exactly. The sweep takes about 7 s in total.

**Sweep seeds:** the same seeds, 0–4, for every ε. All 50 runs completed and
all are reported. For SARSA at ε = 0.2 and at ε = 0.3, seed 3's final greedy
policy does not reach G (greedy return −400 at the 400-step cap):

| ε | Where the greedy policy gets stuck |
|---|---|
| 0.2 | at S, choosing "down" into the border |
| 0.3 | in a loop in the top row |

These runs are kept in `summary.csv` and marked ✕ in the figure, not dropped.
See the Notes section for the cause.

## Seeds

Five seeds, 0–4. Every seed that was run is listed; none failed and none were
excluded. Each (algorithm, seed) pair uses its own `np.random.default_rng(seed)`.

| Seed | SARSA: training return (last 500) / greedy return / highest row | Q-learning: training return (last 500) / greedy return / highest row | Notes |
|---|---|---|---|
| 0 | −24.464 / −17 / 0 | −32.100 / −11 / 3 | |
| 1 | −20.332 / −17 / 0 | −30.872 / −11 / 3 | |
| 2 | −22.846 / −17 / 0 | −32.612 / −11 / 3 | |
| 3 | −25.530 / −15 / 1 | −31.882 / −11 / 3 | SARSA's path reaches row 1, not row 0 |
| 4 | −22.790 / −17 / 0 | −31.176 / −11 / 3 | |
| **Mean** | **−23.192 / −16.6 / 0.2** | **−31.728 / −11.0 / 3.0** | the values in `results.json` |

Every greedy policy, for every seed and both algorithms, reaches the goal.

## Environment

```
python 3.12.5
numpy 2.4.6
matplotlib 3.11.1
pyyaml 6.0.3
gymnasium  not installed (not used; pa1_envs.py has no Gymnasium dependency)
torch      not installed (not used)
```

- OS: Windows 11 Pro
- CPU or GPU: CPU only (Intel Core Ultra 7 270K Plus)
- Total wall-clock time for `train.py`: about 1.3 s (DP + 10 TD runs); `evaluate.py`: about 0.9 s
- The starter README's `conda activate rl2026` was replaced by a Python venv
  named `rl2026`, because conda is not installed on this machine. The pinned
  versions in `requirements.txt` are unchanged.

## AI coding assistants

Claude (Anthropic), used through Claude Code in VS Code.

- **Explanations:** the assistant explained policy evaluation, policy iteration,
  value iteration, ε-greedy, SARSA and Q-learning, and worked through numeric
  examples on these environments.
- **`agent.py`:**
  - `value_iteration` and the first version of `policy_evaluation`: written by
    me, then reviewed by the assistant.
  - `greedy_policy`, `policy_iteration`, `epsilon_greedy`, `sarsa`, `q_learning`:
    the assistant gave code in the chat, which I typed in and adapted.
  - The assistant found and helped fix bugs:
    - `theta` passed positionally into `sweeps`
    - `np.eye(n_states)` instead of `np.eye(n_actions)`
    - `env.step` unpacked into 4 values instead of 5
    - syntax and indentation errors
  - The assistant tidied the comments and whitespace in the DP section.
- **`train.py`:** the assistant gave code for the helpers (`probe_values`,
  `probe_actions`, `rollout`) and for `run_dynamic_programming`, `run_cliff_dp`
  and `run_td_control`, which I typed in.
- **`evaluate.py`:** written by the assistant (all three figures).
- **ε sweep:** the assistant wrote `sweep_summary.py` and the
  `configs/eps_*.yaml` files, made the one-word `mkdir` change in `train.py`,
  ran the sweep, and investigated the two failed SARSA seeds.
- **Testing:** the assistant checked every function against an independent
  reference implementation, and checked the archive with
  `pa1_check_submission.py`.
- **This README:** drafted by the assistant from the actual run output.

- **Supplementary experiments:** the assistant designed and wrote
  `extra_experiments.py` and its figures.
- **Corrections after review (by the assistant):** the cliff-fall count in
  `extra_experiments.py` and `sweep_summary.py` (actual −75 transitions instead
  of a return threshold); `evaluate.py` now loads exactly the seeds listed in
  `configs/pa1.yaml`; `agent.policy_evaluation(..., sweeps=0)` now performs no
  sweep. None of these changes any number in `results.json` (re-run and checked
  byte for byte).
- **Report (`report.pdf`):** the assistant drafted the report text, tables and
  layout from these results, and wrote `report_figures.py`. I reviewed and edited it, and I take
  responsibility for its content.

I have checked the results and can explain every line of the submitted code.

## Notes

- **Conventions** follow Section 2 of the spec:
  - synchronous sweeps (a copy of V_k is read; V_{k+1} is written)
  - V = 0 initially, and terminal states stay 0
  - the first pass counts as sweep 1
  - stop when max|ΔV| < 1e-12
  - argmax ties go to the lowest action index (`np.argmax`)
  - policy iteration starts with "up" everywhere and counts the final, unchanged round
- **Tie at the start state:** in RoomsGridWorld, up and right are exactly tied at
  S (Q = −2.03789182 for both). The tie-break gives "up".
- **Sanity checks** (printed by `train.py` if they fail; none did):
  - policy iteration and value iteration give the same policy
  - every greedy policy reaches the goal
- **TD terminal handling:** the target is just `r` when the step ends the
  episode. Stepping onto the cliff is not terminal (−75, back to S). The
  400-step cap is truncation, not termination.
- **Training-curve figure:**
  - returns are smoothed with a 50-episode trailing moving average
  - the y-axis is clipped at −100, because the first episodes are far lower
  - the numbers printed beside each curve are the raw last-500 means, the same
    as in `results.json`
- **Greedy-path figure:** all five seeds are drawn for each algorithm, slightly
  offset so that identical paths stay visible.
- **SARSA greedy failures (supplementary runs only).** The spec says a greedy
  policy that fails to reach G indicates a bug "at convergence". In the
  supplementary runs, some final SARSA greedy policies loop until the 400-step
  cap:
  - 5-seed sweep: seed 3 at ε = 0.2 and at ε = 0.3
  - 20-seed runs: 1/20 at ε = 0.1, 1/20 at ε = 0.2, 5/20 at ε = 0.3

  All graded runs (ε = 0.1, seeds 0–4) reach G. Experiment G
  (`extra_experiments.py`, report Table 4) shows these failures are
  non-convergence under the prescribed constant α = 0.5, not a bug:
  1. **The convergence target is fine.** SARSA's convergence target, the optimal
     ε-greedy values q*_ε solved exactly from the model, has a greedy policy that
     reaches G at every ε.
  2. **The failures disappear when SARSA can converge.** On the same 20 seeds,
     α = 0.1 or α = 0.5/(1 + n(s,a)/50) gives 0/20 failures at every ε and a stable
     greedy policy in 98–100 % of late snapshots. At ε = 0.1, α = 0.1 learns
     exactly q*_ε's path (row 2, −13) on all 20 seeds.
  3. **The implementation is verified.** The code matches a reference SARSA bit
     for bit.

  The mechanism: SARSA's target samples a′ from the exploring policy, and with
  α = 0.5 each sample moves Q halfway, so near-tied actions keep swapping. For
  example, at ε = 0.2, seed 3 has Q(S, up) = −24.02 < Q(S, down) = −23.85, and
  "down" bumps the border. Q-learning's max-target has no sampled action, so it
  converges even with α = 0.5 and never fails. The same α effect is why SARSA
  with α = 0.5 climbs to rows 0–1 rather than the row 2 that q*_ε predicts.
- `train.py` change for the sweep: the output folder is created with
  `mkdir(parents=True, exist_ok=True)`, so that nested sweep paths work.
- `pa1_envs.py` is unmodified.
