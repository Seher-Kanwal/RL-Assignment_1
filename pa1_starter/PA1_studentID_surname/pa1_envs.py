"""
PA1 environments — Reinforcement Learning (Graduate)

Two deterministic tabular environments, distributed by the instructor.
Do not modify this file. Your submission is run against this exact copy.

The API matches Gymnasium's, so the code you write here carries over to PA2:

    env = RoomsGridWorld()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

For dynamic programming you also need the model. Both environments expose it
as `env.P`, in the same shape Gymnasium's toy_text environments use:

    env.P[state][action] -> [(probability, next_state, reward, terminated)]

Transitions are deterministic, so each list holds exactly one tuple.

Actions (both environments):
    0 = up, 1 = right, 2 = down, 3 = left

States are numbered row-major: state = row * n_cols + col.
"""
from __future__ import annotations

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
_DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class TabularEnv:
    """Shared machinery. Deterministic transitions, no time limit."""

    grid: list[str] = []
    step_reward: float = -1.0
    gamma: float = 1.0

    def __init__(self):
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self._cells = {}
        for r, row in enumerate(self.grid):
            if len(row) != self.n_cols:
                raise ValueError(f"row {r} has inconsistent width")
            for c, ch in enumerate(row):
                self._cells[(r, c)] = ch
        self.start_state = self._find("S")
        self.goal_state = self._find("G")
        self.P = self._build_model()
        self._state = self.start_state

    # ---- geometry -----------------------------------------------------
    def to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def to_s(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def cell(self, s: int) -> str:
        return self._cells[self.to_rc(s)]

    def _find(self, ch: str) -> int:
        for (r, c), v in self._cells.items():
            if v == ch:
                return self.to_s(r, c)
        raise ValueError(f"no cell marked {ch!r}")

    def is_wall(self, r: int, c: int) -> bool:
        return not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or \
            self._cells[(r, c)] == "#"

    def is_terminal(self, s: int) -> bool:
        return self.cell(s) in self.terminal_marks

    # ---- model --------------------------------------------------------
    terminal_marks = "G"

    def _outcome(self, s: int, a: int) -> tuple[int, float, bool]:
        raise NotImplementedError

    def _build_model(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            for a in range(self.n_actions):
                if self.cell(s) == "#":
                    P[s][a] = [(1.0, s, 0.0, True)]      # unreachable
                elif self.is_terminal(s):
                    P[s][a] = [(1.0, s, 0.0, True)]      # absorbing
                else:
                    ns, r, term = self._outcome(s, a)
                    P[s][a] = [(1.0, ns, r, term)]
        return P

    def _move(self, s: int, a: int) -> int:
        r, c = self.to_rc(s)
        dr, dc = _DELTA[a]
        nr, nc = r + dr, c + dc
        return s if self.is_wall(nr, nc) else self.to_s(nr, nc)

    # ---- Gymnasium-style interface ------------------------------------
    def reset(self, seed: int | None = None, options=None):
        self._state = self.start_state
        return self._state, {}

    def step(self, action: int):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be 0-3, got {action}")
        prob, ns, reward, terminated = self.P[self._state][action][0]
        self._state = ns
        return ns, reward, terminated, False, {}

    def render(self, values=None, policy=None) -> str:
        """ASCII view. Pass `values` (array of length n_states) or `policy`."""
        arrows = ["^", ">", "v", "<"]
        out = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                s = self.to_s(r, c)
                ch = self._cells[(r, c)]
                if ch == "#":
                    row.append("  ###  " if values is not None else "#")
                elif values is not None:
                    row.append(f"{values[s]:7.2f}")
                elif policy is not None and not self.is_terminal(s):
                    row.append(arrows[int(policy[s])])
                else:
                    row.append(ch)
            out.append(" ".join(row) if values is not None else "".join(row))
        return "\n".join(out)


class RoomsGridWorld(TabularEnv):
    """
    Dynamic-programming environment: 6 x 7 grid with interior walls.

        . . . . . . G          S  start          #  wall
        . # # . . # .          G  goal, +10, terminal
        . . . . # . .          H  hazard, -8, terminal
        . # . . H . .
        . # . # # . .          every move costs -1
        S . . . . . .          bumping a wall or the border: stay, still -1

    gamma = 0.95
    """
    grid = [
        "......G",
        ".##..#.",
        "....#..",
        ".#..H..",
        ".#.##..",
        "S......",
    ]
    terminal_marks = "GH"
    step_reward = -1.0
    goal_reward = 10.0
    hazard_reward = -8.0
    gamma = 0.95

    def _outcome(self, s, a):
        ns = self._move(s, a)
        ch = self.cell(ns)
        if ch == "G":
            return ns, self.goal_reward, True
        if ch == "H":
            return ns, self.hazard_reward, True
        return ns, self.step_reward, False


class CliffWalk(TabularEnv):
    """
    TD-control environment: 5 x 10 grid with a cliff along the bottom row.

        . . . . . . . . . .    S  start
        . . . . . . . . . .    G  goal, terminal
        . . . . . . . . . .    C  cliff: -75 and teleport back to S
        . . . . . . . . . .       (the episode does NOT end)
        S C C C C C C C C G
                               every move costs -1
    gamma = 0.99

    Why 0.99 and not 1.0: undiscounted values are flat along equal-length
    detours, so a greedy policy read off a near-converged Q can oscillate
    between two states and never terminate. Discounting makes the value
    strictly decrease with distance to the goal, which removes that failure
    mode without changing what SARSA and Q-learning are being compared on.
    """
    grid = [
        "..........",
        "..........",
        "..........",
        "..........",
        "SCCCCCCCCG",
    ]
    terminal_marks = "G"
    step_reward = -1.0
    cliff_reward = -75.0
    gamma = 0.99

    def _outcome(self, s, a):
        ns = self._move(s, a)
        ch = self.cell(ns)
        if ch == "C":
            return self.start_state, self.cliff_reward, False
        if ch == "G":
            return ns, self.step_reward, True
        return ns, self.step_reward, False


# States referenced in the assignment specification.
PROBE_STATES = {
    "gridworld": {
        "A": 35,   # (5, 0)  start
        "B": 16,   # (2, 2)
        "C": 30,   # (4, 2)
        "D": 13,   # (1, 6)
    },
    "cliff": {
        "A": 40,   # (4, 0)  start
        "B": 30,   # (3, 0)
        "C": 34,   # (3, 4)
        "D": 9,    # (0, 9)
    },
}


if __name__ == "__main__":
    for env in (RoomsGridWorld(), CliffWalk()):
        name = type(env).__name__
        print(f"\n=== {name} ===")
        print(env.render())
        print(f"states {env.n_states}  actions {env.n_actions}  "
              f"gamma {env.gamma}  start {env.start_state}  goal {env.goal_state}")
