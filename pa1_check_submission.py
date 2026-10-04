#!/usr/bin/env python3
"""
PA1 submission checker.

Run this on the .zip you are about to upload:

    python pa1_check_submission.py PA1_20251234_Kim.zip

It checks the FORMAT only — whether the archive is laid out the way the
specification asks and whether results.json can be read. It does not check
whether your answers are right.

Every [ERROR] must be fixed before you upload. [WARN] is worth a look.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile

REQUIRED_FILES = ["results.json", "report.pdf", "README.md",
                  "train.py", "evaluate.py", "agent.py"]
REQUIRED_ONE_OF = [("requirements.txt", "environment.yml")]
JUNK = ["__pycache__/", ".git/", ".venv/", "venv/", ".ipynb_checkpoints/",
        ".DS_Store", "env/", ".conda/"]

SCHEMA = {
    "gridworld": {
        "V_random_10_sweeps": dict, "V_random_converged": dict,
        "V_optimal": dict, "optimal_action": dict,
        "policy_iteration_rounds": int, "value_iteration_sweeps": int,
        "optimal_undiscounted_return_from_start": float,
    },
    "cliff": {
        "V_optimal": dict, "optimal_action": dict,
        "value_iteration_sweeps": int,
        "optimal_undiscounted_return_from_start": float,
        "optimal_path_length": int,
        "sarsa_training_return": float, "qlearning_training_return": float,
        "sarsa_greedy_return": float, "qlearning_greedy_return": float,
        "sarsa_highest_row": float, "qlearning_highest_row": float,
    },
}
PROBES = ["A", "B", "C", "D"]
# Fields whose correct answer may legitimately be 0, so a 0 there is not a
# sign of an unfilled template.
ZERO_IS_PLAUSIBLE = {"optimal_undiscounted_return_from_start"}
ACTIONS = {"up", "right", "down", "left"}

errors: list[str] = []
warns: list[str] = []


def err(m):
    errors.append(m)
    print(f"  [ERROR] {m}")


def warn(m):
    warns.append(m)
    print(f"  [WARN ] {m}")


def ok(m):
    print(f"  [ OK  ] {m}")


def check_number(path, value, kind):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        err(f"{path}: expected a number, got {type(value).__name__}")
        return
    if kind is int and float(value) != int(value):
        err(f"{path}: expected a whole number, got {value}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    archive = sys.argv[1]

    print(f"\nChecking {archive}\n" + "-" * 66)

    if not zipfile.is_zipfile(archive):
        err(f"{archive} is not a .zip file")
        return finish()

    name = archive.rsplit("/", 1)[-1]
    if not re.fullmatch(r"PA1_[A-Za-z0-9]+_[A-Za-z가-힣]+\.zip", name):
        warn(f"filename {name!r} does not match PA1_<studentID>_<surname>.zip")

    z = zipfile.ZipFile(archive)
    names = [n for n in z.namelist() if not n.endswith("/")]
    if not names:
        err("the archive is empty")
        return finish()

    roots = {n.split("/")[0] for n in z.namelist()}
    if len(roots) != 1:
        err(f"expected exactly one top-level folder, found {len(roots)}: "
            f"{sorted(roots)[:4]}")
        root = ""
    else:
        root = roots.pop() + "/"
        ok(f"single top-level folder: {root}")

    rel = [n[len(root):] for n in names if n.startswith(root)]

    for f in REQUIRED_FILES:
        if f in rel:
            ok(f"found {f}")
        elif any(r.endswith("/" + f) for r in rel):
            deep = next(r for r in rel if r.endswith("/" + f))
            err(f"{f} is nested too deep (found at {deep}) — it must sit "
                f"directly inside {root or 'the top-level folder'}")
        else:
            err(f"missing {f}")

    for group in REQUIRED_ONE_OF:
        if not any(g in rel for g in group):
            err(f"missing a dependency file — one of {', '.join(group)}")
        else:
            ok(f"found {next(g for g in group if g in rel)}")

    for j in JUNK:
        hits = [r for r in rel if j.rstrip("/") in r.split("/")]
        if hits:
            warn(f"{j} is inside the archive ({len(hits)} files) — remove it")

    size_mb = sum(z.getinfo(n).file_size for n in names) / 1e6
    if size_mb > 60:
        warn(f"uncompressed contents are {size_mb:.0f} MB — that is very large")
    else:
        ok(f"uncompressed size {size_mb:.1f} MB")

    # ---- results.json ------------------------------------------------
    target = root + "results.json"
    if target not in names:
        return finish()
    try:
        data = json.loads(z.read(target))
    except json.JSONDecodeError as e:
        err(f"results.json is not valid JSON: {e}")
        return finish()
    ok("results.json parses")

    zeros = 0
    for env, fields in SCHEMA.items():
        if env not in data:
            err(f"results.json is missing the {env!r} section")
            continue
        for field, kind in fields.items():
            if field not in data[env]:
                err(f"results.json: {env}.{field} is missing")
                continue
            v = data[env][field]
            if kind is dict:
                if not isinstance(v, dict):
                    err(f"{env}.{field}: expected an object keyed A-D")
                    continue
                for probe in PROBES:
                    if probe not in v:
                        err(f"{env}.{field}: probe {probe} is missing")
                    elif field == "optimal_action":
                        if str(v[probe]).lower() not in ACTIONS:
                            err(f"{env}.{field}.{probe}: {v[probe]!r} is not one "
                                f"of up/right/down/left")
                    else:
                        check_number(f"{env}.{field}.{probe}", v[probe], float)
                        if v[probe] == 0 and field not in ZERO_IS_PLAUSIBLE:
                            zeros += 1
            else:
                check_number(f"{env}.{field}", v, kind)
                if v == 0 and field not in ZERO_IS_PLAUSIBLE:
                    zeros += 1

    if zeros:
        warn(f"{zeros} value(s) in results.json are still 0 — if you did not "
             f"compute them, they will be marked wrong")

    if data.get("cliff", {}).get("optimal_path_length", 1) <= 0:
        err("cliff.optimal_path_length must be a positive number of steps")

    return finish()


def finish():
    print("-" * 66)
    if errors:
        print(f"{len(errors)} error(s) — fix these before uploading.")
    else:
        print("No errors. Format looks correct.")
        print("This says nothing about whether your answers are right.")
    if warns:
        print(f"{len(warns)} warning(s) — worth a look.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
