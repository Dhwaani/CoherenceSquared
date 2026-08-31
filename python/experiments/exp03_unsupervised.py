"""
Experiment 03 -- can the classes be discovered without labels?

The headline result conditions on ORACLE class labels, which no deployed system
has. Experiment 02 showed the method tolerates label error up to ~45%, which
suggests a weak labeller would do. This experiment asks the sharper question:

    can the classes be discovered from the body sensor alone, unsupervised,
    with no phoneme recogniser and no labelled corpus?

If yes, the method needs no linguistic resources at all -- it becomes
language-independent and trainable on unlabelled data.

Conditions:
    baseline      -- no conditioning (prior art)
    oracle        -- conditioning on true class labels (upper bound)
    unsupervised  -- conditioning on k-means classes from the body sensor
    mismatched-K  -- unsupervised with the wrong number of clusters
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohsq.model import simulate
from cohsq.sro import estimate_sro_drift, estimate_sro_conditioned
from cohsq.classes import discover_classes

EPS = 100e-6
TRUE_K = 8
N_SEEDS = 10
DURATION = 60.0


def rmse(vals):
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean((v - EPS) ** 2)) * 1e6) if len(v) else float("nan")


def main():
    acc = {"baseline": [], "oracle": [], "unsup_K8": [],
           "unsup_K4": [], "unsup_K16": []}
    kept = {"oracle": [], "unsup_K8": []}

    for seed in range(N_SEEDS):
        air, body, seq, fs = simulate(duration_s=DURATION, eps=EPS,
                                      static_channel=False, seed=seed)

        acc["baseline"].append(estimate_sro_drift(air, body, fs))

        e, f = estimate_sro_conditioned(air, body, fs, seq)
        acc["oracle"].append(e)
        kept["oracle"].append(f)

        for k, name in [(8, "unsup_K8"), (4, "unsup_K4"), (16, "unsup_K16")]:
            lab = discover_classes(body, fs, n_classes=k, seed=seed)
            e, f = estimate_sro_conditioned(air, body, fs, lab)
            acc[name].append(e)
            if name == "unsup_K8":
                kept[name].append(f)

    print(f"true eps = {EPS*1e6:.0f} ppm, {TRUE_K} true classes, "
          f"{N_SEEDS} seeds, {DURATION:.0f} s\n")
    print(f"  {'condition':<26}{'rmse ppm':>10}{'kept':>10}")
    rows = {}
    for name in ["baseline", "oracle", "unsup_K8", "unsup_K4", "unsup_K16"]:
        r = rmse(acc[name])
        rows[name] = r
        k = f"{np.mean(kept[name])*100:.0f}%" if name in kept else "-"
        print(f"  {name:<26}{r:>10.2f}{k:>10}")

    print(f"\n  unsupervised recovers "
          f"{(rows['baseline']-rows['unsup_K8'])/(rows['baseline']-rows['oracle'])*100:.0f}% "
          f"of the oracle's advantage over baseline")

    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "docs", "results", "exp03_unsupervised.json")
    with open(os.path.abspath(out), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
