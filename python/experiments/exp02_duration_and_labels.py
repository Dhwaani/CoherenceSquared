"""
Experiment 02 -- two questions Experiment 01 raises.

(a) Is the error a BIAS or merely slow convergence?
    If the class-dependent phase term telescoped away over long windows, the
    error would shrink as observation time grows and the problem would be one
    of convergence speed, not of consistency. Sweep duration and find out.

(b) How good does the class label have to be?
    Experiment 01 conditions on oracle labels. A deployable system estimates
    the class from the body sensor alone, imperfectly. Corrupt the labels at
    increasing rates and find where the advantage disappears.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohsq.model import simulate
from cohsq.sro import estimate_sro_drift, estimate_sro_conditioned

EPS = 100e-6
N_SEEDS = 10


def corrupt(seq, rate, n_classes, rng):
    """Randomly reassign a fraction of the class track, in runs."""
    out = seq.copy()
    n = len(seq)
    seg = 800  # ~50 ms at 16 kHz
    for start in range(0, n, seg):
        if rng.random() < rate:
            out[start:start + seg] = rng.integers(0, n_classes)
    return out


def duration_sweep():
    print("(a) error vs observation length   [true eps = 100 ppm]\n")
    print(f"  {'duration':>10}{'static':>12}{'switching':>12}{'conditioned':>14}")
    rows = []
    for dur in [15.0, 30.0, 60.0, 120.0, 240.0]:
        sw, st, cond = [], [], []
        for seed in range(N_SEEDS):
            a, b, _, fs = simulate(duration_s=dur, eps=EPS,
                                   static_channel=True, seed=seed)
            st.append(estimate_sro_drift(a, b, fs))
            a, b, seq, fs = simulate(duration_s=dur, eps=EPS,
                                     static_channel=False, seed=seed)
            sw.append(estimate_sro_drift(a, b, fs))
            e, _ = estimate_sro_conditioned(a, b, fs, seq)
            cond.append(e)

        def rmse(v):
            v = np.array(v, dtype=float)
            v = v[np.isfinite(v)]
            return float(np.sqrt(np.mean((v - EPS) ** 2)) * 1e6)

        def bias(v):
            v = np.array(v, dtype=float)
            v = v[np.isfinite(v)]
            return float((np.mean(v) - EPS) * 1e6)

        rows.append({"duration_s": dur, "static_rmse": rmse(st),
                     "switching_rmse": rmse(sw), "switching_bias": bias(sw),
                     "conditioned_rmse": rmse(cond)})
        print(f"  {dur:>8.0f}s{rmse(st):>12.2f}{rmse(sw):>12.2f}"
              f"{rmse(cond):>14.2f}     (switching bias {bias(sw):+.1f} ppm)")
    return rows


def label_sweep():
    print("\n(b) conditioned estimator vs class-label error rate\n")
    print(f"  {'label err':>10}{'rmse ppm':>12}{'kept':>10}")
    rng = np.random.default_rng(0)
    rows = []
    for rate in [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]:
        vals, kept = [], []
        for seed in range(N_SEEDS):
            a, b, seq, fs = simulate(duration_s=60.0, eps=EPS,
                                     static_channel=False, seed=seed)
            noisy = corrupt(seq, rate, 8, rng)
            e, f = estimate_sro_conditioned(a, b, fs, noisy)
            vals.append(e)
            kept.append(f)
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        r = float(np.sqrt(np.mean((v - EPS) ** 2)) * 1e6)
        rows.append({"label_error": rate, "rmse_ppm": r,
                     "kept_frac": float(np.mean(kept))})
        print(f"  {rate:>10.0%}{r:>12.2f}{np.mean(kept):>10.0%}")
    return rows


if __name__ == "__main__":
    d = duration_sweep()
    l = label_sweep()
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "docs", "results", "exp02_duration_labels.json")
    with open(os.path.abspath(out), "w") as fh:
        json.dump({"duration": d, "labels": l}, fh, indent=2)
    print(f"\nwrote {os.path.abspath(out)}")
