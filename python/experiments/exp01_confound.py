"""
Experiment 01 -- the confound, and what conditioning recovers.

Question
--------
Blind sampling-rate-offset estimators from the WASN literature assume the
two channels are related by a time-invariant transfer function. In
multimodal capture (air microphone + body-conduction sensor) that
relationship is signal-dependent: it changes with what is being said.

Does that break the estimator, and does conditioning on signal class fix it?

Conditions
----------
  static     -- one fixed transfer function (the WASN assumption; control)
  switching  -- class-dependent transfer function (multimodal reality)
  switching + class conditioning (the proposed estimator)

Everything else is held identical: same source, same SRO, same noise, same
estimator arithmetic. The ONLY difference is the channel model.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohsq.model import simulate
from cohsq.sro import estimate_sro_drift, estimate_sro_conditioned

N_SEEDS = 20
EPS_LIST = [20e-6, 50e-6, 100e-6, 200e-6]
DURATION = 60.0


def run():
    rows = []
    for eps in EPS_LIST:
        acc = {"static": [], "switching": [], "conditioned": []}
        kept = []
        for seed in range(N_SEEDS):
            a, b, _, fs = simulate(duration_s=DURATION, eps=eps,
                                   static_channel=True, seed=seed)
            acc["static"].append(estimate_sro_drift(a, b, fs))

            a, b, seq, fs = simulate(duration_s=DURATION, eps=eps,
                                     static_channel=False, seed=seed)
            acc["switching"].append(estimate_sro_drift(a, b, fs))
            e, f = estimate_sro_conditioned(a, b, fs, seq)
            acc["conditioned"].append(e)
            kept.append(f)

        row = {"eps_ppm": eps * 1e6, "kept_frac": float(np.mean(kept))}
        for k, v in acc.items():
            v = np.array(v, dtype=float)
            v = v[np.isfinite(v)]
            row[k] = {
                "mean_ppm": float(np.mean(v) * 1e6),
                "bias_ppm": float((np.mean(v) - eps) * 1e6),
                "std_ppm": float(np.std(v) * 1e6),
                "rmse_ppm": float(np.sqrt(np.mean((v - eps) ** 2)) * 1e6),
            }
        rows.append(row)

        print(f"\ntrue eps = {eps*1e6:6.1f} ppm     "
              f"(class-matched pairs retained: {row['kept_frac']*100:.0f}%)")
        print(f"  {'condition':<24}{'mean':>10}{'bias':>10}{'std':>10}{'rmse':>10}")
        for k in ["static", "switching", "conditioned"]:
            r = row[k]
            print(f"  {k:<24}{r['mean_ppm']:>10.2f}{r['bias_ppm']:>10.2f}"
                  f"{r['std_ppm']:>10.2f}{r['rmse_ppm']:>10.2f}")
    return rows


if __name__ == "__main__":
    rows = run()
    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "docs", "results", "exp01_confound.json")
    with open(os.path.abspath(out), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {os.path.abspath(out)}")
