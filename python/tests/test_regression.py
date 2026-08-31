"""
Fast behavioural regression check.

Not a unit test of arithmetic -- a check that the three behaviours the project
claims still hold. Runs in well under a minute so it can gate every commit.

Claims under test:
  1. With a static channel the estimator recovers the true SRO accurately.
     (If this breaks, the implementation is wrong, not the science.)
  2. With a signal-dependent channel the baseline estimator degrades badly.
  3. Class conditioning recovers most of that loss.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cohsq.model import simulate
from cohsq.sro import estimate_sro_drift, estimate_sro_conditioned

EPS = 100e-6
DURATION = 30.0
SEEDS = range(5)


def _rmse(vals):
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    assert len(v) > 0, "all estimates were NaN"
    return float(np.sqrt(np.mean((v - EPS) ** 2)) * 1e6)


def main():
    static, switching, conditioned = [], [], []
    for seed in SEEDS:
        a, b, _, fs = simulate(duration_s=DURATION, eps=EPS,
                               static_channel=True, seed=seed)
        static.append(estimate_sro_drift(a, b, fs))

        a, b, seq, fs = simulate(duration_s=DURATION, eps=EPS,
                                 static_channel=False, seed=seed)
        switching.append(estimate_sro_drift(a, b, fs))
        e, _ = estimate_sro_conditioned(a, b, fs, seq)
        conditioned.append(e)

    r_static = _rmse(static)
    r_switch = _rmse(switching)
    r_cond = _rmse(conditioned)

    print(f"static      RMSE {r_static:8.2f} ppm")
    print(f"switching   RMSE {r_switch:8.2f} ppm")
    print(f"conditioned RMSE {r_cond:8.2f} ppm")

    failures = []
    if r_static > 5.0:
        failures.append(f"static RMSE {r_static:.2f} > 5 ppm -- implementation broken")
    if r_switch < 30.0:
        failures.append(
            f"switching RMSE {r_switch:.2f} < 30 ppm -- the confound did not appear; "
            "the channel model may have stopped switching")
    if r_cond > 0.5 * r_switch:
        failures.append(
            f"conditioned RMSE {r_cond:.2f} is not clearly better than "
            f"baseline {r_switch:.2f}")

    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("\nOK -- all three behavioural claims hold")


if __name__ == "__main__":
    main()
