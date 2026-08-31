"""
Self-test for the D0.3 phase-diversity measurement.

Validates the INSTRUMENT, not the hypothesis. Two controls with known answers:

  identical channel   -> spread must be ~0   (no class dependence exists)
  known phase offsets -> spread must match the value computed analytically

If these pass, a spread measured on Vibravox can be believed. If they fail, any
number the tool produces on real data is meaningless.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cohsq.phase_diversity import phase_diversity, circular_spread, summarise

FS = 16000
N = FS * 20


def make_labels(n, n_classes, seg=1200, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.zeros(n, dtype=int)
    for i in range(0, n, seg):
        lab[i:i + seg] = rng.integers(0, n_classes)
    return lab


def test_no_diversity():
    """Same filter for every class -> spread must be near zero."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)
    h = np.convolve(rng.standard_normal(24), np.ones(4) / 4)
    y = np.convolve(x, h)[:N]
    lab = make_labels(N, 6)
    r = phase_diversity(x, y, lab, FS)
    s = summarise(r)
    print(f"  identical channel : mean spread {s['mean_spread_deg']:7.2f} deg "
          f"({s['n_classes']} classes)")
    assert s["mean_spread_rad"] < 0.15, "spread should vanish with no class dependence"
    return s


def test_known_diversity():
    """Analytic check of the circular spread estimator itself."""
    for true_spread in [0.2, 0.5, 1.0]:
        rng = np.random.default_rng(1)
        ph = rng.normal(0.0, true_spread, size=(2000, 1))
        est = circular_spread(ph)[0]
        print(f"  wrapped-normal    : true {true_spread:.2f} rad -> "
              f"estimated {est:.2f} rad")
        assert abs(est - true_spread) < 0.12, "circular spread estimator is off"


def test_diversity_appears():
    """Different filter per class -> spread must be clearly non-zero."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(N)
    lab = make_labels(N, 6, seed=3)
    filters = [rng.standard_normal(24) for _ in range(6)]
    y = np.zeros(N)
    for c in range(6):
        m = (lab == c).astype(float)
        y += m * np.convolve(x, filters[c])[:N]
    r = phase_diversity(x, y, lab, FS)
    s = summarise(r)
    print(f"  per-class filters : mean spread {s['mean_spread_deg']:7.2f} deg "
          f"({s['n_classes']} classes)")
    assert s["mean_spread_rad"] > 0.3, "spread should appear with class dependence"
    return s


if __name__ == "__main__":
    print("D0.3 instrument self-test\n")
    test_known_diversity()
    test_no_diversity()
    test_diversity_appears()
    print("\nOK -- the measurement responds correctly to known inputs")
