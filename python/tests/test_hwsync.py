"""
Self-test for the Phase 1 hardware analysis tools (D1.1, D1.2).

Synthesises marker trains and clock counts with a KNOWN offset and checks the
tools recover it. Validates the analysis before the rig exists, so that when real
captures arrive the only unknown is the hardware.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cohsq.hwsync import (detect_markers, sro_from_markers, sro_from_counts,
                          cross_check, drift_statistics,
                          required_reacquisition_ppm)

FS = 48000


def synth_marker_channel(n, period, eps, jitter=0.0, seed=0):
    """
    Pulse train as recorded by a clock running (1+eps) relative to nominal.

    A faster clock takes MORE samples per second, so the same physical pulse
    interval spans more samples -- hence period*(1+eps), matching the convention
    sro_from_markers reports (slope of b against a, minus one).
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    k, pos = 0, 0.0
    while pos < n - 10:
        p = int(round(pos + (rng.normal(0, jitter) if jitter else 0.0)))
        if 0 <= p < n - 5:
            x[p:p + 5] = 1.0
        k += 1
        pos = k * period * (1.0 + eps)
    return x


def test_markers():
    print("D1.1 method A -- shared sync markers")
    for true_eps in [10e-6, 50e-6, 100e-6, 250e-6]:
        n = FS * 120
        a = synth_marker_channel(n, FS * 0.5, 0.0)
        b = synth_marker_channel(n, FS * 0.5, true_eps)
        ma, mb = detect_markers(a), detect_markers(b)
        eps, se, npair = sro_from_markers(ma, mb)
        err = (eps - true_eps) * 1e6
        print(f"  true {true_eps*1e6:>6.1f} ppm -> measured {eps*1e6:>8.3f} ppm "
              f"(err {err:>7.3f}, stderr {se:.3f}, {npair} markers)")
        assert abs(err) < 1.0, "marker method must resolve better than 1 ppm"


def test_markers_with_jitter():
    print("\n  with 2-sample marker jitter (realistic edge detection noise)")
    true_eps = 100e-6
    n = FS * 120
    a = synth_marker_channel(n, FS * 0.5, 0.0, jitter=2.0, seed=1)
    b = synth_marker_channel(n, FS * 0.5, true_eps, jitter=2.0, seed=2)
    eps, se, npair = sro_from_markers(detect_markers(a), detect_markers(b))
    print(f"  true {true_eps*1e6:.1f} ppm -> measured {eps*1e6:.3f} ppm "
          f"(stderr {se:.3f} ppm)")
    assert abs(eps - true_eps) * 1e6 < 2.0


def test_counts_and_crosscheck():
    print("\nD1.1 method B -- direct clock counts, and the cross-check")
    true_eps = 100e-6
    rng = np.random.default_rng(0)
    gate = 1.0
    ca = np.full(60, 48_000_000.0) + rng.normal(0, 2, 60)
    cb = ca * (1.0 + true_eps)
    r = sro_from_counts(ca, cb, gate_time_s=gate)
    print(f"  counts: {r['eps_ppm_mean']:.3f} +/- {r['eps_ppm_std']:.3f} ppm")
    ok, diff = cross_check(true_eps, r["eps_ppm_mean"] / 1e6, tol_ppm=1.0)
    print(f"  cross-check vs markers: {'PASS' if ok else 'FAIL'} "
          f"(difference {diff:.3f} ppm)")
    assert ok


def test_drift():
    print("\nD1.2 -- drift statistics")
    t = np.linspace(0, 3600, 600)
    T = 25 + 15 * np.sin(2 * np.pi * t / 3600)
    eps = (80e-6 + 0.8e-6 * (T - 25) + np.random.default_rng(0).normal(0, 1e-6, t.size))
    s = drift_statistics(t, eps, temperature_c=T)
    print(f"  mean {s['eps_ppm_mean']:.1f} ppm, "
          f"range {s['eps_ppm_min']:.1f}..{s['eps_ppm_max']:.1f}")
    print(f"  tempco {s.get('tempco_ppm_per_C', float('nan')):.3f} ppm/C "
          f"(injected 0.800)")
    assert abs(s["tempco_ppm_per_C"] - 0.8) < 0.1


def test_reacquisition():
    print("\nDesign helper -- re-acquisition budget after a power gate")
    for t in [0.1, 0.5, 2.0]:
        e = required_reacquisition_ppm(1500.0, 30.0, t)
        print(f"  stay under 30 deg at 1.5 kHz for {t:>4.1f} s "
              f"-> residual offset < {e*1e6:7.2f} ppm")


if __name__ == "__main__":
    print("Phase 1 analysis self-test\n")
    test_markers()
    test_markers_with_jitter()
    test_counts_and_crosscheck()
    test_drift()
    test_reacquisition()
    print("\nOK -- Phase 1 analysis tools recover known offsets correctly")
