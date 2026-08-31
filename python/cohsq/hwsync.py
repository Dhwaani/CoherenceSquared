"""
Deliverables D1.1 and D1.2 -- ground-truth offset measurement and clock drift
statistics from bench hardware.

These are the ANALYSIS halves of the Phase 1 hardware work. They take logs from a
logic analyser, a counter, or a recorded sync channel and produce the numbers.
The rig produces raw captures; this module turns them into ppm.

D1.1 gates every later hardware capture: without independent ground truth you
cannot score an estimator, so an uncalibrated rig produces uninterpretable data
regardless of how good the recordings are.
"""

import numpy as np


# --------------------------------------------------------------------------
# D1.1 method A -- shared electrical sync marker
# --------------------------------------------------------------------------

def detect_markers(channel, threshold=None, min_gap=100):
    """
    Rising-edge indices of a sync pulse train recorded on a spare ADC channel.

    Both capture subsystems record the SAME physical pulse train. Because each
    counts it in its own sample clock, the ratio of counted intervals is the
    clock ratio directly.
    """
    x = np.asarray(channel, dtype=np.float64)
    if threshold is None:
        threshold = 0.5 * (np.max(x) + np.min(x))
    above = x > threshold
    edges = np.flatnonzero(~above[:-1] & above[1:]) + 1
    if len(edges) == 0:
        return edges
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= min_gap:
            keep.append(e)
    return np.array(keep)


def sro_from_markers(markers_a, markers_b):
    """
    Sampling-rate offset from two marker trains of the same physical pulses.

    Returns (eps, stderr_ppm, n_pairs). eps is defined so that stream B runs
    ahead of stream A by a factor (1 + eps), matching the convention used for
    injection elsewhere in this repository.

    A least-squares fit of b_k against a_k is used rather than a ratio of
    endpoints, so the standard error reports honestly how well the linear model
    holds. A large standard error means jitter or missed markers, not a large
    offset -- check it before believing the estimate.
    """
    n = min(len(markers_a), len(markers_b))
    if n < 3:
        raise ValueError("need at least three matched markers")
    a = np.asarray(markers_a[:n], dtype=np.float64)
    b = np.asarray(markers_b[:n], dtype=np.float64)

    A = np.vstack([a, np.ones_like(a)]).T
    coef, res, *_ = np.linalg.lstsq(A, b, rcond=None)
    slope = coef[0]
    eps = slope - 1.0

    pred = A @ coef
    resid = b - pred
    dof = max(n - 2, 1)
    s2 = np.sum(resid ** 2) / dof
    var_slope = s2 / np.sum((a - a.mean()) ** 2)
    return eps, float(np.sqrt(var_slope) * 1e6), n


# --------------------------------------------------------------------------
# D1.1 method B -- direct clock counting
# --------------------------------------------------------------------------

def sro_from_counts(count_a, count_b, gate_time_s=None):
    """
    Offset from two clock counts accumulated over the same gate interval, e.g.
    from a frequency counter or a logic analyser edge count.

    Build BOTH this and the marker method: they share no failure modes, so
    agreement between them is real evidence of calibration, whereas either alone
    is an assertion.
    """
    ca = np.asarray(count_a, dtype=np.float64)
    cb = np.asarray(count_b, dtype=np.float64)
    if ca.shape != cb.shape:
        raise ValueError("count arrays must be the same shape")
    ratio = cb / ca
    eps = ratio - 1.0
    return {
        "eps_ppm_mean": float(np.mean(eps) * 1e6),
        "eps_ppm_std": float(np.std(eps) * 1e6),
        "n": int(ca.size),
        "gate_time_s": gate_time_s,
    }


def cross_check(eps_markers, eps_counts, tol_ppm=1.0):
    """
    D1.1 exit criterion: two independent methods must agree within tolerance.
    Returns (passed, difference_ppm).
    """
    d = abs(eps_markers - eps_counts) * 1e6
    return bool(d <= tol_ppm), float(d)


# --------------------------------------------------------------------------
# D1.2 -- clock drift statistics
# --------------------------------------------------------------------------

def drift_statistics(time_s, eps_series, temperature_c=None):
    """
    Summarise a drift log into the distribution D1.2 is meant to deliver.

    The motivation for this whole project currently rests on an ASSUMED clock
    offset of order 100 ppm. Replacing that assumption with a measurement is
    cheap and strengthens every downstream argument, so this deliverable is worth
    producing early even if the rest of Phase 1 slips.
    """
    t = np.asarray(time_s, dtype=np.float64)
    e = np.asarray(eps_series, dtype=np.float64) * 1e6

    out = {
        "n_samples": int(e.size),
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "eps_ppm_mean": float(np.mean(e)),
        "eps_ppm_std": float(np.std(e)),
        "eps_ppm_min": float(np.min(e)),
        "eps_ppm_max": float(np.max(e)),
        "eps_ppm_p05": float(np.percentile(e, 5)),
        "eps_ppm_p95": float(np.percentile(e, 95)),
    }

    if t.size > 2:
        A = np.vstack([t, np.ones_like(t)]).T
        slope = np.linalg.lstsq(A, e, rcond=None)[0][0]
        out["drift_rate_ppm_per_hour"] = float(slope * 3600.0)

    if temperature_c is not None:
        T = np.asarray(temperature_c, dtype=np.float64)
        if T.size == e.size and np.ptp(T) > 1.0:
            A = np.vstack([T, np.ones_like(T)]).T
            slope = np.linalg.lstsq(A, e, rcond=None)[0][0]
            out["tempco_ppm_per_C"] = float(slope)
            out["temp_range_c"] = [float(T.min()), float(T.max())]
    return out


def required_reacquisition_ppm(f_hz, max_phase_error_deg, t_s):
    """
    Design helper: the largest residual offset tolerable if inter-channel phase
    error must stay under `max_phase_error_deg` at `f_hz` over `t_s` seconds.

    Useful for the power-gating case, where a sensor that sleeps must re-acquire
    synchronisation on wake and only has a short window to do it in.
    """
    return np.radians(max_phase_error_deg) / (2 * np.pi * f_hz * t_s)
