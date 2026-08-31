"""
Fractional resampling for ground-truth sampling-rate-offset injection.

Deliverable D0.2 (core). The entire validation programme depends on being able to
apply a *precisely known* offset to a real recording. If the resampler's own error
floor is comparable to the offsets under test, nothing downstream is interpretable:
you would be measuring the resampler, not the estimator.

Two implementations are provided so the error floor can be demonstrated rather than
asserted:

  resample_hermite  -- cubic Hermite. Fast, and NOT accurate enough. Kept as the
                       negative control in the accuracy report.
  resample_sinc     -- Kaiser-windowed sinc. The one to use.

Convention (used consistently across this repository):

    y[n] = x[n * (1 + eps)]

so a positive eps means the output stream runs AHEAD of the input -- it consumes
input faster than one sample per output sample.
"""

import numpy as np


def _kaiser_sinc(t, half_taps, beta):
    """Kaiser-windowed sinc evaluated at arbitrary (non-integer) offsets t."""
    h = np.sinc(t)
    r = t / half_taps
    inside = np.abs(r) <= 1.0
    w = np.zeros_like(t)
    w[inside] = np.i0(beta * np.sqrt(1.0 - r[inside] ** 2)) / np.i0(beta)
    return h * w


def resample_sinc(x, eps, num_taps=64, beta=8.6, chunk=20000):
    """
    Kaiser-windowed sinc fractional resampler.

    num_taps controls the error floor; beta controls the stopband. The defaults
    (64 taps, beta 8.6) are chosen to put the floor well below 1 ppm -- verify with
    experiments/d02_resampler_accuracy.py rather than trusting this comment.

    Processed in chunks because the naive vectorisation would allocate
    len(x) * num_taps floats at once.
    """
    x = np.asarray(x, dtype=np.float64)
    n_in = len(x)
    half = num_taps // 2

    n_out = int(np.floor((n_in - num_taps - 2) / (1.0 + eps)))
    if n_out <= 0:
        raise ValueError("input too short for the requested number of taps")

    out = np.empty(n_out, dtype=np.float64)
    taps = np.arange(-half + 1, half + 1)

    for start in range(0, n_out, chunk):
        stop = min(start + chunk, n_out)
        pos = np.arange(start, stop) * (1.0 + eps) + half
        base = np.floor(pos).astype(np.int64)
        frac = pos - base

        idx = base[:, None] + taps[None, :]
        t = frac[:, None] - taps[None, :]
        h = _kaiser_sinc(t, half, beta)

        np.clip(idx, 0, n_in - 1, out=idx)
        out[start:stop] = np.einsum("ij,ij->i", x[idx], h)

    return out


def resample_hermite(x, eps):
    """
    Cubic Hermite resampler. Provided as the NEGATIVE CONTROL for the accuracy
    report -- it is fast and visibly insufficient for ppm-level work. Do not use
    it to inject ground truth.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    idx = np.arange(n) * (1.0 + eps)
    idx = idx[idx < n - 2]
    lo = np.floor(idx).astype(int)
    frac = idx - lo
    xm1 = x[np.maximum(lo - 1, 0)]
    x0 = x[lo]
    x1 = x[np.minimum(lo + 1, n - 1)]
    x2 = x[np.minimum(lo + 2, n - 1)]
    a = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
    b = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
    c = -0.5 * xm1 + 0.5 * x1
    return ((a * frac + b) * frac + c) * frac + x0


def measure_tone_frequency(y, fs, f_guess):
    """
    Estimate a pure tone's frequency to far better than one FFT bin, by
    least-squares fitting the unwrapped analytic phase.

    This is the independent yardstick for the resampler: if a tone at f0 is
    resampled by eps, its frequency must become exactly f0*(1+eps). Any deviation
    is resampler error, measured without reference to any SRO estimator -- so the
    check is not circular.
    """
    from scipy.signal import hilbert

    n = len(y)
    # Taper the ends: the analytic signal is unreliable at the boundaries
    edge = int(0.05 * n)
    sl = slice(edge, n - edge)

    ph = np.unwrap(np.angle(hilbert(y)))[sl]
    t = np.arange(n)[sl] / fs
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, ph, rcond=None)[0]
    return slope / (2 * np.pi)
