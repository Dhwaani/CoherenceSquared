"""
Signal model for asynchronous multimodal capture.

Two streams observe the same speech source:

    air[n]  = x[n]                     + noise      (air microphone)
    body[n] = (h_{p(n)} * x)[n]        + noise      (body-conduction sensor)

where p(n) is a latent *signal class* (phoneme-like) that indexes which
relative transfer function is currently in effect. The body stream is then
resampled by (1 + eps) to model an independent sensor clock.

The key structural property -- and the one this project turns on -- is that
h depends on the *content* of the source, not on time. It is therefore
RECURRENT: it returns to the same value whenever the same class recurs.
Time-varying channel models in the communications literature do not have
this property.
"""

import numpy as np
from scipy import signal as sps


def speechlike_source(n_samples, fs, rng, n_formants=4):
    """Broadband excitation shaped by a slowly drifting formant structure."""
    x = rng.standard_normal(n_samples)
    # Voiced-ish spectral tilt
    x = sps.lfilter([1.0], [1.0, -0.95], x)
    # A few resonances so the spectrum is not flat
    for _ in range(n_formants):
        f0 = rng.uniform(300, 3000)
        bw = rng.uniform(80, 250)
        r = np.exp(-np.pi * bw / fs)
        theta = 2 * np.pi * f0 / fs
        a = [1.0, -2 * r * np.cos(theta), r ** 2]
        x = sps.lfilter([1.0 - r], a, x)
    return x / (np.std(x) + 1e-12)


def make_class_filters(n_classes, order, rng, body_lowpass_hz, fs):
    """
    One relative transfer function per signal class.

    Modelled as short random FIR filters cascaded with a lowpass, since
    body-conducted speech is bandlimited (energy concentrated below ~1.5 kHz).
    Each class gets a genuinely different phase response -- that is the
    nuisance term the estimator must contend with.
    """
    lp = sps.firwin(63, body_lowpass_hz, fs=fs)
    filters = []
    for _ in range(n_classes):
        h = rng.standard_normal(order) * np.exp(-np.arange(order) / (order / 3))
        h = np.convolve(h, lp)
        h = h / (np.linalg.norm(h) + 1e-12)
        filters.append(h)
    return filters


def class_sequence(n_samples, fs, rng, n_classes, mean_dur_ms=80.0):
    """
    Piecewise-constant latent class track. Mean segment duration defaults to
    80 ms, roughly a phoneme.
    """
    seq = np.zeros(n_samples, dtype=int)
    i = 0
    while i < n_samples:
        dur = max(1, int(rng.exponential(mean_dur_ms * 1e-3 * fs)))
        seq[i:i + dur] = rng.integers(0, n_classes)
        i += dur
    return seq[:n_samples]


def apply_switching_channel(x, seq, filters):
    """
    Apply a class-dependent filter, crossfading at class boundaries so the
    result contains no artificial discontinuities that a real signal
    would not have.
    """
    n = len(x)
    y = np.zeros(n)
    banks = [np.convolve(x, h)[:n] for h in filters]
    banks = np.stack(banks, axis=0)

    # Smooth the one-hot class track to get crossfade weights (5 ms ramp)
    n_classes = len(filters)
    ramp = int(0.005 * 16000)
    win = np.ones(ramp) / ramp
    for c in range(n_classes):
        w = (seq == c).astype(float)
        w = np.convolve(w, win, mode="same")
        y += w * banks[c]
    return y


def resample_sro(x, eps):
    """
    Resample by (1 + eps) to emulate a sampling-rate offset, using
    band-limited sinc interpolation on a fractional grid.
    """
    n = len(x)
    idx = np.arange(n) * (1.0 + eps)
    idx = idx[idx < n - 2]
    lo = np.floor(idx).astype(int)
    frac = idx - lo
    # Cubic Hermite is accurate enough for |eps| <= 1e-3 and keeps this fast
    xm1 = x[np.maximum(lo - 1, 0)]
    x0 = x[lo]
    x1 = x[np.minimum(lo + 1, n - 1)]
    x2 = x[np.minimum(lo + 2, n - 1)]
    a = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
    b = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
    c = -0.5 * xm1 + 0.5 * x1
    return ((a * frac + b) * frac + c) * frac + x0


def simulate(fs=16000, duration_s=60.0, eps=100e-6, n_classes=8,
             static_channel=False, snr_db=30.0, seed=0,
             mean_dur_ms=80.0, body_lowpass_hz=1500.0):
    """
    Generate one (air, body) pair with a known sampling-rate offset.

    static_channel=True collapses the class-dependent path to a single fixed
    filter -- the control condition that the WASN literature implicitly
    assumes.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)

    x = speechlike_source(n, fs, rng)
    filters = make_class_filters(n_classes, 64, rng, body_lowpass_hz, fs)

    if static_channel:
        seq = np.zeros(n, dtype=int)
        body = np.convolve(x, filters[0])[:n]
    else:
        seq = class_sequence(n, fs, rng, n_classes, mean_dur_ms)
        body = apply_switching_channel(x, seq, filters)

    body = resample_sro(body, eps)
    m = len(body)
    air = x[:m]
    seq = seq[:m]

    def add_noise(s):
        p = np.mean(s ** 2)
        return s + rng.standard_normal(len(s)) * np.sqrt(p / (10 ** (snr_db / 10)))

    return add_noise(air), add_noise(body), seq, fs
