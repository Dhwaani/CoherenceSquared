"""
Deliverable D0.3 -- phase diversity of the signal-conditioned transfer path.

This is the most informative measurement in the whole validation plan, and it
should be made BEFORE testing any estimator.

The hypothesis is that the air-to-body transfer function varies with signal class
(phoneme) enough to corrupt blind offset estimation. That variation is quantified
by the CIRCULAR SPREAD of the per-class phase response across classes, as a
function of frequency.

If the spread is small, the effect cannot exist and no estimator experiment is
needed. If it is large, the magnitude tells you how large the effect should be.
Either way this is a publishable measurement of a real physical quantity, and it
stands independently of whether the estimator story works out.

Nothing here depends on any offset estimator, so the measurement is not circular.
"""

import numpy as np


def stft(x, n_fft=1024, hop=256):
    win = np.hanning(n_fft + 1)[:-1]
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def frame_labels(labels, n_frames, n_fft, hop):
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.minimum(idx, len(labels) - 1)
    return np.array([np.bincount(labels[r]).argmax() for r in idx])


def estimate_rtf(air, body, labels, fs, n_fft=1024, hop=256, min_frames=20):
    """
    Least-squares relative transfer function per signal class.

        H_p(f) = E_p[ X*(f) Y(f) ] / E_p[ |X(f)|^2 ]

    Returns (classes, H) where H has shape (n_classes, n_bins), complex.
    Classes with fewer than `min_frames` frames are dropped -- an RTF estimated
    from a handful of frames is noise, and including it would inflate the
    diversity measure with estimation variance rather than real physics.
    """
    X = stft(air, n_fft, hop)
    Y = stft(body, n_fft, hop)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    fl = frame_labels(labels, n, n_fft, hop)

    classes, H = [], []
    for c in np.unique(fl):
        m = fl == c
        if m.sum() < min_frames:
            continue
        num = np.mean(np.conj(X[m]) * Y[m], axis=0)
        den = np.mean(np.abs(X[m]) ** 2, axis=0) + 1e-20
        classes.append(int(c))
        H.append(num / den)
    return np.array(classes), np.array(H)


def circular_spread(phases, weights=None):
    """
    Circular standard deviation of a set of angles, in radians.

        R = |mean(exp(j*theta))|,   spread = sqrt(-2 ln R)

    Zero means all classes share a phase response (no diversity, hypothesis
    dead). Values approaching and beyond 1 rad mean the class-dependent phase
    term is comparable to or larger than the offset-induced term that estimators
    are trying to measure.
    """
    z = np.exp(1j * phases)
    if weights is not None:
        w = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-20)
        R = np.abs(np.sum(w * z, axis=0))
    else:
        R = np.abs(np.mean(z, axis=0))
    R = np.clip(R, 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def phase_diversity(air, body, labels, fs, n_fft=1024, hop=256,
                    weight_by_magnitude=True):
    """
    Full D0.3 measurement.

    Returns a dict with:
      freqs           bin centre frequencies (Hz)
      spread_rad      circular spread of class phase responses, per frequency
      n_classes       how many classes had enough data
      mean_coherence  |H| magnitude spread, for context
    """
    classes, H = estimate_rtf(air, body, labels, fs, n_fft, hop)
    if len(classes) < 2:
        raise ValueError("need at least two well-populated classes")

    ph = np.angle(H)
    w = np.abs(H) if weight_by_magnitude else None
    spread = circular_spread(ph, weights=w)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    return {
        "freqs": freqs,
        "spread_rad": spread,
        "n_classes": len(classes),
        "classes": classes,
        "mag_db": 20 * np.log10(np.abs(H) + 1e-12),
    }


def summarise(result, band=(100.0, 1500.0)):
    """One-line summary over the body-conduction speech band."""
    f = result["freqs"]
    m = (f >= band[0]) & (f <= band[1])
    s = result["spread_rad"][m]
    return {
        "band_hz": band,
        "n_classes": result["n_classes"],
        "mean_spread_rad": float(np.mean(s)),
        "max_spread_rad": float(np.max(s)),
        "mean_spread_deg": float(np.degrees(np.mean(s))),
    }
