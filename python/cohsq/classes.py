"""
Signal-class estimation for conditioned SRO estimation.

The theory says: condition coherence accumulation on latent signal class, and
the class-dependent phase term cancels exactly. It does NOT say the classes must
be phonemes, or that they must be labelled by a supervised recogniser.

What the estimator actually requires is weaker, and worth stating precisely:

    the class assignment must be CONSISTENT -- the same acoustic state must
    receive the same label whenever it recurs.

It does not need to be linguistically meaningful. That opens the door to
discovering the classes without labels at all, which removes the dependency on
a phoneme recogniser (and therefore on language, and on a labelled corpus).

`discover_classes` clusters short-time log-band energies of the BODY sensor
alone. Body-sensor spectra are near-immune to the acoustic environment -- the
sensor does not hear the room -- so the clustering is stable in noise where an
air-microphone clustering would not be. That is the same property that makes
the modality valuable in the first place, reused for a second purpose.

Clustering on magnitude spectra is also insensitive to the sampling-rate offset
itself, so there is no circularity: we are not using timing to recover timing.
"""

import numpy as np


def band_features(x, fs, n_fft=1024, hop=256, n_bands=20,
                  fmin=80.0, fmax=1600.0):
    """Log energy in log-spaced bands, per frame. Cheap enough for an M4."""
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = np.hanning(n_fft + 1)[:-1]
    S = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2

    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    kedge = np.clip((edges * n_fft / fs).astype(int), 1, S.shape[1] - 1)
    feats = np.stack([S[:, kedge[i]:max(kedge[i] + 1, kedge[i + 1])].mean(axis=1)
                      for i in range(n_bands)], axis=1)
    feats = np.log(feats + 1e-12)
    # Per-frame mean removal: we want spectral SHAPE, not level. Level tracks
    # loudness; shape tracks which transfer function is in effect.
    feats -= feats.mean(axis=1, keepdims=True)
    return feats


def kmeans(X, k, n_iter=40, seed=0):
    """Plain Lloyd's algorithm with k-means++ seeding. No sklearn dependency."""
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        p = d / (d.sum() + 1e-12)
        centers.append(X[rng.choice(n, p=p)])
    C = np.array(centers)

    lab = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return lab, C


def discover_classes(body, fs, n_classes=8, n_fft=1024, hop=256,
                     median_len=5, seed=0):
    """
    Unsupervised per-sample class track from the body sensor alone.

    Returns a label per sample, so it drops straight into
    `estimate_sro_conditioned` in place of oracle labels.
    """
    feats = band_features(body, fs, n_fft=n_fft, hop=hop)
    lab, _ = kmeans(feats, n_classes, seed=seed)

    # Median-smooth along time: real acoustic states persist for tens of
    # milliseconds, so isolated single-frame flips are clustering noise.
    if median_len > 1:
        pad = median_len // 2
        padded = np.pad(lab, pad, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(padded, median_len)
        lab = np.median(win, axis=1).astype(int)

    # Expand frame labels back to per-sample
    out = np.zeros(len(body), dtype=int)
    for i, l in enumerate(lab):
        out[i * hop:(i + 1) * hop] = l
    out[len(lab) * hop:] = lab[-1] if len(lab) else 0
    return out
