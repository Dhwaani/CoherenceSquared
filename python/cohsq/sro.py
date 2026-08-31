"""
Sampling-rate-offset estimators.

`estimate_sro_drift` is a faithful reimplementation of the coherence-drift
family used throughout the wireless-acoustic-sensor-network literature
(Schmalenstroeer & Haeb-Umbach; Chinaev et al., DXCP). It is the baseline
this project measures against.

`estimate_sro_conditioned` is the contribution: identical arithmetic, but
coherence drift is accumulated only across frame pairs believed to share the
same latent signal class, so the class-dependent phase term cancels by
construction rather than by averaging.
"""

import numpy as np


def _stft(x, n_fft, hop):
    win = np.hanning(n_fft + 1)[:-1]
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1)


def coherence_drift(air, body, n_fft=1024, hop=256, lag=8):
    """
    Per-frame-pair coherence drift.

    Gamma_l  = X_l * conj(Y_l)                    (cross spectrum)
    Psi_l    = Gamma_{l+lag} * conj(Gamma_l)      (drift between two lags)

    With an SRO of eps, arg(Psi_l[k]) = 2*pi*k*eps*lag*hop/n_fft, independent
    of l. Any class-dependent transfer function contributes an ADDITIONAL
    phase equal to the difference of its phase response at the two frames --
    zero when both frames share a class, arbitrary when they do not.
    """
    X = _stft(air, n_fft, hop)
    Y = _stft(body, n_fft, hop)
    n = min(len(X), len(Y))
    G = X[:n] * np.conj(Y[:n])
    return G[lag:] * np.conj(G[:-lag])


def _fit_eps(psi_mean, n_fft, hop, lag, kmin, kmax):
    """
    Weighted least-squares fit of a linear phase ramp across bins.
    Slope s satisfies arg(psi[k]) = s*k, and eps = s*n_fft/(2*pi*lag*hop).
    """
    k = np.arange(kmin, kmax)
    seg = psi_mean[kmin:kmax]
    w = np.abs(seg)
    if w.sum() <= 0:
        return np.nan
    # Unwrap along frequency, anchored at the low end where phase is smallest
    ph = np.unwrap(np.angle(seg))
    # Weighted LS through the origin
    s = np.sum(w * k * ph) / np.sum(w * k * k)
    # Sign convention: a body stream resampled by (1+eps) runs ahead of the
    # air stream, so a positive eps produces a negative phase slope here.
    return -s * n_fft / (2 * np.pi * lag * hop)


def estimate_sro_drift(air, body, fs, n_fft=1024, hop=256, lag=8,
                       fmin=100.0, fmax=1500.0):
    """Baseline: average coherence drift over ALL frame pairs."""
    psi = coherence_drift(air, body, n_fft, hop, lag)
    kmin = max(1, int(fmin * n_fft / fs))
    kmax = int(fmax * n_fft / fs)
    return _fit_eps(psi.mean(axis=0), n_fft, hop, lag, kmin, kmax)


def estimate_sro_conditioned(air, body, fs, labels, n_fft=1024, hop=256,
                             lag=8, fmin=100.0, fmax=1500.0):
    """
    Contribution: average coherence drift only over frame pairs whose latent
    class matches at both ends.

    `labels` is a per-sample class track. In deployment this is replaced by a
    broad manner-class estimate derived from the body sensor alone, which is
    noise-immune precisely because the body sensor does not hear the room.
    """
    psi = coherence_drift(air, body, n_fft, hop, lag)
    n_frames = len(psi) + lag

    # Majority class per frame
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.minimum(idx, len(labels) - 1)
    frame_lab = np.array([np.bincount(labels[row]).argmax() for row in idx])

    keep = frame_lab[lag:] == frame_lab[:-lag]
    if keep.sum() < 8:
        return np.nan, 0.0
    kmin = max(1, int(fmin * n_fft / fs))
    kmax = int(fmax * n_fft / fs)
    est = _fit_eps(psi[keep].mean(axis=0), n_fft, hop, lag, kmin, kmax)
    return est, keep.mean()
