"""
Deliverable D0.2 -- resampler accuracy report.

EXIT CRITERION (from the validation plan): injected and recovered epsilon must
agree to better than 1 ppm, and the resampler error floor must sit at least an
order of magnitude below the smallest offset under test.

The check is deliberately independent of any SRO estimator. A pure tone at f0
resampled by eps must land at exactly f0*(1+eps); its frequency is measured by
least-squares phase regression, which has nothing to do with coherence drift. So
this validates the instrument without assuming the thing being tested.

Run this BEFORE any Vibravox work. If it fails, nothing downstream means anything.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohsq.resample import (resample_sinc, resample_hermite,
                            measure_tone_frequency)

FS = 16000.0
DURATION = 20.0
EPS_LIST = [10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
TONES = [300.0, 1000.0, 3000.0]


def tone_error_ppm(resampler, f0, eps, **kw):
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * f0 * t)
    y = resampler(x, eps, **kw)
    f_meas = measure_tone_frequency(y, FS, f0)
    f_true = f0 * (1.0 + eps)
    return (f_meas / f_true - 1.0) * 1e6


def reconstruction_snr_db(f0, eps, num_taps=64, resampler=None):
    """
    Second, stricter check: compare the resampled tone against the analytically
    correct tone sample by sample. Catches distortion and aliasing that a
    frequency-only check would miss.

    This is the check that actually discriminates. A resampler can place a tone at
    exactly the right FREQUENCY while badly distorting its WAVEFORM -- and it is
    waveform error, not frequency error, that corrupts the inter-channel phase
    these estimators depend on.
    """
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * f0 * t)
    if resampler is None:
        y = resample_sinc(x, eps, num_taps=num_taps)
        # resample_sinc reads from pos = n*(1+eps) + half, i.e. it carries a
        # deliberate group delay of half = num_taps//2 input samples. The
        # reference must be aligned to it or the comparison measures the delay
        # rather than the resampling error.
        delay = num_taps // 2
    else:
        y = resampler(x, eps)
        delay = 0
    m = len(y)
    ideal = np.sin(2 * np.pi * f0 * ((1.0 + eps) * np.arange(m) + delay) / FS)
    edge = int(0.05 * m)
    a, b = y[edge:m - edge], ideal[edge:m - edge]
    err = a - b
    return 10 * np.log10(np.sum(b ** 2) / (np.sum(err ** 2) + 1e-30))


def main():
    report = {"fs": FS, "duration_s": DURATION, "sinc": {}, "hermite": {},
              "taps_sweep": {}, "snr_db": {}}

    print("D0.2 RESAMPLER ACCURACY REPORT")
    print("=" * 62)
    print(f"fs = {FS:.0f} Hz, tone length = {DURATION:.0f} s\n")

    print("Frequency error in ppm (should be ~0; this IS the error floor)\n")
    print(f"  {'eps ppm':>9}  {'tone Hz':>8}  {'sinc-64':>12}  {'hermite':>12}")
    for eps in EPS_LIST:
        for f0 in TONES:
            e_s = tone_error_ppm(resample_sinc, f0, eps)
            e_h = tone_error_ppm(resample_hermite, f0, eps)
            report["sinc"][f"{eps*1e6:.0f}_{f0:.0f}"] = e_s
            report["hermite"][f"{eps*1e6:.0f}_{f0:.0f}"] = e_h
            print(f"  {eps*1e6:>9.0f}  {f0:>8.0f}  {e_s:>12.4f}  {e_h:>12.4f}")

    print("\nTap-count sweep at eps = 100 ppm, 3 kHz tone")
    print(f"  {'taps':>6}  {'freq err ppm':>14}  {'recon SNR dB':>14}")
    for taps in [16, 32, 64, 128]:
        e = tone_error_ppm(resample_sinc, 3000.0, 100e-6, num_taps=taps)
        s = reconstruction_snr_db(3000.0, 100e-6, num_taps=taps)
        report["taps_sweep"][str(taps)] = {"freq_err_ppm": e, "snr_db": s}
        print(f"  {taps:>6}  {e:>14.4f}  {s:>14.1f}")

    print("\nWaveform reconstruction SNR at eps = 100 ppm (the discriminating check)")
    print(f"  {'resampler':>14}  {'tone Hz':>8}  {'recon SNR dB':>14}")
    for f0 in TONES:
        s_sinc = reconstruction_snr_db(f0, 100e-6)
        s_herm = reconstruction_snr_db(f0, 100e-6, resampler=resample_hermite)
        report["snr_db"][f"{f0:.0f}"] = {"sinc64": s_sinc, "hermite": s_herm}
        print(f"  {'sinc-64':>14}  {f0:>8.0f}  {s_sinc:>14.1f}")
        print(f"  {'hermite':>14}  {f0:>8.0f}  {s_herm:>14.1f}")

    worst = max(abs(v) for v in report["sinc"].values())
    print("\n" + "=" * 62)
    print(f"worst-case sinc-64 frequency error: {worst:.4f} ppm")
    smallest_eps = min(EPS_LIST) * 1e6
    print(f"smallest offset under test:         {smallest_eps:.0f} ppm")
    print(f"margin:                             {smallest_eps/max(worst,1e-9):.0f}x")

    ok = worst < 1.0 and (smallest_eps / max(worst, 1e-9)) >= 10
    report["worst_sinc_ppm"] = worst
    report["pass"] = bool(ok)
    print("\nEXIT CRITERION: " + ("PASS" if ok else "FAIL"))
    if not ok:
        print("  increase num_taps, or reduce the smallest eps under test")

    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "docs", "results", "d02_resampler_accuracy.json")
    with open(os.path.abspath(out), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.abspath(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
