"""
Deliverable D0.3 -- phase diversity report on real speech.

Run this BEFORE any estimator experiment. It measures the physical quantity the
whole hypothesis depends on: how much the air-to-body transfer function's PHASE
varies across phonemes, per sensor placement.

Interpretation, decided in advance so the result cannot be rationalised after the
fact:

    spread << 0.3 rad   the class-dependent term is too small to corrupt offset
                        estimation. The hypothesis is dead. Publish the
                        measurement as a negative result and stop.

    spread ~ 0.3-1 rad  comparable to the offset-induced term over practical
                        windows. Proceed to D0.4.

    spread >> 1 rad     dominates. Proceed, and expect a large effect.

Usage:
    python experiments/d03_phase_diversity_report.py --sensor forehead_accelerometer
    python experiments/d03_phase_diversity_report.py --all-sensors --max-items 100
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cohsq.phase_diversity import phase_diversity, summarise  # noqa: E402


def phonemes_to_labels(phoneme_string, n_samples, fs):
    """
    Map Vibravox phoneme annotation onto a per-sample class track.

    Vibravox ships phonemized text rather than time-aligned phones, so this uses
    uniform division as a first approximation. THAT IS A REAL LIMITATION: uniform
    division misassigns boundaries and will UNDERSTATE the measured diversity,
    because frames get mixed-class labels.

    Replace with forced alignment (MFA, or the vibravox phonemizer models) before
    reporting a final number. If the diversity is already large under uniform
    division, the conclusion is safe -- the true value can only be larger.
    """
    if not phoneme_string:
        return np.zeros(n_samples, dtype=int)
    phones = [p for p in str(phoneme_string).split() if p]
    if not phones:
        return np.zeros(n_samples, dtype=int)
    vocab = {p: i for i, p in enumerate(sorted(set(phones)))}
    ids = np.array([vocab[p] for p in phones])
    edges = np.linspace(0, n_samples, len(ids) + 1).astype(int)
    lab = np.zeros(n_samples, dtype=int)
    for i, pid in enumerate(ids):
        lab[edges[i]:edges[i + 1]] = pid
    return lab


def run_sensor(sensor, config, split, max_items, fs):
    from data.vibravox_loader import load_pairs

    spreads, n_used, classes_seen = [], 0, []
    for item in load_pairs(config=config, split=split, body_sensor=sensor,
                           target_fs=fs, max_items=max_items):
        lab = phonemes_to_labels(item["phonemes"], len(item["air"]), fs)
        if len(np.unique(lab)) < 2:
            continue
        try:
            r = phase_diversity(item["air"], item["body"], lab, fs)
        except ValueError:
            continue
        spreads.append(r["spread_rad"])
        classes_seen.append(r["n_classes"])
        n_used += 1

    if not spreads:
        return None

    freqs = np.fft.rfftfreq(1024, 1.0 / fs)
    mean_spread = np.mean(np.stack(spreads), axis=0)
    band = (freqs >= 100) & (freqs <= 1500)
    return {
        "sensor": sensor,
        "n_items": n_used,
        "mean_classes_per_item": float(np.mean(classes_seen)),
        "freqs": freqs.tolist(),
        "mean_spread_rad": mean_spread.tolist(),
        "band_mean_rad": float(np.mean(mean_spread[band])),
        "band_mean_deg": float(np.degrees(np.mean(mean_spread[band]))),
        "band_max_rad": float(np.max(mean_spread[band])),
    }


def verdict(band_mean_rad):
    if band_mean_rad < 0.3:
        return "HYPOTHESIS NOT SUPPORTED -- spread too small to matter"
    if band_mean_rad < 1.0:
        return "PROCEED -- spread comparable to the offset term"
    return "PROCEED -- spread dominates the offset term"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", default="audio.forehead_accelerometer")
    p.add_argument("--all-sensors", action="store_true")
    p.add_argument("--config", default="speech_clean")
    p.add_argument("--split", default="train")
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--fs", type=int, default=16000)
    a = p.parse_args()

    from data.vibravox_loader import BODY_SENSORS
    sensors = BODY_SENSORS if a.all_sensors else [a.sensor]

    print("D0.3 PHASE DIVERSITY REPORT")
    print("=" * 70)
    print("across-phoneme circular spread of the air-to-body transfer phase,")
    print("averaged over the 100-1500 Hz body-conduction band\n")
    print(f"  {'sensor':<40}{'items':>7}{'spread':>12}")

    results = []
    for s in sensors:
        try:
            r = run_sensor(s, a.config, a.split, a.max_items, a.fs)
        except Exception as e:                      # noqa: BLE001
            print(f"  {s:<40}{'ERROR':>7}  {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"  {s:<40}{'0':>7}  no usable items")
            continue
        results.append(r)
        print(f"  {s:<40}{r['n_items']:>7}{r['band_mean_deg']:>10.1f} deg")

    if results:
        best = max(r["band_mean_rad"] for r in results)
        print("\n" + "=" * 70)
        print(f"largest spread across placements: {np.degrees(best):.1f} deg "
              f"({best:.2f} rad)")
        print(f"VERDICT: {verdict(best)}")

    out = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "results",
                                       "d03_phase_diversity.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
