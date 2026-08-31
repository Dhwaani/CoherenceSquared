"""
Deliverable D0.4 -- baseline estimator characterisation on real speech.

Runs the WASN coherence-drift estimator against a KNOWN injected offset across
sensor placements, speakers, offset magnitudes and observation lengths.

The duration sweep is not optional. It separates two outcomes with very different
implications, and reporting only a single duration would conflate them:

    error DECREASES with observation length  -> a convergence-rate problem
    error PERSISTS with observation length   -> a consistency problem

Ground truth comes from cohsq.resample.resample_sinc, whose own error floor must
first be established by d02_resampler_accuracy.py. Run that first.

Usage:
    python experiments/d04_estimator_sweep.py --max-items 50
    python experiments/d04_estimator_sweep.py --all-sensors --max-items 100
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cohsq.resample import resample_sinc                      # noqa: E402
from cohsq.sro import estimate_sro_drift, estimate_sro_conditioned  # noqa: E402
from experiments.d03_phase_diversity_report import phonemes_to_labels  # noqa: E402

EPS_LIST = [20e-6, 50e-6, 100e-6, 200e-6]
DURATIONS = [5.0, 10.0, 20.0, 40.0]


def concat_corpus(sensor, config, split, max_items, fs, min_total_s):
    """
    Concatenate utterances into one long stream per speaker.

    Individual Vibravox utterances are short; the duration sweep needs continuous
    material. Concatenating within a speaker keeps the transfer path consistent,
    which is the property under study -- concatenating ACROSS speakers would mix
    transfer paths and confound the measurement.
    """
    from data.vibravox_loader import load_pairs

    by_speaker = {}
    for item in load_pairs(config=config, split=split, body_sensor=sensor,
                           target_fs=fs, max_items=max_items):
        sid = str(item["speaker_id"])
        lab = phonemes_to_labels(item["phonemes"], len(item["air"]), fs)
        a, b, l = by_speaker.setdefault(sid, ([], [], []))
        a.append(item["air"])
        b.append(item["body"])
        l.append(lab)

    out = {}
    for sid, (a, b, l) in by_speaker.items():
        air = np.concatenate(a)
        body = np.concatenate(b)
        # Keep label spaces disjoint per utterance is wrong -- phonemes recur
        # across utterances and that recurrence is exactly what conditioning
        # exploits. So labels are kept in a shared space.
        lab = np.concatenate(l)
        if len(air) / fs >= min_total_s:
            out[sid] = (air, body, lab)
    return out


def run(sensor, config, split, max_items, fs):
    corpus = concat_corpus(sensor, config, split, max_items, fs,
                           min_total_s=max(DURATIONS) + 5.0)
    if not corpus:
        return None

    rows = []
    for eps in EPS_LIST:
        for dur in DURATIONS:
            base_err, cond_err = [], []
            for sid, (air, body, lab) in corpus.items():
                n = int(dur * fs) + 4096
                if len(air) < n:
                    continue
                a = air[:n]
                b = resample_sinc(body[:n + 512], eps)[:len(a)]
                m = min(len(a), len(b))
                a, b, l = a[:m], b[:m], lab[:m]

                e1 = estimate_sro_drift(a, b, fs)
                if np.isfinite(e1):
                    base_err.append((e1 - eps) * 1e6)
                e2, _ = estimate_sro_conditioned(a, b, fs, l)
                if np.isfinite(e2):
                    cond_err.append((e2 - eps) * 1e6)

            if not base_err:
                continue
            rows.append({
                "sensor": sensor, "eps_ppm": eps * 1e6, "duration_s": dur,
                "n_speakers": len(base_err),
                "baseline_bias_ppm": float(np.mean(base_err)),
                "baseline_rmse_ppm": float(np.sqrt(np.mean(np.square(base_err)))),
                "conditioned_rmse_ppm": (
                    float(np.sqrt(np.mean(np.square(cond_err)))) if cond_err else None),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", default="audio.forehead_accelerometer")
    p.add_argument("--all-sensors", action="store_true")
    p.add_argument("--config", default="speech_clean")
    p.add_argument("--split", default="train")
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--fs", type=int, default=16000)
    a = p.parse_args()

    from data.vibravox_loader import BODY_SENSORS
    sensors = BODY_SENSORS if a.all_sensors else [a.sensor]

    print("D0.4 BASELINE ESTIMATOR CHARACTERISATION")
    print("=" * 76)
    print("known offset injected with the verified sinc resampler; "
          "error in ppm\n")

    all_rows = []
    for s in sensors:
        try:
            rows = run(s, a.config, a.split, a.max_items, a.fs)
        except Exception as e:                       # noqa: BLE001
            print(f"{s}: ERROR {type(e).__name__}: {e}")
            continue
        if not rows:
            print(f"{s}: no usable material")
            continue
        all_rows += rows
        print(f"\n{s}")
        print(f"  {'eps':>7}{'dur':>7}{'base bias':>12}{'base rmse':>12}"
              f"{'cond rmse':>12}")
        for r in rows:
            c = ("%12.2f" % r["conditioned_rmse_ppm"]
                 if r["conditioned_rmse_ppm"] is not None else f"{'-':>12}")
            print(f"  {r['eps_ppm']:>7.0f}{r['duration_s']:>7.0f}"
                  f"{r['baseline_bias_ppm']:>12.2f}"
                  f"{r['baseline_rmse_ppm']:>12.2f}{c}")

    if all_rows:
        print("\n" + "=" * 76)
        print("CONVERGENCE CHECK -- baseline RMSE against observation length")
        print("(decreasing => convergence-rate problem; "
              "flat => consistency problem)")
        for eps in EPS_LIST:
            sel = [r for r in all_rows if abs(r["eps_ppm"] - eps * 1e6) < 1e-6]
            if not sel:
                continue
            byd = {}
            for r in sel:
                byd.setdefault(r["duration_s"], []).append(r["baseline_rmse_ppm"])
            s = "  ".join(f"{d:.0f}s:{np.mean(v):7.1f}"
                          for d, v in sorted(byd.items()))
            print(f"  eps {eps*1e6:>4.0f} ppm   {s}")

    out = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "results",
                                       "d04_estimator_sweep.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
