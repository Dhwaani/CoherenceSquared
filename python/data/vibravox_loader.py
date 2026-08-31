"""
Deliverable D0.1 -- Vibravox corpus loader.

!! NOT YET RUN AGAINST THE LIVE DATASET !!
This was written without network access to HuggingFace. The dataset schema below
(config names, sensor column names) is the expected structure and MUST be checked
on first run. `python data/vibravox_loader.py --inspect` prints the actual schema
so you can correct SENSORS in one place if it differs.

Why Vibravox: it is captured SYNCHRONOUSLY across six body-conduction sensors plus
a reference air microphone. Synchrony is not an inconvenience here -- it is the
whole point. It lets a *known* sampling-rate offset be injected and then recovered,
giving exact ground truth that no natively asynchronous recording could provide.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATASET = "Cnam-LMSSC/vibravox"

# Expected sensor columns. VERIFY WITH --inspect BEFORE TRUSTING.
AIR_SENSOR = "audio.headset_microphone"
BODY_SENSORS = [
    "audio.forehead_accelerometer",
    "audio.temple_vibration_pickup",
    "audio.throat_microphone",
    "audio.soft_in_ear_microphone",
    "audio.rigid_in_ear_microphone",
]
PHONEME_KEY = "phonemized_text"


def inspect(config="speech_clean", split="train"):
    """Print the real schema so the constants above can be corrected."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, config, split=f"{split}[:1]")
    ex = ds[0]
    print(f"dataset : {DATASET}  config={config}  split={split}")
    print(f"features: {len(ds.features)}\n")
    for k, v in ds.features.items():
        print(f"  {k:<42} {type(v).__name__}")
    print("\nfirst example keys:")
    for k in ex:
        val = ex[k]
        if isinstance(val, dict) and "array" in val:
            print(f"  {k:<42} audio, {len(val['array'])} samples "
                  f"@ {val.get('sampling_rate')} Hz")
        else:
            s = str(val)
            print(f"  {k:<42} {s[:60]}")


def _resample_to(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g)


def load_pairs(config="speech_clean", split="train", body_sensor=None,
               target_fs=16000, max_items=None, min_duration_s=2.0):
    """
    Yield dicts with aligned air and body streams at a common rate.

    Yields: {air, body, fs, phonemes, speaker_id, duration_s, sensor}
    """
    from datasets import load_dataset

    body_sensor = body_sensor or BODY_SENSORS[0]
    ds = load_dataset(DATASET, config, split=split, streaming=True)

    n = 0
    for ex in ds:
        if AIR_SENSOR not in ex or body_sensor not in ex:
            raise KeyError(
                f"expected columns not found. Run --inspect and correct "
                f"AIR_SENSOR / BODY_SENSORS. Available: {sorted(ex.keys())}")

        a_raw, b_raw = ex[AIR_SENSOR], ex[body_sensor]
        air = np.asarray(a_raw["array"], dtype=np.float64)
        body = np.asarray(b_raw["array"], dtype=np.float64)

        air = _resample_to(air, a_raw["sampling_rate"], target_fs)
        body = _resample_to(body, b_raw["sampling_rate"], target_fs)

        m = min(len(air), len(body))
        air, body = air[:m], body[:m]
        if m / target_fs < min_duration_s:
            continue

        yield {
            "air": air,
            "body": body,
            "fs": target_fs,
            "phonemes": ex.get(PHONEME_KEY),
            "speaker_id": ex.get("speaker_id", ex.get("gender", "unknown")),
            "duration_s": m / target_fs,
            "sensor": body_sensor,
        }

        n += 1
        if max_items and n >= max_items:
            return


def build_manifest(config="speech_clean", split="train", max_items=200,
                   out="docs/results/d01_manifest.json"):
    """
    Deliverable D0.1: a manifest of what was actually ingested.

    Exit criterion for D0.1 is that these statistics match the published Vibravox
    figures. If they do not, the loader is wrong and everything downstream inherits
    the error.
    """
    stats = {"dataset": DATASET, "config": config, "split": split, "sensors": {}}
    for sensor in BODY_SENSORS:
        durs, speakers = [], set()
        try:
            for item in load_pairs(config, split, sensor, max_items=max_items):
                durs.append(item["duration_s"])
                speakers.add(str(item["speaker_id"]))
        except Exception as e:      # noqa: BLE001 - report and continue
            stats["sensors"][sensor] = {"error": f"{type(e).__name__}: {e}"}
            continue
        stats["sensors"][sensor] = {
            "n_items": len(durs),
            "total_minutes": round(sum(durs) / 60.0, 2),
            "mean_duration_s": round(float(np.mean(durs)), 2) if durs else None,
            "n_speakers": len(speakers),
        }
        print(f"{sensor:<42} {len(durs):>5} items  "
              f"{sum(durs)/60:>7.1f} min  {len(speakers):>3} speakers")

    path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", out))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nwrote {path}")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vibravox loader (D0.1)")
    p.add_argument("--inspect", action="store_true",
                   help="print the real dataset schema and exit")
    p.add_argument("--manifest", action="store_true",
                   help="build the D0.1 ingestion manifest")
    p.add_argument("--config", default="speech_clean")
    p.add_argument("--split", default="train")
    p.add_argument("--max-items", type=int, default=200)
    a = p.parse_args()

    if a.inspect:
        inspect(a.config, a.split)
    elif a.manifest:
        build_manifest(a.config, a.split, a.max_items)
    else:
        p.print_help()
