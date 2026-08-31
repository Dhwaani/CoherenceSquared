# coherence²

**Blind synchronisation for asynchronous multimodal audio capture.**

When an air microphone and a body-conduction sensor run on independent clocks, the
sampling-rate offset (SRO) between them must be estimated blindly. The wireless
acoustic sensor network (WASN) literature solves exactly this problem — but under
an assumption that does not hold here.


This repository characterises the resulting failure and proposes a fix: accumulate
coherence drift only across frame pairs that share a latent signal class, so the
class-dependent phase term cancels by construction rather than by averaging.

---

## Why this matters

Independent clocks give a combined offset of order 100 ppm (MEMS oscillators are
typically ±50–100 ppm, audio codec crystals ±20–50 ppm). Accumulated inter-channel
phase error is `Δφ = 2π·f·ε·t`. Time until the coherence estimate is not merely
degraded but *inverted*:

| SRO ε | at 500 Hz | at 1500 Hz |
|---|---|---|
| 20 ppm | 50 s | 16.7 s |
| 50 ppm | 20 s | 6.7 s |
| 100 ppm | 10 s | **3.3 s** |
| 150 ppm | 6.7 s | **2.2 s** |

Published air+bone fusion work does not encounter this because it is captured wired,
on a single clock, on a lab bench. [VibOmni](https://arxiv.org/abs/2512.02515) (2025),
the current state of the art, reconciles a 1.6 kHz accelerometer with 16 kHz audio by
choosing STFT window sizes, collects over wired I²C, and contains no discussion of
drift or resynchronisation.

## Result

On a synthetic model with a known SRO of 100 ppm, 60 s observation, 20 seeds:

| Condition | Bias (ppm) | Std (ppm) | RMSE (ppm) |
|---|---|---|---|
| Static channel — *the WASN assumption* | −0.26 | 1.36 | **1.38** |
| Signal-dependent channel — *multimodal reality* | +24.13 | 112.99 | **115.54** |
| + class conditioning — *proposed* | +1.47 | 5.76 | **5.94** |

Three things to note:

1. **The baseline is worse than useless.** At a true SRO of 100 ppm its RMSE is
   115 ppm — larger than the quantity being measured. The same holds at 20, 50 and
   200 ppm, where the error is essentially unchanged (≈113–116 ppm RMSE), because the
   nuisance term is independent of ε.
2. **More data does not help.** Sweeping observation length from 15 s to 240 s leaves
   the baseline error flat (≈70–93 ppm). This is not slow convergence. For a *fixed*
   set of transfer functions — i.e. a fixed speaker and device — the estimator
   converges to the wrong value.
3. **Conditioning recovers almost everything**, using only ~31 % of frame pairs. The
   discarded pairs were not merely uninformative; they were actively harmful.

And the classifier can be poor: conditioning beats the unconditioned baseline until
the class-label error rate exceeds roughly 45 %.

### Unsupervised Discovery Capabilities

Partially. Unsupervised k-means over body-sensor log-band energies (`exp03`),
no phoneme recogniser and no labelled corpus:

| Conditioning | RMSE (ppm) |
|---|---|
| none — baseline | 83.80 |
| unsupervised, K=4 | 77.53 |
| unsupervised, K=8 *(true K)* | **47.48** |
| unsupervised, K=16 | 51.35 |
| oracle labels | 5.86 |

Unsupervised discovery recovers **47 %** of the oracle's advantage. The reason is
measurable rather than mysterious: cluster purity against the true class track is
**0.61**, and `exp02` puts the break-even at roughly 0.55 accuracy. Naive clustering
lands just above the threshold, so it earns a real but partial benefit — exactly what
the label-noise sweep predicts.

That quantified gap — 0.61 achieved versus ~0.85 needed for near-oracle performance —
is a well-posed target for a learned classifier, and it is where machine learning
enters this project with something to do rather than as decoration.

## Status & Scope Boundaries

| Component | State |
|---|---|
| Python reference implementation | **Verified** — results above were produced by running it |
| Signal model | **Synthetic.** Class filters are random FIRs, not measured air-to-body paths |
| Class labels | **Oracle** in the headline result; degradation study in `exp02` |
| MATLAB port | **Unverified** — written but not executed; must be checked against the Python reference |
| FAUST kernel | **Untested** — not compiled; no behavioural verification yet |
| Validation on real speech | **Not done.** This is the next and most important step |

The synthetic result establishes that the mechanism is real and that conditioning
addresses it. It does **not** establish the magnitude on real data. Random FIR class
filters may exaggerate the phase diversity of genuine air-to-body transfer paths, in
which case the effect shrinks. Measured phoneme-dependent transfer functions
([Ohlenbusch et al., 2024](https://arxiv.org/abs/2310.06554)) are the correct next input.

## System Validation & Outputs

The synthetic result above is a feasibility sketch, not evidence about real
signals. [`docs/validation-plan.md`](docs/validation-plan.md) sets out how it gets
tested; these are the runnable pieces.

| ID | Deliverable | Status | Run it |
|---|---|---|---|
| D0.1 | Vibravox loader + manifest | **untested vs live dataset** | `python data/vibravox_loader.py --inspect` |
| D0.2 | Resampler accuracy report | **verified** | `python experiments/d02_resampler_accuracy.py` |
| D0.3 | Phase-diversity measurement | instrument **verified**, awaits data | `python experiments/d03_phase_diversity_report.py --all-sensors` |
| D0.4 | Estimator sweep on real speech | awaits data | `python experiments/d04_estimator_sweep.py` |
| D0.6 | Decision memo | template | [`docs/D0.6-decision-memo-template.md`](docs/D0.6-decision-memo-template.md) |
| D1.1 | Sync-marker / clock-count SRO | **verified** to <0.03 ppm | `python -m tests.test_hwsync` |
| D1.2 | Clock drift statistics | **verified** | same |

Two things worth knowing before running any of it:

**D0.2 gates everything.** Ground truth is injected by resampling, so the
resampler's own error floor must sit well below the offsets being tested. The
verified sinc resampler holds ~100 dB reconstruction SNR across the band; cubic
Hermite falls to 30 dB at 3 kHz, which is why it is kept only as a negative
control. A tone-frequency check alone does not catch this — both resamplers pass
it — so the report uses waveform SNR as the discriminating test.

**D0.3 is the gate, not D0.4.** Phase diversity is the physical quantity the
hypothesis rests on. If it is below ~0.3 rad the effect cannot exist and no
estimator experiment is needed. The interpretation thresholds are fixed in the
script and in the memo template *before* seeing data, deliberately.

## Prerequisites

Python 3.9+, NumPy, SciPy, Matplotlib. No deep learning framework, no Kaldi, no
GPU. Everything below runs on a laptop core.

## Getting started

```bash
git clone https://github.com/Dhwaani/coherence-squared
cd coherence-squared
./run_all.sh              # reproduces every number and figure (~7 min)
./run_all.sh --quick      # regression check only (~10 s)
```

Individual stages:

```bash
cd python
python -m tests.test_regression                  # behavioural check, ~10 s
python experiments/exp01_confound.py             # the confound, ~2.5 min
python experiments/exp02_duration_and_labels.py  # duration, label noise, ~4 min
python experiments/exp03_unsupervised.py         # unsupervised classes, ~25 s
python experiments/make_figure.py
```

## Layout

```
python/cohsq/model.py     signal model: source, class-dependent channel, SRO
python/cohsq/sro.py       baseline (coherence drift) and conditioned estimators
python/experiments/       exp01 (the confound), exp02 (duration, label noise)
matlab/                   port for the primary toolchain — UNVERIFIED
faust/lib/coherence.lib   real-time kernel for embedded targets — UNTESTED
theory/                   the phase decomposition and what remains to be proved
docs/results/             saved JSON results and the figure
```

## Future works

1. Replace random FIR class filters with measured phoneme-dependent transfer
   functions and re-run. **This gates everything else.**
2. Validate on [Vibravox](https://vibravox.cnam.fr/), which is synchronously captured across six body sensors and ships phonemizers — so a *known* SRO can be
   injected into real speech and recovered.
3. Replace oracle labels with broad manner classes estimated from the body sensor
   alone.
4. Derive identifiability conditions and a Cramér–Rao bound (see `theory/`).

## Citation

```bibtex
@software{chakraborty_coherence_squared_2026,
  author  = {Chakraborty, Ashmita},
  title   = {coherence²: blind synchronisation for asynchronous
             multimodal audio capture},
  year    = {2026},
  url     = {https://github.com/Dhwaani/coherence-squared}
}
```

## Contact

Ashmita Chakraborty · [ORCID 0009-0001-5506-9588](https://orcid.org/0009-0001-5506-9588)


## Related work

- Schmalenstroeer & Haeb-Umbach — blind SRO estimation via coherence drift
- Chinaev et al. — double-cross-correlation processor (DXCP)
- Gburrek et al. — WASN synchronisation with time-varying SRO and speaker changes
- Ohlenbusch et al. — speech-dependent own-voice transfer characteristics
