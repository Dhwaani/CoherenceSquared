#!/usr/bin/env bash
# coherence-squared — runnable deliverables.
#
#   ./run_all.sh              instrument verification, no data needed (~2 min)
#   ./run_all.sh --quick      fastest sanity check (~20 s)
#   ./run_all.sh --vibravox   the real Phase 0 run (needs the dataset)
#
# The default target verifies the INSTRUMENTS against known inputs. It requires no
# corpus and no hardware, and it is what should pass before any real data is
# touched: a tool that cannot recover a known answer cannot be trusted with an
# unknown one.

set -euo pipefail
cd "$(dirname "$0")/python"

MODE="${1:-default}"

echo "==> dependencies"
python3 -c "import numpy, scipy, matplotlib" 2>/dev/null \
    || pip install -q -r requirements.txt

if [[ "$MODE" == "--quick" ]]; then
    echo; echo "==> behavioural regression"
    python3 -m tests.test_regression
    echo; echo "==> D0.3 instrument self-test"
    python3 -m tests.test_phase_diversity
    exit 0
fi

if [[ "$MODE" == "--vibravox" ]]; then
    echo
    echo "==> D0.1  inspect the dataset schema"
    echo "    (correct AIR_SENSOR / BODY_SENSORS in data/vibravox_loader.py"
    echo "     if the printed columns differ)"
    python3 data/vibravox_loader.py --inspect || true

    echo; echo "==> D0.1  ingestion manifest"
    python3 data/vibravox_loader.py --manifest --max-items 200

    echo; echo "==> D0.3  phase diversity  [THE GATE — read the verdict]"
    python3 experiments/d03_phase_diversity_report.py --all-sensors --max-items 100

    echo; echo "==> D0.4  estimator characterisation"
    python3 experiments/d04_estimator_sweep.py --max-items 50

    echo
    echo "Now fill in docs/D0.6-decision-memo-template.md before"
    echo "ordering hardware or filing anything."
    exit 0
fi

echo
echo "==> D0.2  resampler accuracy   [gates all ground-truth injection]"
python3 experiments/d02_resampler_accuracy.py

echo; echo "==> D0.3  phase-diversity instrument self-test"
python3 -m tests.test_phase_diversity

echo; echo "==> D1.1 / D1.2  hardware analysis self-test"
python3 -m tests.test_hwsync

echo; echo "==> behavioural regression"
python3 -m tests.test_regression

echo
echo "instruments verified. Next:"
echo "  ./run_all.sh --vibravox     run Phase 0 on the real corpus"
