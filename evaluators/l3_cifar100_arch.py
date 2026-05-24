"""Evaluator for L3_cifar100_arch.

Reads a `metrics.json` produced by levels/L3_cifar100_arch/run.py and returns
the primary metric (CIFAR-100 test-set classification accuracy). The
architecture is open at this level; the metric is the same as L2 so wins
attribute to structural change, not metric change.

`best_val_accuracy` is surfaced in `all_metrics` for context — it's the
training-side early-stopping signal, not the autoresearch loop's keep/discard
signal.

Same methodological note as L1, L2: optimizing the loop on `test_accuracy`
means the test set is not held out from architecture search. Acknowledged at
L3 — the lesson here is structural search and stopping discipline. L4 is
where this exact pattern becomes the failure mode under test.

Importable as `score(path) -> dict` and also usable from the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRIMARY_METRIC = "test_accuracy"


def score(metrics_path: str | Path) -> dict:
    data = json.loads(Path(metrics_path).read_text())
    metrics = data["metrics"]
    return {
        "level": data["level"],
        "primary_metric": PRIMARY_METRIC,
        "primary_value": metrics[PRIMARY_METRIC],
        "all_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path, help="Path to metrics.json from a run")
    args = parser.parse_args()

    result = score(args.metrics_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
