"""Evaluator for L1_mnist_mlp.

Reads a `metrics.json` produced by levels/L1_mnist_mlp/run.py and returns the
primary metric (test-set classification accuracy). The validation accuracy
used by `MLPClassifier`'s internal early-stopping is also surfaced in
`all_metrics` for context — it is not the loop's optimization signal.

NOTE: optimizing the loop directly on test_accuracy means the test set is no
longer held out from hyperparameter search. This is a deliberate choice for
L1 (whose lesson is loop discipline, not proxy-gaming resistance — that's
L4). At L4 we expect this exact pattern to be flagged as the failure mode.

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
