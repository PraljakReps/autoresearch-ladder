"""Evaluator for L0_linear_regression.

Reads a `metrics.json` produced by levels/L0_linear_regression/run.py and
returns the primary metric (R²). Importable as `score(path) -> dict` and also
usable from the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRIMARY_METRIC = "r2"


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
