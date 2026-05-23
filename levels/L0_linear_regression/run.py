"""L0: linear regression — autoresearch loop scaffold.

Generates a fixed synthetic regression problem (y = X @ w + b + noise), fits a
model on the train split, scores MSE and R² on a held-out test split, and
writes metrics to a timestamped directory under results/.

This file is the editable surface of L0. Each autoresearch experiment changes
the model definition inside `fit_and_predict` and nothing else. The data
generation (`make_data`), splits, and metric reporting are part of the fixed
evaluation harness — do not modify them mid-loop. Per-level rules live in
`program.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "L0_linear_regression"

MODEL_DESCRIPTION = "OLS (sklearn LinearRegression)"


def make_data(n: int, d: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed data-generating process. Do not modify during the autoresearch loop."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    b = rng.standard_normal()
    y = X @ w + b + noise * rng.standard_normal(n)
    return X, y


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """The only function you change between experiments."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    run_dir = RESULTS_ROOT / started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_data(args.n, args.d, args.noise, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    y_pred = fit_and_predict(X_train, y_train, X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    finished_at = datetime.now(UTC)

    metrics = {
        "level": "L0_linear_regression",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "config": vars(args),
        "model": MODEL_DESCRIPTION,
        "metrics": {"mse": mse, "r2": r2},
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    transcript = run_dir / "transcript.md"
    transcript.write_text(
        f"# L0 run @ {started_at.isoformat()}\n\n"
        f"**Model:** {MODEL_DESCRIPTION}\n\n"
        f"**Config:** `{vars(args)}`\n\n"
        f"**Result:** MSE = {mse:.6f}, R² = {r2:.6f}\n"
    )

    print(f"L0 metrics → {metrics_path}")
    print(f"RESULT_JSON {json.dumps({'mse': mse, 'r2': r2, 'model': MODEL_DESCRIPTION})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
