"""L0: linear regression wiring test.

Generates synthetic y = X @ w + b + noise, fits sklearn LinearRegression on a
train split, evaluates MSE and R^2 on a held-out test split, and writes results
to both stdout and a timestamped directory under results/.

This is a plumbing check, not a research task. No knobs to tune.
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


def make_data(n: int, d: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    b = rng.standard_normal()
    y = X @ w + b + noise * rng.standard_normal(n)
    return X, y


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

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    finished_at = datetime.now(UTC)

    metrics = {
        "level": "L0_linear_regression",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "config": vars(args),
        "metrics": {"mse": mse, "r2": r2},
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    transcript = run_dir / "transcript.md"
    transcript.write_text(
        f"# L0 run @ {started_at.isoformat()}\n\n"
        f"**Hypothesis:** none — this is a wiring test.\n\n"
        f"**Config:** `{vars(args)}`\n\n"
        f"**Result:** MSE = {mse:.6f}, R² = {r2:.6f}\n\n"
        f"**Attribution:** synthetic data is linear with Gaussian noise sigma={args.noise}; "
        f"OLS recovers w and b up to noise, so R^2 should be near 1 and MSE near sigma^2.\n"
    )

    print(f"L0 metrics → {metrics_path}")
    print(f"mse={mse:.6f} r2={r2:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
