"""L1: MNIST classification with a 1-hidden-layer ReLU MLP trained by Adam.

The autoresearch loop at L1 tunes **hyperparameters only**. Architecture is
locked: one hidden layer, ReLU activation, softmax output, Adam optimizer.
Data, splits, scoring, and the "Adam + 1-hidden + ReLU" choice are part of
the fixed harness — do not modify them mid-loop. The single editable surface
is the `MLPClassifier(...)` kwargs inside `fit_and_predict`. Per-level rules
live in `program.md`.

Primary loop metric: **test_accuracy** on the standard 10k MNIST test set.
A validation slice is also used — sklearn's `MLPClassifier(early_stopping=True)`
carves `validation_fraction` of the 60k MNIST train as an internal val and
stops fitting when val score plateaus. That val drives intra-fit stopping;
the autoresearch loop above the fit pivots keep/discard on test_accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "L1_mnist_mlp"
DATA_HOME = Path("/workspace/data/sklearn-cache")

MODEL_DESCRIPTION = (
    "MLPClassifier(hidden=(64,), lr=1e-3, alpha=1e-4, batch=128, "
    "max_iter=50, early_stopping=True, val_frac=0.1, patience=10)"
)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch MNIST from openml (cached under DATA_HOME) and return the standard
    60k/10k train/test split with pixels scaled to [0, 1]. Fixed harness."""
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home=str(DATA_HOME),
        parser="liac-arff",  # avoids the pandas dep that parser='auto' requires
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """The only function you change between experiments.

    Constraints (enforced by program.md, not by code):
      - hidden_layer_sizes must be a 1-tuple (single hidden layer)
      - activation must be 'relu'
      - solver must be 'adam'
    Everything else is fair game (width, learning_rate_init, alpha, batch_size,
    max_iter, beta_1, beta_2, early_stopping, validation_fraction,
    n_iter_no_change, tol, ...).

    Returns (test_predictions, best_val_score, epochs_actually_run). The val
    score comes from sklearn's internal early-stopping holdout — surfaced for
    context, not used as the loop's keep/discard signal.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=128,
        max_iter=50,
        beta_1=0.9,
        beta_2=0.999,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    best_val = float(model.best_validation_score_)
    return model.predict(X_test), best_val, int(model.n_iter_)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    run_dir = RESULTS_ROOT / started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = load_mnist()
    load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_test_pred, best_val_accuracy, n_iter_run = fit_and_predict(
        X_train, y_train, X_test, args.seed
    )
    fit_seconds = time.perf_counter() - t0

    test_accuracy = float(accuracy_score(y_test, y_test_pred))

    finished_at = datetime.now(UTC)

    metrics = {
        "level": "L1_mnist_mlp",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "config": {
            **vars(args),
            "n_train_full": len(X_train),
            "n_test": len(X_test),
            "n_iter_run": n_iter_run,
        },
        "model": MODEL_DESCRIPTION,
        "metrics": {
            "test_accuracy": test_accuracy,
            "best_val_accuracy": best_val_accuracy,
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    transcript = run_dir / "transcript.md"
    transcript.write_text(
        f"# L1 run @ {started_at.isoformat()}\n\n"
        f"**Model:** {MODEL_DESCRIPTION}\n\n"
        f"**Config:** `{metrics['config']}`\n\n"
        f"**Fit time:** {fit_seconds:.1f}s (load {load_seconds:.1f}s); "
        f"trained {n_iter_run} epochs before early stop\n\n"
        f"**Result:** test_accuracy = {test_accuracy:.4f} "
        f"(best internal val_accuracy = {best_val_accuracy:.4f})\n"
    )

    print(f"L1 metrics → {metrics_path}")
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "test_accuracy": test_accuracy,
                "best_val_accuracy": best_val_accuracy,
                "n_iter_run": n_iter_run,
                "fit_seconds": round(fit_seconds, 2),
                "model": MODEL_DESCRIPTION,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
