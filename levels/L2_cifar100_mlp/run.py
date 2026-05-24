"""L2: CIFAR-100 classification with a 1-hidden-layer ReLU MLP trained by Adam.

Same autoresearch contract as L1, harder dataset. Architecture stays locked:
one hidden layer, ReLU, softmax over 100 classes, Adam optimizer. The single
editable surface is the `build_and_train(...)` config dict. Data, splits,
training loop scaffolding, and the architecture/optimizer-family choices are
part of the fixed harness. Per-level rules live in `program.md`.

Primary loop metric: **test_accuracy** on the standard 10k CIFAR-100 test set.
A val slice (carved from the 50k train) drives early stopping inside each fit;
the autoresearch loop above the fit pivots keep/discard on test_accuracy.

Runs on the A40 GPU when CUDA is available, falls back to CPU otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "L2_cifar100_mlp"
DATA_HOME = Path("/workspace/data/torchvision")

INPUT_DIM = 3 * 32 * 32  # CIFAR-100 RGB 32x32 flattened
NUM_CLASSES = 100


@dataclass
class TrainConfig:
    """Editable surface. Tune these between experiments; nothing else."""

    hidden_width: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    val_fraction: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999


def load_cifar100() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed harness: load CIFAR-100, scale pixels to [0,1], flatten, return
    standard 50k train / 10k test as four contiguous CPU tensors. The training
    loop moves them to the GPU once at startup so per-batch transfers don't
    dominate the wall clock."""
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    train = datasets.CIFAR100(root=str(DATA_HOME), train=True, download=True, transform=tx)
    test = datasets.CIFAR100(root=str(DATA_HOME), train=False, download=True, transform=tx)

    def materialize(ds):
        xs = torch.stack([x for x, _ in ds])
        ys = torch.tensor([y for _, y in ds], dtype=torch.long)
        return xs, ys

    X_train, y_train = materialize(train)
    X_test, y_test = materialize(test)
    return X_train, y_train, X_test, y_test


class MLP(nn.Module):
    """Locked architecture: input -> hidden (ReLU) -> 100-class logits.

    This class is part of the fixed harness. Do not add layers, swap
    activations, or change the topology. To search architectures, that is L3.
    """

    def __init__(self, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, hidden_width)
        self.fc2 = nn.Linear(hidden_width, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eval_batch: int = 4096) -> float:
    """Accuracy over (X, y) — assumes both tensors already live on the same
    device as the model. Chunked to keep peak memory bounded for big test sets."""
    model.eval()
    correct = 0
    for i in range(0, len(X), eval_batch):
        preds = model(X[i : i + eval_batch]).argmax(dim=1)
        correct += int((preds == y[i : i + eval_batch]).sum())
    return correct / len(X)


def build_and_train(
    X_train_full: torch.Tensor,
    y_train_full: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[float, float, int]:
    """The only function you change between experiments — and only by editing
    the `TrainConfig` defaults above.

    Constraints (enforced by program.md, not by code):
      - one hidden ReLU layer (MLP class is part of the fixed harness)
      - Adam optimizer (no SGD/L-BFGS swap)
      - softmax / cross-entropy loss

    Returns (test_accuracy, best_val_accuracy, epochs_actually_run).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Move everything to GPU once. Stable train/val split via a shuffled
    # permutation of train indices seeded for reproducibility.
    X_train_full = X_train_full.to(device)
    y_train_full = y_train_full.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    n_total = len(X_train_full)
    n_val = int(n_total * cfg.val_fraction)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val].to(device)
    train_idx = perm[n_val:].to(device)
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    model = MLP(cfg.hidden_width).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )

    best_val = -1.0
    best_state: dict | None = None
    epochs_since_best = 0
    epochs_run = 0
    n_train = len(X_tr)

    epoch_gen = torch.Generator(device=device).manual_seed(seed)
    for _epoch in range(1, cfg.max_epochs + 1):
        epochs_run = _epoch
        model.train()
        # Shuffle on-device, batch via slicing — no host->device transfer per step.
        shuffle = torch.randperm(n_train, generator=epoch_gen, device=device)
        for start in range(0, n_train, cfg.batch_size):
            idx = shuffle[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                break

    assert best_state is not None  # at least one epoch ran
    model.load_state_dict(best_state)
    test_acc = evaluate(model, X_test, y_test)
    return test_acc, best_val, epochs_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started_at = datetime.now(UTC)
    run_dir = RESULTS_ROOT / started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = load_cifar100()
    load_seconds = time.perf_counter() - t0

    cfg = TrainConfig()
    t0 = time.perf_counter()
    test_accuracy, best_val_accuracy, n_iter_run = build_and_train(
        X_train, y_train, X_test, y_test, cfg, device, args.seed
    )
    fit_seconds = time.perf_counter() - t0

    finished_at = datetime.now(UTC)

    metrics = {
        "level": "L2_cifar100_mlp",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "load_seconds": load_seconds,
        "fit_seconds": fit_seconds,
        "device": str(device),
        "config": {
            **vars(args),
            **asdict(cfg),
            "n_train_full": len(X_train),
            "n_test": len(X_test),
            "n_iter_run": n_iter_run,
        },
        "model": f"MLP(hidden={cfg.hidden_width}) Adam(lr={cfg.learning_rate}, wd={cfg.weight_decay})",
        "metrics": {
            "test_accuracy": float(test_accuracy),
            "best_val_accuracy": float(best_val_accuracy),
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    transcript = run_dir / "transcript.md"
    transcript.write_text(
        f"# L2 run @ {started_at.isoformat()}\n\n"
        f"**Device:** {device}\n\n"
        f"**Config:** `{metrics['config']}`\n\n"
        f"**Fit time:** {fit_seconds:.1f}s (load {load_seconds:.1f}s); "
        f"trained {n_iter_run} epochs before early stop\n\n"
        f"**Result:** test_accuracy = {test_accuracy:.4f} "
        f"(best val_accuracy = {best_val_accuracy:.4f})\n"
    )

    print(f"L2 metrics → {metrics_path}")
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "test_accuracy": float(test_accuracy),
                "best_val_accuracy": float(best_val_accuracy),
                "n_iter_run": n_iter_run,
                "fit_seconds": round(fit_seconds, 2),
                "device": str(device),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
