"""L3: CIFAR-100 classification with open-architecture search (Adam, no schedule).

L2 capped a shallow ReLU MLP on raw flattened pixels at test_accuracy = 0.2677.
L3 opens the architecture surface: depth, width, conv vs. linear vs. RNN,
activations, normalization, dropout, residual connections — all editable.
Per-channel image normalization is also editable. The optimizer family stays
Adam with no learning-rate schedule, and data augmentation beyond normalization
is off limits (see program.md for the full contract).

The editable surface is:
  - `build_model(model_cfg)` — the architecture
  - `ModelConfig` — architecture hyperparameters
  - `TrainConfig.data_normalization` — `"none"` or `"channel"`
  - other `TrainConfig` fields (lr, wd, batch, epochs, patience, betas)

Floor (iter 0): deliberately-broken MLP — hidden=16, max_epochs=2, raw pixels,
no normalization. Expected near-chance test_accuracy (~0.01-0.05), matching the
L0/L1/L2 convention of starting from a clearly-undertrained baseline so the
loop visibly climbs. Iter 1 and 2 will recover toward L2's wrap (hidden 16→1024
and max_epochs 2→50) before real architecture search begins at iter 3.

Runs on the A40 GPU when CUDA is available, falls back to CPU otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "L3_cifar100_arch"
DATA_HOME = Path("/workspace/data/torchvision")

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 100

CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


@dataclass
class ModelConfig:
    """Editable surface: architecture knobs. Add/rename fields as experiments
    require (e.g. `dropout_p`, `n_conv_blocks`, `use_batchnorm`)."""

    conv_channels: tuple[int, ...] = (64, 128, 256, 512)
    head_hidden: int = 256
    use_batchnorm: bool = True


@dataclass
class TrainConfig:
    """Editable surface: training-loop knobs and data prep."""

    data_normalization: str = "none"  # "none" | "channel"

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999

    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    val_fraction: float = 0.1

    model: ModelConfig = field(default_factory=ModelConfig)


def load_cifar100() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed harness: load CIFAR-100, return NCHW float tensors in [0, 1].

    Returns shape (N, 3, 32, 32). Normalization (if any) is applied inside
    `build_and_train` after data is on device.
    """
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    tx = transforms.ToTensor()
    train = datasets.CIFAR100(root=str(DATA_HOME), train=True, download=True, transform=tx)
    test = datasets.CIFAR100(root=str(DATA_HOME), train=False, download=True, transform=tx)

    def materialize(ds):
        xs = torch.stack([x for x, _ in ds])
        ys = torch.tensor([y for _, y in ds], dtype=torch.long)
        return xs, ys

    X_train, y_train = materialize(train)
    X_test, y_test = materialize(test)
    return X_train, y_train, X_test, y_test


def apply_normalization(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply data normalization. x is NCHW in [0, 1]. Returns a new tensor."""
    if mode == "none":
        return x
    if mode == "channel":
        mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    raise ValueError(f"unknown data_normalization mode: {mode!r}")


class SimpleConv(nn.Module):
    """Small CNN: stacked Conv->ReLU->MaxPool blocks, then a 2-layer MLP head.

    Each block halves spatial dims (32 -> 16 -> 8 -> 4 for the default
    3-channel tuple). Final feature map flattened and fed to a ReLU MLP head.
    No batchnorm, no dropout, no residuals — those are separate experiments.
    """

    def __init__(self, conv_channels: tuple[int, ...], head_hidden: int, use_batchnorm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        in_c = INPUT_SHAPE[0]
        spatial = INPUT_SHAPE[1]
        for out_c in conv_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_c = out_c
            spatial //= 2
        self.features = nn.Sequential(*layers)
        feat_dim = in_c * spatial * spatial
        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


def build_model(model_cfg: ModelConfig) -> nn.Module:
    """The editable architecture surface. Replace this with whatever you're
    testing this iteration — Conv2d stacks, residual blocks, dropout, norms.

    Contract: takes input shape INPUT_SHAPE (NCHW), returns logits of shape
    (B, NUM_CLASSES). No softmax — cross_entropy applies it.
    """
    return SimpleConv(
        conv_channels=model_cfg.conv_channels,
        head_hidden=model_cfg.head_hidden,
        use_batchnorm=model_cfg.use_batchnorm,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, eval_batch: int = 512) -> float:
    """Accuracy over (X, y). Both already on the model's device."""
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
) -> tuple[float, float, int, int]:
    """Training-loop harness. Edit `build_model` and `TrainConfig` defaults
    between experiments; don't edit this function.

    Returns (test_accuracy, best_val_accuracy, epochs_run, n_params).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_full = X_train_full.to(device)
    y_train_full = y_train_full.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    X_train_full = apply_normalization(X_train_full, cfg.data_normalization)
    X_test = apply_normalization(X_test, cfg.data_normalization)

    n_total = len(X_train_full)
    n_val = int(n_total * cfg.val_fraction)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    val_idx = perm[:n_val].to(device)
    train_idx = perm[n_val:].to(device)
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    model = build_model(cfg.model).to(device)
    n_params = count_params(model)

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

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc = evaluate(model, X_test, y_test)
    return test_acc, best_val, epochs_run, n_params


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
    test_accuracy, best_val_accuracy, n_iter_run, n_params = build_and_train(
        X_train, y_train, X_test, y_test, cfg, device, args.seed
    )
    fit_seconds = time.perf_counter() - t0

    finished_at = datetime.now(UTC)

    metrics = {
        "level": "L3_cifar100_arch",
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
            "n_params": n_params,
        },
        "model": repr(build_model(cfg.model)),
        "metrics": {
            "test_accuracy": float(test_accuracy),
            "best_val_accuracy": float(best_val_accuracy),
        },
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    transcript = run_dir / "transcript.md"
    transcript.write_text(
        f"# L3 run @ {started_at.isoformat()}\n\n"
        f"**Device:** {device}\n\n"
        f"**Config:** `{metrics['config']}`\n\n"
        f"**Fit time:** {fit_seconds:.1f}s (load {load_seconds:.1f}s); "
        f"trained {n_iter_run} epochs before early stop; n_params = {n_params:,}\n\n"
        f"**Result:** test_accuracy = {test_accuracy:.4f} "
        f"(best val_accuracy = {best_val_accuracy:.4f})\n"
    )

    print(f"L3 metrics → {metrics_path}")
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "test_accuracy": float(test_accuracy),
                "best_val_accuracy": float(best_val_accuracy),
                "n_iter_run": n_iter_run,
                "n_params": n_params,
                "fit_seconds": round(fit_seconds, 2),
                "device": str(device),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
