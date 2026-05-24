"""Render `figures/loop_progress.png` from `results.tsv`.

Plots only the kept commits (the branch trajectory). Discards live in the
results.tsv table and RESULTS.md hypothesis log; including them on the
trajectory line would visually obscure the climb.

Run from the L3 level directory or repo root:
    python levels/L3_cifar100_arch/plot_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LEVEL_DIR = Path(__file__).resolve().parent
TSV = LEVEL_DIR / "results.tsv"
OUT = LEVEL_DIR / "figures" / "loop_progress.png"


def load_keeps() -> list[dict[str, str]]:
    with TSV.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r["status"] == "keep"]


def short_change(description: str) -> str:
    """First clause of the description, before any colon — that's the knob change."""
    return description.split(":", 1)[0].strip()


def main() -> int:
    if not TSV.exists():
        print(f"missing {TSV} — run experiments first", file=sys.stderr)
        return 1

    keeps = load_keeps()
    iters = list(range(len(keeps)))
    accs = [float(r["test_accuracy"]) for r in keeps]
    labels = [f"{r['commit']}\n{short_change(r['description'])}" for r in keeps]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(iters, accs, marker="o", linewidth=2, color="#2b6cb0", zorder=2)
    ax.scatter(iters, accs, s=80, color="#2b6cb0", zorder=3)

    ax.axhline(
        0.55, color="#c53030", linestyle=":", linewidth=1.2, label="ceiling stop (test=0.55)"
    )
    ax.axhline(0.2677, color="#888", linestyle="--", linewidth=1, label="L2 wrap (test=0.2677)")
    ax.axhline(0.01, color="gray", linestyle="--", linewidth=1, label="chance floor (test=0.01)")

    # Stagger labels above/below to avoid collisions on the steep climb.
    placements = [
        (15, 25),  # iter 0 floor
        (15, 25),  # iter 1 (h=1024)
        (15, -50),  # iter 2 (ep=50) — below
        (15, 25),  # iter 4 (MLP->Conv)
        (15, -50),  # iter 5 (+BN) — below
        (15, 25),  # iter 6 (wider)
        (15, -50),  # iter 7 (deeper) — below
        (-15, 25),  # iter 8 (chnorm) — left
        (-15, -55),  # iter 9 (wider2, final) — left and below
    ]
    has_alignments = ["left"] * 7 + ["right", "right"]
    for i, (x, y, txt) in enumerate(zip(iters, accs, labels, strict=True)):
        offset = placements[i] if i < len(placements) else (15, 25)
        ha = has_alignments[i] if i < len(has_alignments) else "left"
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            ha=ha,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#888"},
            arrowprops={"arrowstyle": "->", "color": "#888"},
        )

    ax.set_xlabel("kept-commit iteration on autoresearch/L3-may24")
    ax.set_ylabel("test_accuracy (CIFAR-100, 10k held-out)")
    ax.set_title("L3 CIFAR-100 architecture-open — autoresearch loop progress")
    ax.set_xticks(iters)
    ax.set_xlim(-0.5, len(iters) - 0.5)
    ax.set_ylim(-0.04, 0.68)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
