"""Render `figures/loop_progress.png` from `results.tsv`.

Plots only the kept commits (the branch trajectory). Discards live in the
results.tsv table and RESULTS.md hypothesis log; including them on the
trajectory line would visually obscure the climb.

Run from the L2 level directory or repo root:
    python levels/L2_cifar100_mlp/plot_results.py
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
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(iters, accs, marker="o", linewidth=2, color="#2b6cb0", zorder=2)
    ax.scatter(iters, accs, s=80, color="#2b6cb0", zorder=3)

    ax.axhline(
        0.30, color="gray", linestyle=":", linewidth=1, label="aspirational ceiling (test=0.30)"
    )
    ax.axhline(0.01, color="gray", linestyle="--", linewidth=1, label="chance floor (test=0.01)")

    # Stagger labels above/below to avoid collisions on the steep climb.
    placements = [
        (15, -45),  # iter 0 (floor) — below, right
        (15, 25),  # iter 1 — above, right
        (15, 25),  # iter 2
        (15, -45),  # iter 3 — below
        (15, 25),  # iter 4
        (-15, -55),  # iter 5 — left and below (final point near right edge)
    ]
    has_alignments = ["left", "left", "left", "left", "left", "right"]
    for i, (x, y, txt) in enumerate(zip(iters, accs, labels, strict=True)):
        offset = placements[i] if i < len(placements) else (15, 25)
        ha = has_alignments[i] if i < len(has_alignments) else "left"
        ax.annotate(
            txt,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#888"},
            arrowprops={"arrowstyle": "->", "color": "#888"},
        )

    ax.set_xlabel("kept-commit iteration on autoresearch/L2-may23")
    ax.set_ylabel("test_accuracy (CIFAR-100, 10k held-out)")
    ax.set_title("L2 CIFAR-100 shallow MLP — autoresearch loop progress")
    ax.set_xticks(iters)
    ax.set_xlim(-0.5, len(iters) - 0.5)
    ax.set_ylim(-0.04, 0.36)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
