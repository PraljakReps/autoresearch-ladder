"""Render `figures/loop_progress.png` from `results.tsv`.

Reads the per-level experiment log (gitignored, session-local) and produces
the committed summary figure: R² vs. commit iteration, annotated with the
model innovation introduced at each step.

Run from the L0 level directory or repo root:
    python levels/L0_linear_regression/plot_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LEVEL_DIR = Path(__file__).resolve().parent
TSV = LEVEL_DIR / "results.tsv"
OUT = LEVEL_DIR / "figures" / "loop_progress.png"


def load_rows() -> list[dict[str, str]]:
    with TSV.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    if not TSV.exists():
        print(f"missing {TSV} — run experiments first", file=sys.stderr)
        return 1

    rows = load_rows()
    iters = list(range(len(rows)))
    r2 = [float(r["metric"]) for r in rows]
    labels = [f"{r['commit']}\n{r['description']}" for r in rows]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(iters, r2, marker="o", linewidth=2, color="#2b6cb0", zorder=2)
    ax.scatter(iters, r2, s=80, color="#2b6cb0", zorder=3)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="noise ceiling (R²=1)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label="mean-predictor floor (R²=0)")

    # Place annotation to the right of the first point, to the left of the last,
    # so neither runs off the axis. Vertical offset keeps the box clear of the line.
    for i, (x, y, txt) in enumerate(zip(iters, r2, labels, strict=True)):
        if i == len(iters) - 1:
            offset, ha = (-15, -45), "right"
        else:
            offset, ha = (15, 30), "left"
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

    ax.set_xlabel("commit iteration on autoresearch/L0-may23")
    ax.set_ylabel("R² (test split)")
    ax.set_title("L0 linear regression — autoresearch loop progress")
    ax.set_xticks(iters)
    ax.set_xlim(-0.5, len(iters) - 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
