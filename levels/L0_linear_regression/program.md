# L0 — linear regression: program

Per-level instructions for the autoresearch loop. Read this before every L0 experiment.

## Task

Fit `y = X @ w + b + noise` (synthetic, n=1000, d=5, noise=0.1, seed=0) and predict on a held-out test split. The data-generating process and split are fixed in `run.py` and **never** change mid-loop. The only editable surface is `fit_and_predict`.

## Primary metric

- **R²** on the held-out test set, as reported by `evaluators/l0_linear_regression.py`.
- **Direction:** higher is better.
- MSE is also recorded for sanity but does not drive keep/discard.

## Compute budget

Per run: < 1 s wall-clock on CPU. No GPU. No tmux required at this level — the long-job machinery starts mattering at L2+.

## Stop criterion (within-level)

Stop and declare diminishing returns when **any** of:

- R² ≥ 0.99 on the keep branch (essentially at the noise ceiling, since `Var(noise) / Var(y) ≈ 0.01² · 1 / (||w||² + 0.01²) ≪ 0.01`).
- Two consecutive experiments fail to improve R² by ≥ 1e-3.
- Total experiments on the branch ≥ 6.

Whichever triggers first wins. The point of L0 is to exercise the loop, not to chase noise.

## Loop discipline (echoes CLAUDE.md, re-stated here so it's visible at the point of work)

1. State a hypothesis: what change, what you expect, why.
2. Edit `fit_and_predict` only — one variable at a time unless explicitly justified.
3. `git commit` on `autoresearch/L0-<tag>`.
4. `python run.py > run.log 2>&1` (no `tee`; don't flood context).
5. Score via the latest `results/L0_linear_regression/<ts>/metrics.json` (or `grep RESULT_JSON run.log`).
6. Append one row to `levels/L0_linear_regression/results.tsv` (tab-separated, gitignored):
   `commit<TAB>metric<TAB>status<TAB>description`
   where `metric` is R² and `status ∈ {keep, discard, crash}`.
7. Keep → stay on the new commit. Discard → `git reset --hard HEAD~1`.

## What "done" looks like for L0

A short transcript in the session handoff stating: starting R², ending R², number of experiments, and the reason the loop stopped (which clause of the stop criterion fired). The transcript matters more than the final number — L0 is a wiring + discipline test.
