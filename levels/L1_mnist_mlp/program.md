# L1 — MNIST classification (1-hidden ReLU MLP, Adam): program

Per-level instructions for the autoresearch loop. Read this before every L1 experiment.

## Task

Classify the 10 MNIST digit classes from raw 784-dim pixel vectors (scaled to [0, 1]) using a **single-hidden-layer ReLU MLP trained with Adam**. The architecture, activation, optimizer family, data, and splits are part of the fixed harness. The loop tunes hyperparameters only.

## Fixed surface (do not modify mid-loop)

- **Data:** MNIST via `fetch_openml('mnist_784', version=1)`, pixels → float32 / 255, cached under `/workspace/data/sklearn-cache/`.
- **Splits:** standard 60k MNIST train / 10k MNIST test, then carve 10% of the 60k train as **validation** (stratified, `random_state=seed`). Sizes per run: 54k train, 6k val, 10k test.
- **Architecture:** `hidden_layer_sizes` must be a 1-tuple `(N,)`. No deeper MLPs. No different activation than ReLU.
- **Optimizer family:** `solver='adam'`. No SGD, no L-BFGS swap.
- **Evaluator:** `evaluators/l1_mnist_mlp.py` — primary metric is `val_accuracy`.

## Editable surface

Only the `MLPClassifier(...)` kwargs inside `fit_and_predict`. Tunable hyperparameters (this is the L1 search space):

| Knob                  | sklearn arg               | Effect                                   |
|-----------------------|---------------------------|------------------------------------------|
| hidden width          | `hidden_layer_sizes=(N,)` | model capacity (single layer)            |
| learning rate         | `learning_rate_init`      | Adam step size                           |
| L2 weight decay       | `alpha`                   | regularization                           |
| batch size            | `batch_size`              | stochastic gradient noise / step count   |
| training duration     | `max_iter`                | passes over training data                |
| Adam β₁, β₂           | `beta_1`, `beta_2`        | momentum / second-moment EMA decay       |
| early stopping        | `early_stopping`          | bool; if True, holds out 10% of *train*  |
| early-stop patience   | `n_iter_no_change`        | epochs of no val improvement before stop |
| convergence tolerance | `tol`                     | minimum loss improvement                 |

**Note on dropout:** sklearn's `MLPClassifier` does not support dropout. If a hypothesis requires it, switch frameworks deliberately (and write that switch up as its own experiment).

**Note on `random_state`:** the harness passes `args.seed` (default 0) into the model. Keep it fixed during the loop so metric deltas attribute to hparam changes, not to seed noise. To check stability of a kept configuration, run an off-loop seed sweep at wrap time.

## Primary metric

- **`val_accuracy`** on the 6k held-out validation slice.
- **Direction:** higher is better.
- `test_accuracy` is recorded too but **never used to make keep/discard decisions** — optimizing on test would proxy-game the wrap report. Test gets quoted once, at wrap, for the kept end-state model.

## Compute budget

Per run: **wall-clock < 90 s on CPU**. A 64-unit hidden layer with `max_iter=10` and `batch_size=128` fits comfortably; pushing hidden width past ~512 or `max_iter` past ~50 will blow the budget. If a hypothesis requires more compute than the budget allows, justify it before running — or split it into a cheaper proxy experiment.

## Stop criterion (within-level)

Stop and declare diminishing returns when **any** of:

- `val_accuracy ≥ 0.985` on the keep branch (~ceiling for a single-hidden-layer MLP on raw MNIST pixels).
- Three consecutive experiments fail to improve `val_accuracy` by ≥ 0.002.
- Total experiments on the branch ≥ 12.

Whichever triggers first wins. **Knowing when to stop is part of what's being tested at L1.** Do not chase the fourth decimal place.

## Researcher discipline reminders (echoes CLAUDE.md)

- **One variable at a time.** Change a single MLP kwarg per commit; if you must change two coupled ones (e.g., batch size and learning rate), justify it in the commit message.
- **State a hypothesis before each run.** Predict the direction and rough magnitude of the val_accuracy change, then check.
- **Attribute outcomes.** After each run, say *which* knob change moved the metric and how confident you are it wasn't seed-noise.
- **Flag proxy-gaming.** If val_accuracy keeps creeping up but test_accuracy stops moving (or worsens), that's val-set overfitting — call it out and propose a check (e.g., reseed the val split).

## Loop mechanics

1. Hypothesis: what kwarg, what value, expected delta, why.
2. Edit `fit_and_predict` — one MLPClassifier kwarg.
3. `git commit` on `autoresearch/L1-<tag>`.
4. `python run.py > run.log 2>&1`.
5. `grep RESULT_JSON run.log` for the metric; on empty output, `tail -n 50 run.log`.
6. Append a row to `levels/L1_mnist_mlp/results.tsv`:
   `commit<TAB>val_accuracy<TAB>status<TAB>description`
   `status ∈ {keep, discard, crash}`.
7. Keep → branch advances. Discard → `git reset --hard HEAD~1`.

## Wrap deliverables (per CLAUDE.md "Wrap convention")

At level wrap, commit under `levels/L1_mnist_mlp/`:

- `RESULTS.md` — table of kept commits, hypothesis log, stop reason, **both val and test accuracy for the kept end-state**.
- `figures/loop_progress.png` — val_accuracy per commit iteration, annotated with the hparam change at each step.
- `results.tsv` — full experiment log including discards.
- `summary/iter<N>_<tag>_<sha>.json` — kept-commit metrics snapshots.

Then merge `autoresearch/L1-<tag>` into `main` with `--no-ff`.
