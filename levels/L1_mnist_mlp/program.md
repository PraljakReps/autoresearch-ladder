# L1 — MNIST classification (1-hidden ReLU MLP, Adam): program

Per-level instructions for the autoresearch loop. Read this before every L1 experiment.

## Task

Classify the 10 MNIST digit classes from raw 784-dim pixel vectors (scaled to [0, 1]) using a **single-hidden-layer ReLU MLP trained with Adam**. The architecture, activation, optimizer family, data, and splits are part of the fixed harness. The loop tunes hyperparameters only.

## Fixed surface (do not modify mid-loop)

- **Data:** MNIST via `fetch_openml('mnist_784', version=1)`, pixels → float32 / 255, cached under `/workspace/data/sklearn-cache/`.
- **Splits:** standard 60k MNIST train / 10k MNIST test. The full 60k goes into `MLPClassifier.fit(...)`. With `early_stopping=True`, sklearn internally carves `validation_fraction` (default 0.1, i.e. 6k) of those 60k as a held-out val to drive early stopping; the remaining 54k is what the optimizer actually trains on. The val is **inside** the fit, not a separate harness-level split.
- **Architecture:** `hidden_layer_sizes` must be a 1-tuple `(N,)`. No deeper MLPs. No different activation than ReLU.
- **Optimizer family:** `solver='adam'`. No SGD, no L-BFGS swap.
- **Evaluator:** `evaluators/l1_mnist_mlp.py` — primary metric is `test_accuracy`.

## Editable surface

Only the `MLPClassifier(...)` kwargs inside `fit_and_predict`. Tunable hyperparameters (this is the L1 search space):

| Knob                  | sklearn arg               | Effect                                          |
|-----------------------|---------------------------|-------------------------------------------------|
| hidden width          | `hidden_layer_sizes=(N,)` | model capacity (single layer)                   |
| learning rate         | `learning_rate_init`      | Adam step size                                  |
| L2 weight decay       | `alpha`                   | regularization                                  |
| batch size            | `batch_size`              | stochastic gradient noise / step count          |
| max epochs            | `max_iter`                | upper bound; early stop usually cuts it short   |
| Adam β₁, β₂           | `beta_1`, `beta_2`        | momentum / second-moment EMA decay              |
| early stopping        | `early_stopping`          | bool; if True, internal val drives stop         |
| internal val size     | `validation_fraction`     | fraction of fit-input used for early-stop val   |
| early-stop patience   | `n_iter_no_change`        | epochs of no val improvement before stop        |
| convergence tolerance | `tol`                     | minimum loss improvement                        |

**Note on dropout:** sklearn's `MLPClassifier` does not support dropout. If a hypothesis requires it, switch frameworks deliberately (and write that switch up as its own experiment).

**Note on `random_state`:** the harness passes `args.seed` (default 0) into the model. Keep it fixed during the loop so metric deltas attribute to hparam changes, not to seed noise. To check stability of a kept configuration, run an off-loop seed sweep at wrap time.

## Primary metric

- **`test_accuracy`** on the standard 10k MNIST test set.
- **Direction:** higher is better.
- `best_val_accuracy` (from sklearn's internal early-stopping val) is also recorded — it's a *training-side* signal (used by the fit to decide when to stop), not the loop's keep/discard signal.

**Methodological caveat — read this once and remember it.** Optimizing the autoresearch loop directly on `test_accuracy` means the 10k MNIST test set is *not* held out from hyperparameter search; every commit reads it and every keep/discard pivots on it. This is acknowledged-and-accepted at L1 (whose lesson is loop discipline — one variable at a time, attribute cause). It would be unsafe at L4 (proxy-gaming resistance), where exactly this pattern is the failure mode being tested. The val-driven early stopping inside each fit is unrelated to this — that's an internal regularizer, not the loop's signal.

## Compute budget

Per run: **wall-clock < 90 s on CPU**. A 64-unit hidden layer with `max_iter=10` and `batch_size=128` fits comfortably; pushing hidden width past ~512 or `max_iter` past ~50 will blow the budget. If a hypothesis requires more compute than the budget allows, justify it before running — or split it into a cheaper proxy experiment.

## Stop criterion (within-level)

Stop and declare diminishing returns when **any** of:

- `test_accuracy ≥ 0.985` on the keep branch (~ceiling for a single-hidden-layer MLP on raw MNIST pixels).
- Three consecutive experiments fail to improve `test_accuracy` by ≥ 0.002.
- Total experiments on the branch ≥ 12.

Whichever triggers first wins. **Knowing when to stop is part of what's being tested at L1.** Do not chase the fourth decimal place.

## Researcher discipline reminders (echoes CLAUDE.md)

- **One variable at a time.** Change a single MLP kwarg per commit; if you must change two coupled ones (e.g., batch size and learning rate), justify it in the commit message.
- **State a hypothesis before each run.** Predict the direction and rough magnitude of the val_accuracy change, then check.
- **Attribute outcomes.** After each run, say *which* knob change moved the metric and how confident you are it wasn't seed-noise.
- **Flag proxy-gaming.** Test accuracy is the loop signal here, so the usual val→test divergence check doesn't apply. Instead, watch for: (a) `best_val_accuracy` (training-side) climbing while `test_accuracy` stops moving — suggests the early-stop val and test have drifted; (b) wins that only show up at one `--seed` value — propose a seed sweep before keeping.

## Loop mechanics

1. Hypothesis: what kwarg, what value, expected delta, why.
2. Edit `fit_and_predict` — one MLPClassifier kwarg.
3. `git commit` on `autoresearch/L1-<tag>`.
4. `python run.py > run.log 2>&1`.
5. `grep RESULT_JSON run.log` for the metric; on empty output, `tail -n 50 run.log`.
6. Append a row to `levels/L1_mnist_mlp/results.tsv`:
   `commit<TAB>test_accuracy<TAB>status<TAB>description`
   `status ∈ {keep, discard, crash}`.
7. Keep → branch advances. Discard → `git reset --hard HEAD~1`.

## Wrap deliverables (per CLAUDE.md "Wrap convention")

At level wrap, commit under `levels/L1_mnist_mlp/`:

- `RESULTS.md` — table of kept commits, hypothesis log, stop reason. Quote both `test_accuracy` (the loop metric) and `best_val_accuracy` (the early-stop signal) for the kept end-state.
- `figures/loop_progress.png` — `test_accuracy` per commit iteration, annotated with the hparam change at each step.
- `results.tsv` — full experiment log including discards.
- `summary/iter<N>_<tag>_<sha>.json` — kept-commit metrics snapshots.

Then merge `autoresearch/L1-<tag>` into `main` with `--no-ff`.
