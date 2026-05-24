# L2 — CIFAR-100 classification (1-hidden ReLU MLP, Adam): program

Per-level instructions for the autoresearch loop. Read this before every L2 experiment.

## Task

Classify the 100 CIFAR-100 fine-grained classes from raw 3072-dim flattened RGB pixel vectors (32×32×3, scaled to [0, 1]) using a **single-hidden-layer ReLU MLP trained with Adam on GPU**. The architecture, activation, optimizer family, data, and splits are part of the fixed harness. The loop tunes hyperparameters only.

**Why this rung exists:** L1 was MNIST — the lesson there is loop discipline on an easy dataset (almost any reasonable hparam config lands at ~97-98%). On CIFAR-100 a shallow MLP on raw pixels caps somewhere around **20-30% test accuracy**, and hparam choices actually move the needle by several points. This is where one-variable-at-a-time discipline pays off (or fails visibly).

## Fixed surface (do not modify mid-loop)

- **Data:** CIFAR-100 via `torchvision.datasets.CIFAR100`, pixels → float32 in [0, 1], flattened to 3072-dim, cached under `/workspace/data/torchvision/`. The torchvision-default RGB normalization is **not** applied — keeping it consistent with L1's raw-pixel inputs.
- **Splits:** standard 50k CIFAR-100 train / 10k CIFAR-100 test. Within `build_and_train`, `val_fraction` (default 0.1, so 5k) is carved off the 50k train via `torch.utils.data.random_split` for early-stopping val. Effective per-run sizes: 45k train, 5k val, 10k test.
- **Architecture:** the `MLP` class in `run.py` (input 3072 → linear → ReLU → linear → 100 logits, softmax via cross-entropy loss). No deeper MLPs, no different activation, no dropout/batchnorm (that's an L3 architecture move).
- **Optimizer family:** `torch.optim.Adam`. No SGD, no AdamW swap, no schedulers.
- **Evaluator:** `evaluators/l2_cifar100_mlp.py` — primary metric is `test_accuracy`.

## Editable surface

Only the `TrainConfig` dataclass defaults at the top of `run.py`. Tunable hyperparameters:

| Knob              | TrainConfig field | Effect                                          |
|-------------------|-------------------|-------------------------------------------------|
| hidden width      | `hidden_width`    | model capacity (single layer)                   |
| learning rate     | `learning_rate`   | Adam step size                                  |
| L2 weight decay   | `weight_decay`    | regularization (Adam's `weight_decay` arg)      |
| batch size        | `batch_size`      | stochastic gradient noise / step count          |
| max epochs        | `max_epochs`      | upper bound; early stop usually cuts it short   |
| early-stop patience | `patience`      | epochs of no val improvement before stop        |
| val fraction      | `val_fraction`    | size of internal val split (default 0.1 = 5k)   |
| Adam β₁, β₂       | `beta1`, `beta2`  | momentum / second-moment EMA decay              |

**Note on `random_state` / seed:** the harness passes `--seed` (default 0) to `torch.manual_seed`, `np.random.seed`, and the `random_split` generator. Keep it fixed during the loop so metric deltas attribute to hparam changes, not to seed noise. To check stability of a kept configuration, run an off-loop seed sweep at wrap time.

## Primary metric

- **`test_accuracy`** on the standard 10k CIFAR-100 test set.
- **Direction:** higher is better.
- `best_val_accuracy` (from the internal early-stopping holdout) is also recorded — it's a *training-side* signal, not the loop's keep/discard signal.

**Methodological caveat — same as L1.** Optimizing the loop directly on `test_accuracy` means the 10k CIFAR-100 test set is *not* held out from hyperparameter search; every commit reads it and every keep/discard pivots on it. Acknowledged at L2 (lesson is loop discipline). It would be unsafe at L4 (proxy-gaming resistance), where exactly this pattern is the failure mode being tested.

## Compute budget

Per run: **wall-clock < 120 s on the A40 GPU** (or whichever GPU is present; CPU fallback works but blows the budget by a wide margin and should be flagged as a budget-busting experiment).

A reference fit at the default `TrainConfig` should land in the 30–90s range on an A40. If a hypothesis pushes hidden width past ~1024 or batch size below ~32, sanity-check the expected fit time before committing.

## Stop criterion (within-level)

Stop and declare diminishing returns when **any** of:

- `test_accuracy ≥ 0.30` on the keep branch (rough empirical ceiling for shallow MLP on CIFAR-100 raw pixels; if we sail past this, the ceiling assumption was wrong and we re-derive it).
- Three consecutive experiments fail to improve `test_accuracy` by ≥ 0.005 (note: looser than L1's 0.002 — CIFAR-100 has higher variance per run).
- Total experiments on the branch ≥ 12.

Whichever triggers first wins. **Knowing when to stop is part of what's being tested.** Do not chase the fourth decimal place on a metric whose run-to-run noise is in the third decimal.

## Researcher discipline reminders (echoes CLAUDE.md)

- **One variable at a time.** Change a single `TrainConfig` field per commit; if you must change two coupled ones (e.g., batch size and learning rate together), justify it in the commit message.
- **State a hypothesis before each run.** Predict the direction and rough magnitude of the test_accuracy change, then check.
- **Attribute outcomes.** After each run, say *which* knob change moved the metric and how confident you are it wasn't seed-noise. CIFAR-100 run-to-run noise is in the ±0.005 range — call a "win" only if Δ > noise.
- **Flag proxy-gaming.** Watch for (a) `best_val_accuracy` (training-side) climbing while `test_accuracy` stops moving — suggests the early-stop val and test have drifted; (b) wins that only show up at one `--seed` — propose a seed sweep before keeping.
- **Stop the GPU when not actively running jobs** (CLAUDE.md environment rule). After each experiment, no leftover Python processes should be holding the device.

## Loop mechanics

1. Hypothesis: what `TrainConfig` field, what value, expected delta, why.
2. Edit `TrainConfig` defaults in `run.py` — one field.
3. `git commit` on `autoresearch/L2-<tag>`.
4. `python run.py > run.log 2>&1`.
5. `grep RESULT_JSON run.log` for the metric; on empty output, `tail -n 50 run.log`.
6. Append a row to `levels/L2_cifar100_mlp/results.tsv`:
   `commit<TAB>test_accuracy<TAB>status<TAB>description`
   `status ∈ {keep, discard, crash}`.
7. Keep → branch advances. Discard → `git reset --hard HEAD~1`.

## Wrap deliverables (per CLAUDE.md "Wrap convention")

At level wrap, commit under `levels/L2_cifar100_mlp/`:

- `RESULTS.md` — table of kept commits, hypothesis log, stop reason. Quote both `test_accuracy` (loop metric) and `best_val_accuracy` (early-stop signal) for the kept end-state.
- `figures/loop_progress.png` — `test_accuracy` per commit iteration, annotated with the hparam change at each step.
- `results.tsv` — full experiment log including discards.
- `summary/iter<N>_<tag>_<sha>.json` — kept-commit metrics snapshots.

Then merge `autoresearch/L2-<tag>` into `main` with `--no-ff`.
