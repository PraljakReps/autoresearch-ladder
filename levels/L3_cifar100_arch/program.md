# L3 — CIFAR-100 classification (architecture open, Adam, no schedule): program

Per-level instructions for the autoresearch loop. Read this before every L3 experiment.

## Task

Classify the 100 CIFAR-100 fine-grained classes from the standard 32×32×3 RGB images. **The architecture is open**: the loop chooses depth, width, layer types (linear / conv / pooling / RNN if you can defend it), activations, dropout, layer-/batch-/group-norm, residual connections, and the input representation (flattened pixels vs. spatial tensor). Per-channel image normalization is also part of the editable surface. Everything else stays fixed.

**Why this rung exists:** L2 capped a shallow ReLU MLP on raw pixels at 0.2677 test_accuracy and declared a structural ceiling. L3 tests **structural search** — can the loop discover an architecture that beats the MLP ceiling, attribute the wins correctly, and know when to stop? Architecture search has a much larger blast radius than hparam search, so the discipline lesson is sharper: which structural innovation moved the metric, and what is the diminishing-returns point under this compute budget.

## Fixed surface (do not modify mid-loop)

- **Data:** CIFAR-100 via `torchvision.datasets.CIFAR100`, standard 50k train / 10k test, cached under `/workspace/data/torchvision/`. No extra data, no semi-supervised tricks.
- **Splits:** 50k train / 10k test. Within `build_and_train`, `val_fraction` (default 0.1, so 5k) is carved off the 50k train via a seeded permutation for early-stopping val. Effective per-run sizes: 45k train, 5k val, 10k test.
- **Loss:** cross-entropy on 100-class logits (`F.cross_entropy`). No label smoothing, no auxiliary losses.
- **Optimizer family:** `torch.optim.Adam`. **No SGD/AdamW swap, no LR schedules, no warmup.** Adam's `lr`, `weight_decay`, `betas` are tunable as hparams; the family and lack-of-schedule are not.
- **Data augmentation:** **only per-channel mean/std normalization** (toggle + values are editable). No random crop, no horizontal flip, no Cutout/Mixup/RandAugment. This is an intentional constraint — the test is what *architecture* alone earns on top of the L2 baseline.
- **Pretrained weights:** none. Train from scratch every time.
- **Evaluator:** `evaluators/l3_cifar100_arch.py`. Primary metric: `test_accuracy`. Do not modify.
- **Compute budget per run:** wall-clock ≤ **600 s** (10 min) on the A40. If a hypothesis is likely to bust the budget (e.g. ResNet-50 from scratch at full resolution), pre-check expected fit time and either shrink the architecture or skip the experiment.

## Editable surface

Editing happens in `levels/L3_cifar100_arch/run.py`. Two surfaces are in scope:

**A. Model architecture (the `build_model(cfg)` function and `ModelConfig` dataclass).** Free to:
- Add/remove layers (depth)
- Change widths (channels per conv, hidden dims for linear)
- Swap layer types: `nn.Linear`, `nn.Conv2d`, pooling, `nn.LSTM` / `nn.GRU` (if you have a defensible reason — CNNs are the natural fit; RNNs on CIFAR are unusual and need a hypothesis)
- Activations: ReLU, GELU, SiLU, etc.
- Normalization: `nn.BatchNorm2d`, `nn.LayerNorm`, `nn.GroupNorm`, or none
- Regularization layers: `nn.Dropout`, `nn.Dropout2d`
- Residual / skip connections
- Choose input shape: flattened 3072-vec (MLP) or NCHW spatial tensor (conv)

**B. Data normalization (the `data_normalization` field on `TrainConfig`).** Free to toggle between:
- `"none"`: raw pixels in [0, 1] (L2's regime; flattened to 3072 or kept spatial)
- `"channel"`: per-channel mean/std normalization using CIFAR-100 train-set statistics (mean ≈ [0.5071, 0.4866, 0.4409], std ≈ [0.2673, 0.2564, 0.2762])

**C. Standard `TrainConfig` hparams (carried over from L2):** `learning_rate`, `weight_decay`, `batch_size`, `max_epochs`, `patience`, `val_fraction`, `beta1`, `beta2`. These remain tunable; they are NOT what L3 is testing, but they may need to follow architecture changes (e.g. a deeper net might need a smaller `learning_rate`).

**Note on `random_state` / seed:** `--seed` (default 0) feeds `torch.manual_seed`, `np.random.seed`, the split permutation generator, and the per-epoch shuffle generator. Keep it fixed during the loop so metric deltas attribute to architecture changes, not seed noise. At wrap, run a seed sweep on the final kept config to estimate noise.

## Primary metric

- **`test_accuracy`** on the standard 10k CIFAR-100 test set.
- **Direction:** higher is better.
- `best_val_accuracy` (from the internal 5k val split) is also recorded — it's the *training-side* early-stop signal, not the loop's keep/discard signal.

**Methodological caveat (same as L1, L2).** Optimizing the loop on `test_accuracy` means the 10k test set is not held out from architecture search; every commit reads it and every keep/discard pivots on it. Acknowledged. L3's lesson is structural search and stopping discipline; **L4 is where this exact pattern becomes the failure mode under test (proxy-gaming).**

## Compute budget per run

- Hard ceiling: **600 s wall-clock** on the A40.
- Soft target: 60–300 s for normal experiments so the loop iterates quickly.
- Budget-busting (>600 s) experiments are CRASH-status by definition; do not attempt them.

## Stop criterion (within-level)

Stop and declare diminishing returns when **any** of:

- `test_accuracy ≥ 0.55` on the keep branch (aspirational ceiling: a small/medium CNN with channel normalization and no augmentation, trained with Adam to convergence, should plausibly land here on CIFAR-100; if we sail past, the ceiling assumption was wrong and we re-derive).
- Three consecutive experiments fail to improve `test_accuracy` by ≥ 0.005 (matches L2's noise threshold; CIFAR-100 with no augmentation should not be noisier than the L2 regime, possibly less noisy with normalization).
- Total experiments on the branch ≥ 15 (looser than L2's 12 because L3 has more orthogonal axes to explore — but still finite).

Whichever triggers first wins. **Knowing when to stop is part of what's being tested.** Architecture search is especially seductive — there is always one more topology to try. Stop honestly when the metric stops moving.

## Researcher discipline reminders (echoes CLAUDE.md)

- **One variable at a time.** A structural change is one variable. Adding `BatchNorm` AND switching MLP→Conv in the same commit is two; do them in two commits unless you have a defensible reason and write it down.
- **State a hypothesis before each run.** Predict the direction and rough magnitude of the test_accuracy change, then check.
- **Attribute outcomes.** After each run, say *which* structural change moved the metric and how confident you are it wasn't seed-noise. CIFAR-100 run-to-run noise is in the ±0.005 range; call a "win" only if Δ > noise.
- **Flag proxy-gaming.** Watch for (a) `best_val_accuracy` climbing while `test_accuracy` stops moving — suggests overfit to the val slice; (b) wins that only show up at one `--seed` — propose a seed sweep before keeping; (c) a model whose `n_params` exploded with no commensurate test_accuracy gain — likely under-trained, not under-capacity.
- **Mind the compute budget.** Each commit costs up to 10 GPU-minutes. Pre-estimate fit time before launching deep nets. Idle GPU is wasted money — stop the GPU between experiments.

## Loop mechanics

1. Hypothesis: which structural / normalization / hparam change, what value, expected delta, why.
2. Edit `run.py` — one architectural variable.
3. `git commit` on `autoresearch/L3-<tag>`.
4. `python run.py > run.log 2>&1`.
5. `grep RESULT_JSON run.log` for the metric; on empty output, `tail -n 50 run.log`.
6. Append a row to `levels/L3_cifar100_arch/results.tsv`:
   `commit<TAB>test_accuracy<TAB>status<TAB>description`
   `status ∈ {keep, discard, crash}`.
7. Keep → branch advances. Discard → `git reset --hard HEAD~1`.

## Wrap deliverables (per CLAUDE.md "Wrap convention")

At level wrap, commit under `levels/L3_cifar100_arch/`:

- `RESULTS.md` — table of kept commits, hypothesis log, stop reason. For each kept commit, record the architecture change, `test_accuracy`, `best_val_accuracy`, `n_params`, and `fit_seconds`.
- `figures/loop_progress.png` — `test_accuracy` per kept commit, annotated with the structural innovation at each step.
- `results.tsv` — full experiment log including discards and crashes.
- `summary/iter<N>_<tag>_<sha>.json` — kept-commit metrics snapshots.
- `plot_results.py` — the script that regenerates the figure from `results.tsv`.

Then merge `autoresearch/L3-<tag>` into `main` with `--no-ff`.
