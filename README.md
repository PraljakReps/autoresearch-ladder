# autoresearch-ladder

A harness for piloting **autonomous research with Claude** across a ladder of increasing difficulty.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which has one agent edit `train.py` to lower `val_bpb` on a small GPT, this repo generalizes that loop into a **ladder** of tasks. The goal is not just "get a better model on one task" — it is to **measure how well an autonomous Claude-driven research loop behaves** as the search problem gets harder, the targets less known, and the evaluators noisier.

You (Claude Code) are both the operator of the harness and, in many runs, the researcher being tested. The "LLM" in this project is **Claude itself** — the agent doing the experimenting. The "models" being built are whatever each level requires (linear regression → MNIST classifier → discovered algorithms).

## The ladder

Two axes climb together: how **open-ended** the search is (known target → genuine discovery), and how **trustworthy** the evaluator is (instant unambiguous verifier → expensive noisy proxy). Each level isolates a new failure mode.

| Level | Task | New capability under test |
|-------|------|---------------------------|
| L0 | Linear regression fit | Plumbing: launch job, parse result, report |
| L1 | Hyperparameter search, fixed model | Loop discipline: one variable at a time |
| L2 | MNIST — find an architecture | Structural search; knowing when to stop |
| L3 | Unknown-but-checkable target | Discovery vs. optimizing toward a known number |
| L4 | Gameable metric vs. true objective | Proxy-gaming resistance |
| L5 | Algorithm discovery w/ automatic verifier | Open-ended code search strategy |

## Per-level shape (karpathy pattern, generalized)

Each level mirrors the three-file split that makes karpathy's loop tight:

```
levels/LN_<name>/
  run.py        ← Claude edits this. Architecture, optimizer, hyperparams — fair game.
  program.md    ← per-level agent instructions (the "skill")
evaluators/
  lN_<name>.py  ← fixed scorer. Not modified. Returns the metric.
```

Per-level invariants:
- **Fixed compute/time budget per run** so experiments are directly comparable. The unit varies by level — minutes of training, k folds, fixed-step optimizer budget, etc.
- **One primary metric** returned by the evaluator. Direction (lower-better / higher-better) is declared in the level's `program.md`.
- **Branch per run tag**: experiments at level N run on `autoresearch/LN-<tag>` (e.g. `autoresearch/L2-mar5`). Each idea is a commit; keep if metric improves, `git reset` if not.
- **`results.tsv` per level** (gitignored, untracked): `commit \t metric \t status \t description` with status in `{keep, discard, crash}`.

## The experiment loop

After setup, for each experiment at a level:

1. State a hypothesis: what change, what you expect, and why.
2. Edit `run.py` — one variable at a time unless explicitly justified.
3. `git commit` the change.
4. Run: `python run.py > run.log 2>&1` (redirect; do not flood context with stdout).
5. Score: invoke the level's evaluator on the run's metrics file, or `grep` the summary line.
6. Append to `results.tsv` with status `keep` / `discard` / `crash`.
7. If improved → branch stays advanced. If not → `git reset --hard HEAD~1`.

Loop until the level's stopping criterion fires — per-level: "diminishing returns declared", "verifier says done", or a budget cap. Knowing when to stop is part of what's being measured (see `CLAUDE.md`).

## Quick start

L0 (smoke test the harness end-to-end):

```bash
source /workspace/envs/autoresearch-uv/bin/activate
cd levels/L0_linear_regression
python run.py
python -m evaluators.l0_linear_regression results/<timestamp>/metrics.json
```

Persistent state lives on `/workspace` only — the container root is throwaway. See `CLAUDE.md` for the full environment rules.

## What's the same, what's different vs. karpathy

|  | karpathy/autoresearch | this repo |
|---|---|---|
| Scope | Single task (LLM training, `val_bpb`) | Ladder of L0..L5 |
| Compute budget | Fixed 5 min/run | Per-level fixed budget |
| Editable file | `train.py` | `levels/LN_*/run.py` |
| Human-edited prompts | `program.md` | `CLAUDE.md` + per-level `program.md` |
| Branch convention | `autoresearch/<tag>` | `autoresearch/LN-<tag>` |
| Metric | `val_bpb` (lower-better) | per-level, declared in `program.md` |
| Agent | Any (Claude/Codex/etc.) | Claude specifically |
| Stopping rule | Never stop | Per-level — stop on diminishing returns or verifier-done |

## Sessions and handoffs

When starting a session, read `agent-docs/session-handoff.md` first. When wrapping up, write a new dated handoff at `agent-docs/handoffs/YYYY-MM-DD.md` and repoint the symlink. This is the audit trail of the loop.

## License

See `LICENSE`.
