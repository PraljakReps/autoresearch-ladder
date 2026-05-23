# agent-docs

Extended documentation for agents working in this repo. The top-level `CLAUDE.md` and `README.md` are the entry points; deeper, task-specific notes live here and are pulled in on demand.

## Conventions

- **Session handoffs** live under `agent-docs/handoffs/YYYY-MM-DD.md` and are **gitignored** (per-pod, ephemeral session state — not part of the repo). The `session-handoff.md` symlink at the root of this folder points at the most recent one and is likewise gitignored.
- When wrapping a session, write a new dated handoff under `handoffs/` and repoint `session-handoff.md` at it. This is the audit trail of the loop on a given pod.
- Other documentation files in this folder (topic-named, e.g. `harness-protocol.md`, `evaluator-conventions.md`) **are** tracked and should be committed.

Rule of thumb: if a doc captures **state of an in-flight session**, gitignore it. If it captures **durable knowledge about the repo**, commit it.
