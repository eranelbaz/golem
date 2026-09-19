# Implementation Plans

One reviewed plan per task board entry (see [`../../worklog.md`](../../worklog.md),
protocol steps 5–7). Naming: `YYYY-MM-DD-<task-id>-<slug>.md`, written in the
[superpowers:writing-plans](https://github.com/eranelbaz/superpowers/tree/main/skills/writing-plans)
format — header with Goal/Architecture/Tech Stack/Spec, global constraints,
bite-sized TDD tasks with exact files, commands, and commit points.

Rules:

- A task's plan is written and **plan-reviewed** (one zai reviewer,
  `plan-document-reviewer-prompt.md`, fix-and-redo ≤ 3 rounds) before any
  implementation commit for that task.
- The plan travels in the same PR as its implementation, with checkboxes
  ticked as steps complete.
- Merged plans are historical record — corrections go in follow-up PRs,
  never silent edits.

(No plans yet — the first lands with task M0.1.)
