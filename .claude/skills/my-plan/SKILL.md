---
name: my-plan
description: Draft a design/implementation plan for a change, pressure-test it with /dual-review (design mode), iterate with the user, then commit it under plans/. Use when the user wants to plan something before building it - "plan X", "let's plan the Y refactor", "/my-plan", "work up a plan for Z and review it". This is the planning dance for this repo - it produces a tracked plan doc, not code.
user_invocable: true
---

# My Plan

Codifies this repo's planning workflow: **ground -> draft -> dual-review -> iterate -> commit**. The output is a reviewed plan document under `plans/`. It does **not** write production code - that's a later, separate build pass against the committed plan.

This is tuned for `~/dev/vibe/explainshell`. Honor the repo conventions throughout: conventional-commit messages (`feat(web):`, `chore(deps):`, ...), commits go to `master`, and `CLAUDE.md`/`AGENTS.md` are the source of truth for structure, workflow, and settled conventions.

## Process

1. **Scope it.** Confirm what the plan is for. If the ask is vague, ask one or two sharp questions before drafting - a plan for the wrong thing wastes the review.

2. **Ground the plan in real code first.** This is the step that makes the plan worth reviewing. Read the actual files, symbols, routes, and schemas the change touches - do not hand-wave. Reference real `file:line`, function names, CLI subcommands, and DB tables. Check `CLAUDE.md`/`AGENTS.md` before reopening a settled convention or workflow call. A grounded plan names the crux (the hard tradeoff), the invariants it must preserve, and the open questions; a vague one does not.

3. **Write the plan to `plans/<slug>.md`.** Short kebab-case slug (e.g. `batch-extraction.md`). Structure that has worked:
   - A status line (`design draft - not yet implemented`) and the origin of the ask.
   - **Problem** - what's wrong today, grounded in real symbols.
   - **The crux** - the central tradeoff and the proposed resolution, with the alternative named.
   - **Changes** - extraction pipeline / matching / storage / web / tooling, concrete (function signatures, CLI flags, schema changes, routes).
   - **Invariants to preserve** - the conventions and design rules the change must not break.
   - **Tests and validation** - which suite applies (`make tests-quick` vs `make tests-all`), and whether the change needs an eval pass (`/eval-llm` for extractor changes, `/eval-render` for mandoc rendering changes) before it can land.
   - **Migration/data** - DB rebuild or `db-latest` release implications, if any.
   - **Open questions** - the calls you want the user (and the reviewers) to weigh.
   `plans/` is tracked in this repo (not gitignored) - the committed plan is the durable record.

4. **Dual-review it.** Invoke the `dual-review` skill in **design** mode pointed at the plan: `/dual-review design plans/<slug>.md`. Let it run both reviewers (codex + a Claude subagent) and produce the synthesis. Do not skip this - the cross-check is the point of the dance.

5. **Synthesize and iterate with the user.** Present the synthesis: where the reviewers agree (highest confidence), what each caught that the other missed, and - critically - **push back where a reviewer contradicts the repo's own docs** (`CLAUDE.md`/`AGENTS.md`, an invariant, an established workflow), with the citation. Then propose a revision set (apply / defer / skip-with-reason). **Get the user's sign-off on the direction before editing the plan** - this is a human-in-the-loop step, not an auto-apply.

6. **Fold accepted revisions into the plan body.** Update `plans/<slug>.md`: rewrite the affected sections, resolve the open questions with the review outcome, and record any findings deliberately **deferred** (e.g. spun out as follow-up work) so they aren't silently dropped. Note the review in the status line (`dual-reviewed (codex + claude <date>), revisions folded in`).

7. **Commit on sign-off.** When the user is satisfied, commit the plan with a conventional-commit message (e.g. `docs(plans): add <slug> design plan`). Commit to `master` per repo convention, but **don't push** unless asked - pushes to `master` trigger a production deploy via CI. Stage only the plan file - never bundle unrelated working-tree changes, and never commit production code from this skill. End your turn by reporting the commit hash.

## Constraints

- **Plan, don't build.** This skill produces a plan document. It does not edit production code, run builds, or apply the change. The build is a separate pass against the committed plan. The `make format`/test workflow in `CLAUDE.md` doesn't apply here - the only output is markdown.
- **Don't auto-apply review findings.** Synthesize, recommend, get sign-off, then fold in. The user decides what's adopted, deferred, or skipped.
- **Don't commit without the user's go-ahead** on the plan's content. "Commit it" / clear satisfaction is the signal.
- **Respect repo conventions:** check `CLAUDE.md`/`AGENTS.md` before reopening settled questions; plans that touch the LLM extractor or mandoc rendering must name the eval that will validate them.
- If the user explicitly wants only some steps (e.g. "just draft it, skip the review"), follow that - the dance is the default, not a straitjacket.
