# Positional prefix matching (sigil-prefixed operands)

Status: design draft - dual-reviewed (codex + claude 2026-06-12), revisions folded in - not yet implemented
Origin: [GitHub issue #361](https://github.com/idank/explainshell/issues/361) — `dig ns foo.bar @8.8.8.8` explains `@8.8.8.8` with the wrong positional's text.

## Problem

The matcher assigns positional operands purely by order. When a word is not a
flag, `Matcher.visitword` (matcher.py:670-706) walks
`ParsedManpage.positionals` — an `OrderedDict` of positional name → merged
help text (models.py:107-118) — using a single counter
(`matched_group.positional_index`, matcher.py:28): first unmatched word gets
the first positional, second word the second, and so on. The only exception is
an exact-name match (`word in d`, matcher.py:686) for keyword-style
positionals like `start`/`stop`.

For dig, the extracted positional order is `server, name, type`, so
`dig ns foo.bar @8.8.8.8` produces:

- `ns` → **server** (693 chars, the longest text — worst possible landing spot)
- `foo.bar` → **name**
- `@8.8.8.8` → **type**

The correct mapping for the `@`-token is unambiguous: dig's synopsis is
`dig [@server] [name] [type] ...` — the `@` sigil is attached to exactly one
operand. The matcher has no notion of token *shape*, so it can't use that.

## The crux

**Where does the sigil knowledge live?** Three candidates:

1. **LLM-emitted schema field (chosen).** The extraction prompt already asks
   the model to identify positionals; the sigil is visible in the synopsis
   (`[@server]`). Add an optional `prefix` field to the option JSON schema and
   the `Option` model. Generic — any page whose synopsis attaches a literal
   sigil to an operand can carry it; pages without it behave exactly as today.
2. *Matcher-side heuristic (rejected).* Hardcoding "`@` + a positional named
   `server`" in matcher.py fixes dig only, encodes command-specific knowledge
   in the wrong layer, and grows unboundedly.
3. *UI truncation (orthogonal).* Collapsing long help texts is a separate
   product question; it doesn't fix wrong assignment.

**Secondary design call:** a prefix-bearing positional is removed from ordered
consumption entirely — it can *only* be claimed by a token carrying its
prefix. This is what fixes the whole command, not just the `@`-token: with
`server` out of the ordered pool, `ns` and `foo.bar` consume `name, type`
instead of `server, name`. Tradeoff: `dig 8.8.8.8` (no `@`) maps `8.8.8.8` to
`name` — which matches dig's actual semantics (a bare address is a query
name, not a server).

## Changes

### Model (`explainshell/models.py`)

- `Option` gains `prefix: str | None = None` — a literal string a token must
  start with for this positional to claim it. Documented as only meaningful
  when `positional` is set (same rule as `positional` vs flags).
  - Not stuffed into `Option.meta`: `meta` is extraction side-band data
    (`{"lines": [start, end]}`, response.py:212); `prefix` is matching
    behavior and belongs as a first-class field.
  - Serialization is free: `to_store()` uses `model_dump()` (models.py:133)
    and old DB rows without the key deserialize to the `None` default.
- `ParsedManpage.positionals` keeps its current shape (name → text) but
  excludes prefix-bearing options; a new property (e.g.
  `prefixed_positionals`, returning an ordered name → `(prefix, text)`
  mapping) exposes the rest. `positionals` is consumed only by matcher.py
  (two call sites), so the shape change is contained.

### Matcher (`explainshell/matcher.py`)

Two-pool assignment in the positional branch of `visitword`
(matcher.py:670-706):

1. **Widen the branch gate.** The whole branch is currently gated on
   `if self.man_page.positionals:` (matcher.py:670). Since prefixed entries
   leave that dict, the gate must become "any positionals at all"
   (`positionals or prefixed_positionals`) — otherwise a page whose *only*
   positional is prefixed (`cmd [@server]`) never reaches the prefix pool
   and `cmd @8.8.8.8` goes unknown.
2. **Prefix pool first:** if the word starts with any declared prefix, claim
   that positional and return. Multiple prefixed tokens may claim the same
   positional (reuse, mirroring the existing variadic behavior). If several
   positionals declare the same prefix, first in document order wins.
3. **Ordered pool, with one new guard:** exact-name match, then index-ordered
   consumption with variadic reuse of the last key — today's logic
   (matcher.py:686-695) over the prefix-free dict. The single
   `positional_index` counter survives; no consumed-set needed, because the
   prefix pool never interacts with the index. New guard: when the ordered
   pool is *empty* (all positionals prefixed), a non-prefixed token must fall
   through to unknown — the existing variadic fallback (`keys[-1]`,
   matcher.py:695) would raise IndexError on an empty list, on the web
   serving path.

A token bearing a prefix that no positional declares falls through to the
ordered pool as today (no behavior change for pages without prefixes).

**Precedence with sigil-style flags:** option/flag matching runs before the
positional branch, so pages that store `+`-style flags as `long` options
(dig's `+trace` family) are unaffected by `+` being in the sigil allowlist —
verified live: `dig +trace foo.bar` matches `+trace` as a flag today, and
the prefix pool only ever sees tokens no flag claimed.

**Acknowledged behavior delta:** today a literal `dig server` exact-name
matches the `server` positional via `word in d` (matcher.py:686). Under this
plan that entry is excluded from the ordered dict, so the literal word falls
to ordered consumption (`name`). That is arguably *more* correct — dig treats
a bare word as a query name — but it is a change on existing data shape, made
deliberately.

### Extraction (`explainshell/extraction/llm/`)

- `prompt.py` JSON schema: add `"prefix": "@"` with an instruction along the
  lines of: *literal sigil the synopsis attaches to this positional operand
  (e.g. `@` in `dig [@server]`); only valid together with `positional`; omit
  otherwise.*
- **Sigil constraint (load-bearing):** because a prefixed positional leaves
  ordered consumption entirely, a false-positive `prefix` on a popular page
  breaks the *bare* form of its most common operand. Two concrete traps:
  ssh/scp synopses use `[user@]hostname` — an optional sub-component, not an
  operand sigil — and a model emitting `prefix: "@"` there would make plain
  `ssh example.com` stop matching `hostname`; and placeholder-styled names
  like `<FILE>` must not be normalized into a `<` prefix. Therefore `prefix`
  is restricted to a **single punctuation character from a corpus-grounded
  allowlist**, enforced in *both* sanitize sites; anything else is dropped
  with a debug log.
- **Allowlist: `@`, `+`, `:` — grounded in a scan of the corpus**, not
  guessed. Method: SYNOPSIS sections of all 61,322 raw pages in the DB,
  matching *standalone* bracketed sigil operands (`[` preceded by
  whitespace, so suffix syntax glued to another token doesn't count).
  Results:
  - `@` — 197 pages: dig/delv/adig `@server`, GNU `as`/gcc `@FILE`
    argfiles, javadoc/jar `@files`, bdep `@cfg-name`, cargo-install
    `@version`.
  - `+` — 117 pages: `date +FORMAT`, vi-style `+line` in editors (mg,
    mcedit, vile, geany), `pr +page`, `xset +dpms`.
  - `:` — 19 pages: X display numbers (`Xorg :display`,
    `tightvncserver :display`, `broadwayd :DISPLAY`).
  Rejected with evidence: `=` and `%` appear almost exclusively glued to a
  preceding token (`--flag[=VALUE]` suffix syntax, samba's
  `user[%password]`) — exactly the false-positive shape this constraint
  exists to block; `&` (TeX `&format`) and `#` are shell metacharacters
  that never reach the matcher as plain words; `<`, `[`, `{`, `(`, quotes
  are placeholder/alternation syntax, not sigils. The seed list is in place
  up front because pages gain prefixes opportunistically on every future
  re-extraction.
- `response.py`:
  - `llm_option_to_store_option` (response.py:176) reads `prefix` and passes
    it to `Option`.
  - `sanitize_option_fields` (response.py:153) clears `prefix` when
    `positional` is empty (parallel to the existing positional-vs-flags rule)
    and drops values outside the sigil allowlist.
  - `normalize_option_fields`: if the model embeds the sigil in the positional
    name itself (`"positional": "@server"` with no prefix field), strip it
    into `prefix` — this is the most likely model mistake. The rule applies
    **only to allowlisted sigil characters**, so `<FILE>`-style placeholders
    are untouched.
- `postprocess.py` — `sanitize_option` (postprocess.py:33) gets the same
  prefix-requires-positional + allowlist rules, and every site that
  reconstructs an `Option` (postprocess.py:60, 76, 182) must carry `prefix`
  through — implementation may switch these to
  `opt.model_copy(update={...})` so future fields are free.

### Diff tooling (`explainshell/diff.py`)

Add `prefix` to `_OPT_FIELDS` (diff.py:25) and to `_FALSY_EQUIVALENT`
(diff.py:28, so `None` vs absent compares clean against old rows). `diff db`
then surfaces prefix changes during the data refresh.

### Web

No changes to normal rendering — matching happens server-side in the matcher;
`views.py:434` only relays `positional` for display. One small exception: the
DEBUG panel mirrors option fields (views.py:429, matcher.py:52), so add
`prefix` to that debug dict — otherwise the new matching hint is invisible
exactly when debugging why a token matched.

## Invariants to preserve

- **Quote-the-manpage contract:** help text remains the manpage's defining
  block verbatim; this plan changes *assignment*, never text content.
- **Degrade-safe data:** pages without `prefix` (the entire current corpus)
  must match byte-for-byte as today. The prefix pool must be a strict superset
  feature.
- **Sanitization parity:** response.py and postprocess.py enforce the same
  field rules (existing convention for positional-vs-flags); prefix follows it.
- **Plain `Store` for tooling, `CachingStore` only for prod web** (CLAUDE.md);
  nothing here touches store lifecycles, keep it that way.
- **Old-row compatibility:** options JSON is read back through Pydantic;
  missing `prefix` keys must not fail validation (default `None` handles it —
  verify no `extra="forbid"` style config interferes).

## Tests and validation

- **Unit (`tests/test_matcher.py` + fixture in `tests/helpers.py`):** add a
  fixture page with a prefixed positional (alongside `withmultipos`,
  helpers.py:197) and cover: prefix token claimed out of order (last
  position), prefix token in first position, non-prefixed tokens skipping the
  prefixed positional in ordered consumption, prefix token with no declared
  prefix falling through, two prefixed tokens reusing the positional, and —
  per review — a page whose **only** positional is prefixed (prefix token
  matches; non-prefix token goes unknown instead of raising on the empty
  ordered pool). Existing positional tests (test_matcher.py:77-109) must pass
  unchanged.
- **Suite:** matcher + models are on the web serving path → `make tests-all`.
- **e2e:** regenerate `tests/e2e/e2e.db` with a dig page and snapshot the
  issue-361 command (`dig ns foo.bar @8.8.8.8`) so the fix is pinned
  end-to-end.
- **Eval:** the prompt change touches the LLM extractor → run `/eval-llm`
  (baseline vs change per the CLAUDE.md stash workflow) before landing.
  Watch for perturbation on pages with no sigils — the new schema field
  should be a no-op for them (zero-option deltas, no positional churn) — and
  specifically check ssh/scp-shaped pages (`[user@]hostname`) for
  false-positive prefix emission.
- **Docs:** AGENTS.md's Data Model section enumerates `Option` fields and the
  matcher description — update both for `prefix` (workflow requirement #4 in
  AGENTS.md itself).
- **Manual check:** `python -m explainshell.manager diff db --mode llm:<model>
  manpages/arch/latest/1/dig.1.gz` shows `server` gaining `prefix: "@"`, then
  a local matcher run of `dig ns foo.bar @8.8.8.8` maps `@8.8.8.8` → server,
  `ns` → name, `foo.bar` → type.

## Migration / data

- Two dig rows need the new field: `ubuntu/26.04/1/dig.1.gz` and
  `arch/latest/1/dig.1.gz`. Re-extract both with `--overwrite` after the
  prompt change (preferred over hand-editing JSON — exercises the real
  pipeline).
- The rest of the corpus stays as-is; pages gain prefixes opportunistically on
  future re-extractions. No bulk re-extraction is part of this plan.
- Production: the DB is baked into the Docker image, so the fix reaches
  explainshell.com only via `make upload-live-db` + a push to `master`
  (deploy pipeline). The code change must deploy **with or before** the data
  change — old code reading a `prefix` key would only matter if Pydantic
  rejected unknown fields, which it doesn't by default, so ordering is
  flexible; still, ship code first as a courtesy.
- e2e fixture (`tests/e2e/e2e.db`) has no dig page; add one and snapshot the
  issue-361 command (`dig ns foo.bar @8.8.8.8`) so the end-to-end behavior is
  pinned, not just the matcher unit. The fixture DB needs regenerating as
  part of the build pass.

## Out of scope (recorded so it isn't silently dropped)

- `ns` vs `foo.bar` remain order-assigned (`name`, `type`) — semantically
  `ns` is the type. Fixing that needs value-set knowledge (the schema's
  `has_argument` allowed-values list extended to positionals, matched by
  membership). Deliberate follow-up, not part of this plan.
- UI-level truncation/expand of long help texts (the "overly long text"
  complaint) — orthogonal product decision.

## Open questions

Resolved by the dual review (codex + claude, 2026-06-12):

1. **Field name:** `prefix` — neither reviewer objected; it lives next to
   `positional`, which provides the context.
2. **Prompt scope:** synopsis-only — reinforced by the review's
   false-positive analysis (`[user@]hostname` lives in the synopsis too, so
   the real guard is the sigil allowlist, but synopsis-only keeps the
   instruction simple and conservative).

Resolved with the user (2026-06-12):

3. **e2e coverage:** yes — add a dig page to `tests/e2e/e2e.db` and snapshot
   the issue-361 command (folded into Tests and Migration above).
4. **Backfill breadth:** dig only; other pages gain prefixes
   opportunistically on future re-extractions.
5. **Allowlist contents:** seed the broader sigil vocabulary up front, but
   ground it in the corpus rather than guessing. A scan of all 61,322 raw
   SYNOPSIS sections yielded `@`, `+`, `:` as the real standalone sigils
   and disqualified the initially proposed `%` and `=` as glued-suffix
   artifacts (evidence folded into Extraction above).

No open questions remain.
