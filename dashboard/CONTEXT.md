# Dashboard domain model

Ubiquitous language for the Neural Bridge evidence dashboard (`dashboard/`). This is the
frontend tool's vocabulary, distinct from the VEATIC research programme contract in the
repository-root `AGENTS.md`.

Use these terms exactly in code, commits, and review conversations. When a term here has a
name in the code, that name is authoritative — rename the code or the glossary, never let
them drift.

## Evidence

**Evidence file** — any Markdown, JSON, JSONL, CSV, image, or source file under an
[Allowlist root](#allowlist-root) that the dashboard can display. Evidence is *read-only and
authored elsewhere*: the dashboard never writes it and has no write endpoints at all.

**Allowlist root** — one of the exactly four top-level paths the server will serve:
`README.md`, `results/`, `studies/`, `docs/`. The repository root listing is **synthesized**,
not read from disk, so everything else (`internal/`, `src/`, `.git/`, `AGENTS.md`,
`pyproject.toml`) is structurally unreachable rather than merely filtered.

**Safe path** — a request path that has survived `server/src/lib/safePath.ts`: no null bytes,
no colons, not absolute, no `.`/`..` segment, first segment in the allowlist, resolves inside
`REPO_ROOT`, and still inside `REPO_ROOT` after `realpath` (which catches a symlink planted in
the tree pointing outward). Anything that is not a safe path is a 400 or 403, never content.

**File kind** — the extension-derived classification (`markdown`, `json`, `jsonl`, `csv`,
`code`, `text`, `image`) that decides which viewer the client mounts. Parsing happens
client-side; the server only labels. Owned by exactly one module,
`shared/src/fileKind.ts`, which both workspaces import — the `shared` workspace exists for
this. `/api/file` returns only the text kinds; `/api/tree` reports a kind per file and `null`
for directories.

**Entry type** — a tree entry's `dir` / `file` distinction, carried as `entryType`. Named apart
from [File kind](#evidence) on purpose: an entry has both, and reusing one word for two ideas
is what let `FileTree` re-derive `.md is markdown` for itself.

**Oversized file** — an evidence file past the 5MB inline cap. Served truncated with a
`truncated: true` flag and a designed notice, not an error page.

## Scorecard

**Scorecard** — the curated cross-phase visualization at `/scorecard`. Its numbers are
**hand-transcribed** from `results/README.md` into `web/src/data/scorecard.ts`, because each
research phase's `result.json` carries its own field names and `schema_version`. See
[ADR-0002](docs/adr/0002-hand-curated-scorecard-data.md).

**Landmark** — one of the programme's four headline wins (I Original VEATIC event/spike,
II AGAIN event/spike reconstruction, III Continuous response intelligence, IV Video-only
zero-label inference). Each carries a primary metric, a baseline gain, a control gain, a
positive-group count, and a `sourceDoc` for provenance.

**Baseline** / **Frozen AR baseline** — the trained autoregressive predictor a landmark is
measured against. Distinct from a **control lane**.

**Control lane** — a matched null/shuffled/random condition. The scorecard always reports the
*strongest* control, not the weakest, so the gain shown is the conservative one.

**Progression stage** — one step in an ordered development story (the six-stage AGAIN event
bridge, or the three-bar motivating-failure comparison). Rendered by `ProgressionBarChart`
with a sequential single-hue ramp keyed to rank.

**Motivating failure** — the recorded result where naive feature fusion made a strong
predictor *worse*. It is evidence, not an embarrassment: it is the reason the bridge exists,
and it stays on the scorecard deliberately.

**Story** — the scroll-driven telling of the bridge result at `/story`. Argues the same numbers
as the [Scorecard](#scorecard) in sequence rather than in a grid. Built from `scrolly/steps.ts`;
adds no new figures.

**Act** — one half of the Story, each pinned to its own benchmark. Act one is the
[Motivating failure](#scorecard) on the *early blocked* AGAIN benchmark; act two is the
six-stage progression on the *grouped* benchmark. **The two are not on the same scale and their
numbers must never be compared across the break** — the chart names its benchmark at every step
for exactly this reason.

**Step** — one beat of an Act: a heading, a paragraph, and the set of bars visible at that
point. Hidden bars keep their slot on the category axis, so bars rise in place rather than the
chart re-flowing.

## Surfaces

**Internal surface** — the evidence dashboard (`web/`) and the scorecard bundle
built from it. Its audience is the programme's own authors and anyone they grant
Access to. It says "Blocked event PR-AUC" because that is the correct term for
its reader.

**External surface** — Neural Bridge Studio (`studio/`), the client-facing
product. Its audience is prospective customers and investors: people not under
NDA who may work for, advise, or later fund a competitor.

**Disclosure line** — what the [External surface](#surfaces) may say about how
the system works. Outcomes and validated figures framed plainly are allowed.
Upstream model names, metric names, dataset names, baseline and control
vocabulary, internal repo paths and raw absolute metric values are not. Held in
`studio/src/content/claims.ts` and enforced by two build gates
(`scripts/verify-boundary.mjs`, `scripts/verify-bundle.mjs`). See
[ADR-0006](docs/adr/0006-internal-external-separation.md).

**External claim** — one outward statement: a digit-free `plain` sentence, an
approved `figure`, and a `context` saying what the figure measures in plain
words. Distinct from a [Plain gloss](#plain-language), which is the internal
surface's version and sits beside the precise metric rather than replacing it.

**UI package** — `@dashboard/ui`. The Industry design language as code: tokens,
base type, the blueprint frame, tags, buttons, the nav band, the theme toggle.
Purely presentational, and that is load-bearing — it is what lets both surfaces
look identical without the external one gaining a path to programme data.

**Package boundary** — the rule that `studio/` may import only its own source
and its declared dependencies. Not a convention: npm hoists every workspace into
the root `node_modules`, so `import "dashboard-web/…"` resolves whether or not
you depend on it, and `../../web/src/…` is an ordinary relative path. A probe
proved both holes, which is why `verify-boundary.mjs` exists.

## Plain language

**Plain gloss** — the plain-English sentence that *leads* a card, chart or hero, with the exact
metric kept directly beneath it as the technical line. Neural Bridge is read by prospective
customers and investors now, not only by the people who ran the programme, and a card labelled
"Blocked event PR-AUC" loses that reader on sight. A gloss adds a way in; it never replaces the
precise wording. Every gloss lives in `web/src/data/plainLanguage.ts` — one file, so the claims
the product makes can be reviewed in one sitting.

Two rules bind a gloss, both enforced or explained in that module:

- **No digits.** Figures stay in the numeral and the technical line, beside the metric that
  defines them. `+28.80%` means a gain in blocked event PR-AUC over a trained AR baseline;
  lifted into a friendly sentence it reads as a far broader claim than the evidence supports.
  `plainLanguage.test.ts` fails the build on a digit — spell the quantity out or move it.
- **Inside the Honest boundaries.** The root `README.md` lists what Neural Bridge does not
  claim: mind reading, individual profiling, medical inference, universal emotion recognition,
  exact second-by-second trajectories, guaranteed audience or commercial outcomes. Simplifying
  is not licence to overclaim.

**Technical line** — the precise half beneath a [Plain gloss](#plain-language). For landmarks
and datasets it is *composed from* `scorecard.ts` by `landmarkTechnicalLine` /
`datasetTechnicalLine` rather than restated, because ADR-0002 already carries a manual-sync
risk on those figures and a second transcription would double it.

## Presentation

**Theme pin** — an explicit `light`/`dark` choice in `localStorage.theme`, stamped onto
`<html data-theme>`. **No pin means follow the OS** — the attribute is absent, not set to a
default. A pin beats `prefers-color-scheme` in both directions. These semantics are unchanged
since Phase A3; what changed is the mechanism. `:root` now declares `color-scheme: light dark`
and every color is a single `light-dark()` pair, so the pin only has to flip `color-scheme`
instead of restating the whole palette. The two duplicated light blocks are gone.

**Blueprint frame** — the wireframe frame every card, figure and primary button wears: square
corners, a 1px `--divider` border, a transparent fill, and four **registration marks**. Cards
are line drawings, never elevated surfaces. Rendered by `components/Blueprint.tsx`, which is a
component rather than a bare class precisely because the frame cannot be drawn without its four
mark elements and hand-writing them at each call site is how one goes missing.

**Registration mark** — the `+` crosshair sitting 6px outside each corner of a
[Blueprint frame](#presentation), drawn by an `aria-hidden` `<i class="corner tl|tr|bl|br">`.
Never drop them; on the green band they lighten to paper so they read as drawn-on.

**Band** — the forest `#123524` field the nav sits in, with type reversed to paper. Identical in
both themes: it reads as *ink* against the light ground and as a *lift* above the dark one, so
it stays recognisably itself. It also carries the current route — the Phase A3 sliding pill was
removed because the band plus `aria-current` already says which page you are on.

**Token** — a CSS custom property in `web/src/styles/global.css`. Colors belong in tokens, not
in JS: chart marks are handed `var(--viz-seq-N)` strings so they re-resolve on a theme flip
without a re-render. Verified: `fill="var(--viz-series-1)"` resolves a nested `light-dark()`
inside an SVG presentation attribute and re-paints on the same DOM node when the pin changes.

**Ink level** — `--ink` / `--ink-2` / `--ink-3`. Every piece of text is assigned one
deliberately; that assignment is what makes the interface read as calm rather than flat.

**Mark floor** — the 3:1 contrast minimum a filled data mark must clear against `--ground`.
It is a real constraint on the palette, not a checkbox: it caps how faint the light end of the
sequential ramp may be, and on the light ground it is what limits the ramp's total travel. The
ramp's measured range is recorded beside it in `global.css`. Retune by measuring, not by eye —
a first attempt at the light ramp measured 1.20 and the lowest bar was invisible.

**Elevation ramp** — `--page` / `--canvas` / `--surface` / `--inset`. Under the blueprint
language these carry only the surfaces that genuinely need a fill — input fills, empty progress
tracks, code blocks, and sticky table headers that must occlude. Separation elsewhere is the
hairline border, not a surface change.
