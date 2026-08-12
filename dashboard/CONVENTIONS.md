# Dashboard conventions

Scope: `dashboard/` only. The repository-root `AGENTS.md` governs the VEATIC research
programme and does not apply here; nothing in this file authorizes research or compute work.

**Why this file is not named `AGENTS.md`.** `tests/test_veatic21_authority.py` asserts the
repository holds exactly one agent contract. A second `AGENTS.md` here would break that test
the moment `dashboard/` is committed, and the one-contract rule is worth more than the
filename. Agents working in this directory should read this file plus
[`CONTEXT.md`](CONTEXT.md).

## Read first

1. [`CONTEXT.md`](CONTEXT.md) — the domain glossary. Use these terms exactly.
2. [`docs/adr/`](docs/adr/) — decisions already taken. Do not re-litigate them; if friction is
   real enough to reopen one, say so explicitly and argue against its stated consequences.

## Hard constraints

- **This directory is committed as of Phase E**, after the security audit that gated it. The
  `/dashboard` line is gone from the repository `.gitignore`; `dashboard/.gitignore` now
  carries the rules, and it is deliberately duplicated with the root file so this directory
  stays safe to commit even if the root rules are edited. **Never remove `.wrangler/` from
  it** — `cache/cf.json` records the developer's city, postal code and latitude/longitude to
  five decimal places, and `tmp/` retains bundled worker sources with absolute local paths.
- **The Studio is public and unauthenticated.** Anything that reaches `studio/dist` is a
  disclosure to the open internet, not an internal tidiness question. The two gates in
  `scripts/` are the control; when one fails, fix the copy, never the gate.
- **Read-only server.** No write endpoints, ever. Four GET routes, bound to `127.0.0.1`, with
  a Host allowlist in front so DNS rebinding cannot reach them same-origin.
- **No external requests at runtime.** Fonts and libraries ship as npm packages, never CDN
  links. Any change that adds a network call to a third party is out of scope by default.
- **The allowlist is the security boundary.** Widening it (`server/src/config.ts`) is a
  deliberate decision that needs an ADR, not a config tweak.

## Review cadence

The user asked to stay ahead of this codebase rather than be handed finished work. Two
commands drive that. **Both are `disable-model-invocation: true` — the agent cannot fire them.**
The agent's job is to reach the checkpoint, stop, and say which command to type.

| When | Command | Purpose |
|---|---|---|
| **Before** implementing a new phase's plan | `/mattpocock-skills:grill-with-docs` | Stress-test the plan; extends `CONTEXT.md` and writes ADRs as decisions get pinned |
| **After** a phase ships and verifies | `/mattpocock-skills:improve-codebase-architecture` | Surface deepening opportunities as an HTML report, then grill the chosen one |

An agent finishing a phase must **not** roll straight into the next one. It stops, reports
verification results, and names the command.

## Running it

```bash
cd dashboard
npm install
npm run dev
```

Express on `127.0.0.1:4319`, Vite on `127.0.0.1:5173` (auto-opens, proxies `/api/*`).

## Phase history

- **Phase A** — evidence browser: file tree, Markdown rendering with internal-link rewriting,
  JSON/CSV/image viewers, traversal guard. ADR-0001.
- **Phase A2** — cross-phase scorecard: four landmark KPI cards, gain comparison, AGAIN bridge
  progression, motivating-failure chart. ADR-0002.
- **Phase A3** — brand palette and design language: token layer (elevation ramp + ink scale),
  theme pin with OS-following default, Inter, sliding nav pill, count-up KPIs, skeletons,
  chart colors moved from JS hexes to CSS variables. Presentation only — no API or data change.
- **Phase A4** — scroll-driven story at `/story`: one pinned chart, six steps, two acts.
  Reuses the scorecard's numbers; adds none.
- **Phase A4.5** — tests, one live bug, one owner for file kind. Vitest at the root with a
  `server` and a `web` project; 73 tests covering `safePath`, `linkRewrite` and
  `evidenceLink`, each mutation-checked. Breadcrumbs stopped emitting `/doc/<dir>` (a 415 on
  every intermediate crumb) — the `base` prop is gone and ancestors always link to `/browse`.
  The extension→kind decision moved into the new `shared` workspace, `/api/tree` now reports
  `kind`, and `TreeEntry.kind` became `entryType`.
- **Phase A5** — planned, not started: static evidence snapshot + a second adapter behind
  `fetchTree` / `fetchFile` / `rawUrl`. Two open questions before it starts — the deepest
  evidence path is 307 characters against Windows' 260-char `MAX_PATH` once mirrored under
  `dist-dashboard/evidence/`, and `ALLOWED_TOP_LEVEL` wants to move into `shared/`, which is a
  change to the security boundary.
- **Phase B0** — the Industry redesign and the Studio surface. Three decisions, three ADRs:
  [0003](docs/adr/0003-studio-behind-an-analysis-client.md) puts the client product behind a
  swappable `AnalysisClient` so it exists without a write endpoint;
  [0004](docs/adr/0004-industry-design-language.md) replaces the Phase A3 token layer with the
  Industry blueprint language, light-first on a `light-dark()` cascade, plus a dark variant that
  is ours rather than the designer's; [0005](docs/adr/0005-plain-gloss-layer.md) adds the
  plain-gloss layer — every card leads in plain English, keeps its exact metric beneath, and
  carries no digits in the gloss. `/` became a designed Overview (the raw README stays at
  `/doc/README.md`), the sliding nav pill went away with the green band that replaced it, and
  a second static bundle (`dist-studio`) now deploys alongside the scorecard.
- **Phase C** — full internal/external separation. The Studio bundle was shipping the
  programme's internals to a customer-facing artifact: upstream model names as visible UI
  text, metric and control vocabulary, raw metric values, and internal study paths — all from
  one import plus a handful of literals. The Studio is now its own workspace
  (`@dashboard/studio`) with a disclosure line in `studio/src/content/claims.ts`, the design
  language moved to `@dashboard/ui`, and **two build gates that fail the build**:
  `scripts/verify-boundary.mjs` (source imports) and `scripts/verify-bundle.mjs` (built
  output). Also: real URLs and a Pages SPA rewrite so a shared link survives a refresh,
  generated on-brand artwork with drop-in slots for real photography, and reveal-on-scroll
  motion that fails open. See [ADR-0006](docs/adr/0006-internal-external-separation.md).
- **Phase E** — the security audit, and the first commit. Four blockers, all artifact hygiene
  and all from one cause: the root `/dashboard` line was doing every bit of the ignoring, so
  deleting it to commit would have tracked `dist-scorecard/` and both `.wrangler/` trees —
  including the developer's geolocation and the full source of the Basic Auth worker
  `DEPLOY.md` describes as removed. Fixed with `dashboard/.gitignore`. Three real security
  fixes went in alongside: a `sandbox` CSP and `nosniff` on `/api/raw` (an SVG opened as a
  top-level document ran same-origin script), a Host allowlist against DNS rebinding, and
  `/api/health` no longer returning the absolute repo root.
  **The internal evidence dashboard is shelved from this point.** Its Cloudflare project is
  deleted and it is local-only. Do not resume work on `web/` or `server/` without saying so
  deliberately — all effort goes to `studio/`.
- **Phase B/C** — future, not started: a real inference API behind the `AnalysisClient`
  interface, replacing `MockAnalysisClient`. It still must not be built into this read-only
  server (see ADR-0001 consequences, upheld by ADR-0003).

## Publishing

Anything under `studio/` is published to people outside the programme. Two rules:

- **Never widen a gate to make a build pass.** If `verify-bundle.mjs` flags a term, the copy is
  wrong, not the list. Rephrase it in `content/claims.ts` without method vocabulary.
- **Never import across the package boundary**, including "just a type". A type import of
  `LandmarkComparison` documents the field names it is trying to hide.

## Testing

`npm test` from `dashboard/` runs both workspaces. Tests live next to their subject
(`src/lib/foo.test.ts`). Both projects run in `node` — everything tested so far is pure logic,
so jsdom is not a dependency; add it to the `web` project only when a component test needs it.

`safePath.test.ts` mocks `../config.js` rather than changing `safePath.ts`, so the guard stays
byte-identical for the audit that gates committing this directory. A test suite that has never
been seen failing is not known to test anything — new tests for security-relevant code get
mutation-checked against a scratch copy, never by editing real source.
