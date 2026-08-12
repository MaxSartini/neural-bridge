# Neural Bridge — frontend

Two surfaces live here. They share a design language and nothing else — that separation is
enforced by the build, not by convention.

| | **Studio** | **Evidence dashboard** |
|---|---|---|
| Who it is for | Prospective customers and investors | The programme's own authors |
| Where | **https://neural-bridge-studio.pages.dev** — public, no login | Local only, never deployed |
| Status | **Active.** All effort goes here | **Shelved** |
| Package | `studio/` | `web/` + `server/` |

## The live site

**https://neural-bridge-studio.pages.dev**

Public and shareable — send the link to anyone. It is deliberately not behind a login, because
investors should not need to be added to an allowlist to open it. It is `noindex`, so it will
not turn up in search results while it runs on sample data.

What it shows is a demo over invented fixture data, labelled as such on the report page. The
methodology figures are real; the per-video numbers are not.

## Run it locally

**Studio** — http://127.0.0.1:5174

```bash
npm install && npm run dev -w @dashboard/studio
```

**Evidence dashboard** (shelved, but still runnable) — http://127.0.0.1:5173, with its
read-only API server on http://127.0.0.1:4319:

```bash
npm install && npm run dev
```

Tests, across every workspace:

```bash
npm test
```

## The two gates

The Studio is public and unauthenticated, so the only thing standing between a stranger and
the programme's method is a pair of checks wired into its build. **Both fail the build.**

- **`scripts/verify-boundary.mjs`** — rejects any import that leaves the `studio/` package.
  An npm workspace is not a wall: npm hoists every workspace into the root `node_modules`, so
  `import "dashboard-web/…"` resolves even without a dependency, and `../../web/src/…` is an
  ordinary relative path. This check is what actually stops it.
- **`scripts/verify-bundle.mjs`** — greps the built output for upstream model names, metric
  names, dataset names, control vocabulary, internal repo paths and the raw metric values.

**When a gate fails, the copy is wrong — not the gate.** Rephrase it in
`studio/src/content/claims.ts` without method vocabulary. Never widen the banned list to make
a build pass.

## Deploying

```bash
npm run deploy -w @dashboard/studio
```

Builds through both gates and uploads. See [`DEPLOY.md`](DEPLOY.md) for the full picture,
including why `base: "/"`, `_redirects` and an absolute `theme-init.js` all have to agree or
deep links break.

## Read next

- [`docs/NEXT.md`](docs/NEXT.md) — what tomorrow is for, and the decision waiting at the top of it.
- [`CONVENTIONS.md`](CONVENTIONS.md) — hard constraints, review cadence, phase history.
- [`CONTEXT.md`](CONTEXT.md) — the domain glossary. Use these terms exactly.
- [`docs/adr/`](docs/adr/) — six decisions already taken. Do not re-litigate them; argue
  against their stated consequences if you need to reopen one.

## Layout

```
shared/   fileKind — the one owner of extension→kind, used by server and web
ui/       @dashboard/ui — design tokens and the five presentational primitives
server/   read-only Express API, 127.0.0.1 only, four GET routes
web/      the evidence dashboard (shelved)
studio/   the client-facing product (active)
scripts/  the two build gates
```
