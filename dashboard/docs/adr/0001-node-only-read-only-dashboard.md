# 0001 — Node-only, read-only, localhost-bound dashboard

**Status:** Accepted
**Date:** 2026-08-11 (recorded retroactively; decision taken during Phase A)

## Context

Neural Bridge is a Python research programme. The obvious instinct was to serve its evidence
from the existing Python toolchain. Three facts argued against it:

- The dashboard only reads local text files. It needs no ML dependencies whatsoever.
- `uv`, the repo's normal Python toolchain, is not installed on this machine. Node v20 / npm 10
  are already present and working.
- The one genuinely security-sensitive piece is a path-traversal guard on an endpoint that
  takes a file path as a query parameter. That wants to be small, plain, and reviewable in one
  sitting — the user's stated condition for ever committing this code.

## Decision

- Node/TypeScript only. No Python, no venv, no pip.
- A small Express server (`server/`) exposing four **GET-only** routes — `/api/health`,
  `/api/tree`, `/api/file`, `/api/raw` — bound to `127.0.0.1`. No write endpoints exist.
- Path safety is an **allowlist**, not a blocklist: the repository root listing is synthesized
  rather than read from disk, and every other request's first path segment is checked against
  `README.md`, `results`, `studies`, `docs` before the filesystem is touched. See
  [Allowlist root](../../CONTEXT.md#allowlist-root).
- The traversal guard lives in one file, `server/src/lib/safePath.ts`, deliberately kept out
  of Vite's middleware lifecycle so it can be audited standalone.
- Vite's dev proxy joins the two so the browser only ever talks to one origin. No `cors()`.
- `/dashboard` is in the repository `.gitignore`. Nothing here is committed until the user has
  audited it.

## Consequences

- Adding a feature that needs the ML stack (running actual inference — the future Phase B/C)
  cannot reuse this server. That is accepted: it should be a separate service, and this
  dashboard should not become the place inference lives.
- `internal/` and `AGENTS.md` are unreachable by construction. Exposing them later is a
  deliberate allowlist edit, not a config toggle.
- The gitignore entry is a safety net against `git add -A`, **not a guarantee** — an explicit
  `git add dashboard/<file>` would still stage it. The real guarantee is behavioral.
- No production build exists. Dev-server only, single local user, localhost bind is the
  security boundary.
