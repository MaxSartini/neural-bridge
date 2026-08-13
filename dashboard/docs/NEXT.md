# What's next

## Where things stand

- **The Studio is live and public** at https://neural-bridge-studio.pages.dev, with no login.
- **The evidence dashboard is shelved.** Local-only, Cloudflare project deleted.
- **Everything is committed and pushed** at `7891fc7`.
- 133 frontend tests and 38 Python tests pass; both build gates pass; typechecks clean.

## The goal: structural soundness, not polish

**Aesthetics are not the focus.** The priority is that the logic and structure are sound, have
no known defects, and can be built upon — a house on rock, not on sand.

> This supersedes an earlier direction recorded here, which was to take the Studio to
> pudding.cool standard and write an ADR-0007 superseding the Industry design language. That
> is deferred, not cancelled. The root `AGENTS.md` is authoritative on direction; reconcile
> against it.

### The problem, measured

**41 `.tsx` files have zero tests, and the harness cannot test them.**
`dashboard/vitest.config.ts` sets `environment: "node"` on every project with
`include: ["src/**/*.test.ts"]` — `.tsx` is not collected and jsdom is deliberately absent.

Worse than the count is the shape: **every module extracted *for* testability is tested, and
every place those modules are *called* is not.** `stateAt` has twelve tests and cannot
plausibly be wrong; the polling loop calling it in `pages/AnalyzingPage.tsx` — timers, the
cancellation guard, the 600 ms redirect, the `AnalysisNotFound`-versus-transient
classification, the retry backoff — has none, and sits on the live public surface. The same
inversion holds for `bucketSeries` versus the heat-cell rounding, and for `resolveSafePath`
versus the `readdir` loop that calls it.

### The scoping filter

For each candidate ask: **when a real inference backend replaces `MockAnalysisClient`, does
this break or need rework?** Yes → foundation, do it now. No → finish, it waits.

This rules most of the parked list out, because it lives in the shelved `web/`.

### Order

1. **Make the frontend testable.** A jsdom vitest project collecting `.tsx`, for `studio` only
   — `web/` is shelved and should not grow dependencies. Update the comment in
   `vitest.config.ts` rather than quietly contradicting it.
2. **Extract and test the run lifecycle** out of `AnalyzingPage`. Prefer a module taking an
   `AnalysisClient` and a clock, testable in the existing `node` environment with a fake
   clock, over a render test of the same behaviour.
3. **Make `AnalysisInput` honest.** It collects `contentType`, `objective` and `notes`;
   the mock discards all three, and `Objective`'s doc comment claims it "shapes the report's
   emphasis". Wire them or remove them.
4. **Record what a real backend must honour** in an ADR — reject `AnalysisNotFound` for an
   unknown id, omit `progress` where a step cannot measure itself, omit `etaSeconds` where
   there is no honest estimate — plus the fact that a real backend means relaxing
   `connect-src 'none'` on a public, unauthenticated surface.
5. **Python**, per [`../../docs/backend-handoff.md`](../../docs/backend-handoff.md): test
   `resolve_backend()`, resolve `cuda`, make the all-skipped backend suite visible.

### Deferred visual work, already in place

- **Imagery slots.** `ImageSlot` applies the blueprint frame and duotone treatment and falls
  back to generated artwork. Drop a file into `studio/public/media/` and pass its path — no
  code change. See that directory's README for constraints.
- **Generated artwork.** `studio/src/components/Artwork.tsx` — frame strip, registration
  field, response map. Licence-free, theme-aware, replaceable.
- **Motion primitives.** `Reveal` and `useReveal`, deliberately restrained and fail-open.

### One hard-won lesson to carry forward

Three separate attempts to animate the report's timeline all **hid the data** when the
animation did not run — a Recharts `className` that never reached the path, then `both` fill
mode, then `forwards`. The line now renders as a plain stroke with no animation at all.

**Do not animate a primary data display.** Animate decoration; let the data be present from
the first frame. `useReveal` carries a 1200 ms fail-open timer for the same reason, and the
heat strip animates only `transform` because its opacity *is* the value.

## Parked work

Closed in Phase F (`d6b5a3c`) — do not re-open:

- The `AnalysisClient` seam. `api/index.ts` is now the seam, screens import from `../api`,
  `AnalysisNotFound` is in the interface, and `getReport` rejects on an unknown id. Nine tests.
  ADR-0003's inaccurate claim is amended in place.
- `forgetVideo` and `isImageExt`, both dead. `localVideo` now holds one file and evicts on
  write, which also closed a real leak — every upload was resident for the tab's lifetime.
- Root `typecheck` now covers `server/`.
- `verify-boundary.mjs` scanned comments and reported prose as an undeclared dependency; it
  strips comments now.

Still open:

| | Where | Note |
|---|---|---|
| **The polling loop is untested** while `stateAt` has 12 tests | `studio/src/pages/AnalyzingPage.tsx` | **Highest priority.** Live public surface, and Phase F added retry logic to it |
| Four copies of the cancellable-load pattern with incompatible error models | `web/src/pages/` | Shelved surface — payoff dropped. Two of the original six are typed now |
| `LandmarkCard`'s interface sits at the wrong altitude | `web/src/components/` | Shelved |
| The theme-pin contract is known in six places | both surfaces | One of the original seven went with the scorecard deployment |
| `Tag` fails the deletion test | `ui/src/Tag.tsx` | Buys the `TagVariant` vocabulary and little else |

The grilling session on the first of these was interrupted mid-round-one and never resumed.

## Audit items accepted, not fixed

From the Phase E audit, judged not worth blocking on. Revisit if the threat model changes:

- **TOCTOU between the path guard and the route's read.** Requires local write access to
  `results/`, at which point the attacker already has what the server would hand over.
- **`tree.ts` stats children without re-running the guard**, so a symlink's target size and
  mtime are visible. Content stays protected; existence disclosure only.
- **Windows reserved device names** (`results/CON`) reach `fs.stat`. Worst case is an odd
  error, not a read.
- **The 405 catch-all is registered after the routers**, so `GET /api/nope` returns 405
  rather than 404. The message is wrong; the behaviour is safe.
- **`LinkRenderer` does not scheme-check `href`.** Modern React blocks `javascript:` URLs, so
  this is defence-in-depth — but `ImageRenderer` already has the pattern to copy.
