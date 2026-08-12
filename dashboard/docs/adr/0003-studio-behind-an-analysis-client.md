# 0003 — The Studio product sits behind a swappable AnalysisClient

**Status:** Accepted
**Date:** 2026-08-12

## Context

A design handoff added three client-facing screens — upload a video, watch it
process, read a response heat-map report — plus a landing page. This is the
product a customer or an investor sees, and the programme is now pushing
outward, not only inward.

The obvious way to build it is the wrong one. [ADR-0001](0001-node-only-read-only-dashboard.md)
established a read-only server with four GET routes bound to `127.0.0.1`, and
stated in its consequences that a feature needing the ML stack "cannot reuse
this server… it should be a separate service, and this dashboard should not
become the place inference lives." An upload endpoint would reopen that
decision, and the real service does not exist yet.

The alternative of waiting for a real service means no client surface at all,
which is the thing the programme most needs to show.

## Decision

Build the screens for real and put the *only* mocked thing behind one seam.

- `studio/api/analysisClient.ts` declares `AnalysisClient` — `createAnalysis`,
  `getAnalysis`, `getReport` — and the types they exchange.
- `studio/api/mockAnalysisClient.ts` implements it in-process: a module-level
  `Map`, a pure state-machine reducer in `studio/lib/pipeline.ts`, and one
  fixture. It issues no `fetch`.
- `studio/api/index.ts` **is the seam**: it re-exports the types and selects the
  adapter. Screens import from `../api` and never from an implementation
  module, so swapping the mock is one line there.

  *Amended.* This ADR originally claimed swapping was "a one-module change that
  touches no screen" while the singleton was exported from
  `mockAnalysisClient.ts` and all three screens imported it by that name — so
  the swap meant either editing every screen or leaving a file called
  `mockAnalysisClient` serving production. The barrel added in Phase F is what
  makes the original claim true.
- **No write endpoint is added to `dashboard/server`.** Nothing in `studio/`
  imports `api/client.ts`.
- The selected `File` never leaves the browser. It is held in
  `studio/lib/localVideo.ts`, deliberately *outside* `AnalysisClient`, because a
  real service would return a frame URL rather than hand a `File` back — keeping
  it out of the interface stops the seam from encoding a browser-only detail.
- Honesty is encoded in the types, not left to the UI. `PipelineStep.progress`
  is optional and present only when a step can actually measure itself, so an
  unmeasurable step gives the view nothing to draw a fake bar from.
  `AnalysisReport.isSample` travels with the report and the page renders a
  notice from it.
- **The error contract is part of the interface.** `getAnalysis` and `getReport`
  reject with `AnalysisNotFound` for an unknown id. Leaving failure unspecified
  meant three screens invented three different policies and the mock's
  `getReport` never rejected at all — so a report page rendered complete,
  plausible output for a run that had never existed.

## Consequences

- ADR-0001 is **upheld, not reopened**. The read-only server gains nothing; the
  allowlist is untouched; the traversal guard stays byte-identical.
- Because the mock makes no network call, the Studio ships as a static bundle
  under the existing `connect-src 'none'` CSP. That is what makes a shareable
  demo possible without a new security decision. See
  [DEPLOY.md](../../DEPLOY.md) for the one policy addition (`media-src blob:`)
  and why it does not widen the contract.
- **Every report shows the same fixture.** The shape is real and the numbers are
  not. This is flagged in the UI rather than pretended away, and it is the most
  likely reason someone mistakes the demo for a measurement — the `isSample`
  notice is load-bearing and must not be removed for looking untidy.
- Run state lives in page memory, so a reload loses an in-flight analysis. That
  is correct for a demo: a reload should not resurrect a run that was never
  really happening. The processing screen says so when an id is missing.
- Pipeline step durations in `pipeline.ts` are demonstration timings chosen so a
  visitor can watch a run finish. They are not a claim about real inference
  cost, and a real client ignores the table entirely.
- When the real service arrives it must honour the same two honesty rules —
  omit `progress` where it cannot measure, omit `etaSeconds` where it cannot
  estimate. Those are the parts of this interface worth defending.
