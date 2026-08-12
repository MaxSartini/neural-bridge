# 0006 — The external surface is a separate package with an enforced boundary

**Status:** Accepted
**Date:** 2026-08-12

## Context

The Studio was built as a directory inside `web/`, sharing one build root, one
`tsconfig.json` and one stylesheet with the internal evidence dashboard. Both
[ADR-0003](0003-studio-behind-an-analysis-client.md) and the header comments in
its Vite config asserted that the two surfaces were separate.

They were not. A grep of the built `dist-studio` bundle — the artifact intended
for prospective customers and investors — returned the programme's internals
verbatim:

```
romanNumeral:"I", title:"Original VEATIC event/spike breakthrough",
metricLabel:"Blocked event PR-AUC", real:.2536, baseline:.1969,
baselineLabel:"Trained AR", control:.184, controlLabel:"Shuffled features",
positiveGroups:"beats AR, shuffled, and random",
sourceDoc:"studies/original-veatic/v2-closure/README.md"
```

and, as visible UI text on the processing screen, `"Encoding video — V-JEPA
2.1"` and `"Predicting cortical response — TRIBE v2"` — the entire upstream
stack, named.

Two causes, one structural and one editorial:

- **One import.** `StudioLandingPage.tsx` imported `landmarks` to render four
  percentages. Rollup tree-shakes *exports*, not object *fields*, so every
  field of every record shipped — including four internal repo paths and the
  raw absolute values the UI never displayed. `againBridgeProgression` was
  genuinely absent because nothing imported it; that contrast is the whole
  lesson.
- **Hardcoded literals** in the pipeline step titles, the report's methodology
  prose, and the landing page's step copy.

The intent was documented in five places and enforced in none. `DEPLOY.md`'s
verification table greps only for network symbols, so it could never have
caught this class.

## Decision

**Separation is structural, and the structure is checked.**

- The Studio is its own npm workspace, `@dashboard/studio`, with its own
  `package.json`, `tsconfig.json`, Vite config, `public/` and stylesheet.
- `@dashboard/ui` holds the design language — tokens plus the five
  presentational primitives — so both surfaces share a look without sharing
  data. It knows nothing about the programme.
- **A disclosure line**, held in `studio/src/content/claims.ts`: outcomes and
  validated figures framed plainly may appear; upstream model names, metric
  names, dataset names, baseline and control vocabulary, internal paths and raw
  absolute values may not.
- **Two build gates**, both of which fail the build:
  - `scripts/verify-boundary.mjs` — allowlist over source imports. Legal only if
    relative and inside the package, or bare and a declared dependency.
  - `scripts/verify-bundle.mjs` — greps the built output for banned vocabulary,
    dataset names as whole words, raw values, the internal API surface, and
    runtime network primitives. Runs on both bundles with different profiles:
    the internal scorecard is *supposed* to say "PR-AUC".

## Consequences

- **The claim "unresolvable by construction" is false, and was corrected before
  it shipped.** This ADR was first drafted saying an npm workspace makes
  internal imports impossible. A probe disproved it: npm hoists *every*
  workspace into the root `node_modules`, so `import "dashboard-web/…"`
  resolves whether or not you depend on it; and `../../web/src/…` is an ordinary
  relative path that simply leaves the package. A package boundary is a
  convention npm does not enforce. `verify-boundary.mjs` exists because of that
  probe. **If a future review is tempted to drop it as redundant, it is not.**
- Both gates are mutation-checked, per CONVENTIONS.md. Injecting `V-JEPA`, a
  whole-word `AGAIN`, a raw value, a relative escape and a hoisted-package
  import each fail; the word "again" in ordinary prose does not.
- **Some duplication is now correct.** The four plain claim sentences exist in
  both `web/src/data/plainLanguage.ts` and `studio/src/content/claims.ts`. That
  is the boundary working: they are authored copy for two audiences, and a
  shared module would be a path between the packages. Do not "fix" it.
- The internal dashboard no longer links to the Studio. A cross-deployment link
  needs a real URL, so the "New analysis" call to action renders only when
  `VITE_STUDIO_URL` is set at build time. A dead button is worse than none.
- The Studio drops `papaparse`, `react-markdown`, `remark-gfm` and
  `react-json-view-lite`, which it never used.
- **The gate greps a minified bundle.** It catches vocabulary and known
  constants, not paraphrase. Someone can still describe the architecture in
  their own words and pass. It is a backstop against regression, not a
  substitute for reading what you publish.
- A third surface would need its own workspace, its own `_headers`, and its own
  entry in both gates. That cost is deliberate.

## Related

[ADR-0003](0003-studio-behind-an-analysis-client.md) still holds: the Studio
reaches no server, and the `AnalysisClient` seam is unchanged by this move. Its
claim that swapping the mock is "a one-module change" remains inaccurate for a
different reason — the singleton is exported from the implementation module —
which the parked architecture-review candidate addresses.
