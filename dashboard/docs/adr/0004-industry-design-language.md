# 0004 — Industry design language, light-first, on a light-dark() cascade

**Status:** Accepted
**Date:** 2026-08-12

## Context

The design handoff delivers the interface in **Industry**: a light technical
ground, Barlow Condensed over Barlow, and cards, figures and primary buttons
drawn as blueprint objects — square-cornered, hairline-bordered, carrying `+`
registration marks. Cards are line drawings, not elevated surfaces.

That inverts the Phase A3 token layer, which was dark-first with a periwinkle
accent, rounded corners, real elevation shadows, and forest `#123524` used as a
*surface*. In Industry the same forest is *ink*.

Two problems had to be solved rather than inherited:

1. **The handoff is light-only.** The user chose to keep dark mode, so a dark
   variant had to be designed. It is ours, not the designer's.
2. **The old cascade duplicated its palette.** Light values were written out
   twice — once under `@media (prefers-color-scheme: light)` and again under
   `:root[data-theme="light"]` — about eighty lines that had to stay in sync by
   hand.

## Decision

- **Light is the base.** `:root` declares `color-scheme: light dark` and every
  color is a single `light-dark()` pair. The pin rules do nothing but flip
  `color-scheme`. The two duplicated blocks are gone.
- **Theme-pin semantics are unchanged.** No pin follows the OS; a pin wins in
  both directions. `ThemeToggle.tsx` and both pre-paint scripts were not
  touched — they still only write `data-theme`.
- **The dark variant's rule:** the green band stays `#123524` in both themes,
  reading as ink against the light ground and as a lift above the dark one, so
  it stays recognisably itself. Green survives neither as text nor as a data
  mark once the ground goes dark, so `--accent` and the marks lighten. The
  primary button inverts to a paper fill, because "the one solid object on the
  board" has to be the light one on a dark ground.
- **Colors are explicit `light-dark()` pairs, never `color-mix()` over a
  `light-dark()` base.** Nesting two color functions inside an SVG presentation
  attribute stacks a second unverified behaviour on the first; explicit pairs
  cost a few lines and remove the risk.
- Radii go to zero, elevation collapses to hairline rings, and the spacing scale
  becomes Industry's 0.85× steps. Eight dead tokens were deleted.
- Inter is replaced by Barlow and Barlow Condensed, vendored as npm packages,
  imported once from `styles/fonts.ts` at only the latin subset and the weights
  in use — four static files against Inter's full variable axis across seven
  subsets.

## Consequences

- **`light-dark()` inside an SVG presentation attribute was verified, not
  assumed.** Recharts passes `fill="var(--viz-series-1)"`; reading the painted
  fill off a bar while flipping the pin gives `#377557` light and `#4f9d75`
  dark, re-colouring on the same DOM node with no re-render. Had it failed, the
  fallback was explicit `@media`/`[data-theme]` blocks for the `--viz-*` tokens
  only. It did not fail, so that fallback is not in the code.
- **The mark floor is a real constraint on the palette.** Every filled data mark
  must clear 3:1 against its own ground. A first sequential ramp starting at
  `#cfe3d8` measured **1.20** — the lowest bar was invisible. Both ramps were
  re-solved numerically: light runs 3.16 → 12.03, dark 3.30 → 9.50. On the light
  ground this caps how far the ramp can travel, which is a genuine loss of
  range accepted in exchange for every bar being visible. **Retune by measuring.**
- Dark mode is now a maintenance obligation on a design nobody signed off. Any
  new surface needs checking in both themes; the handoff cannot answer questions
  about the dark side.
- `light-dark()` requires a 2023-or-later browser. Acceptable: this is a local
  tool plus an access-gated demo, not a public site.
- The periwinkle accent, the elevation ramp's separating role, and the hero's
  radial bloom are gone. Separation is the hairline border now.
