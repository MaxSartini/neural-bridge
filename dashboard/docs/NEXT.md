# What's next

Written at the end of Phase E so tomorrow starts from a decision, not a blank page.

## Where things stand

- **The Studio is live and public** at https://neural-bridge-studio.pages.dev, with no login.
- **The evidence dashboard is shelved.** Local-only, Cloudflare project deleted. There is
  enough internal documentation; effort goes to the customer surface now.
- **Everything is committed.** First time — Phases A–C existed only as untracked files on one
  disk until Phase E.
- 124 tests pass, both build gates pass, both typechecks clean.

## The goal: take the Studio to pudding.cool standard

Bold visual storytelling, scrollytelling, playful interactive graphics, editorial confidence.
The product has a genuinely interesting thing to show — where attention moves across a cut —
and it is currently shown as a restrained wireframe.

### The decision waiting at the top of it

**ADR-0004 and The Pudding cannot both govern this surface.**

ADR-0004 fixed the Industry design language: square corners, hairline borders, transparent
cards, *no decorative colour, no elevation, one accent*. The Pudding is close to the opposite
— maximalist, colourful, animated, generous with imagery.

This is a real architectural decision, not a styling preference, and drifting into it would
leave the codebase governed by an ADR nobody is following. Take it deliberately:

- **Write ADR-0007 superseding ADR-0004 for the external surface only.** The internal code is
  shelved and can keep the wireframe language it was designed in.
- The split already supports this. `@dashboard/ui` holds the shared tokens and primitives;
  `studio/src/styles/studio.css` is already the Studio's own stylesheet. The question is how
  much of `ui/` the Studio keeps sharing versus forking.

Worth being honest about the tension in the other direction too: the wireframe language is
what makes the product *look* like measurement rather than marketing, and the honesty framing
("Calibrated, not overpromised") is load-bearing with investors. Going maximalist should not
quietly cost that.

### Already in place for it

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

From the architecture review, unaddressed and still valid:

| | Where |
|---|---|
| `AnalysisClient` seam is nominal — screens import the adapter, not the interface; `getReport` never rejects | `studio/src/api/` |
| Six copies of the cancellable-load pattern, four incompatible error models | both surfaces' pages |
| The polling loop is untested while `stateAt` has 12 tests — the risk is in the caller | `studio/src/pages/AnalyzingPage.tsx` |
| `LandmarkCard`'s interface sits at the wrong altitude | `web/src/components/` (shelved) |
| The theme-pin contract is known in seven places | across both surfaces |
| `forgetVideo` and `isImageExt` are exported and never called | `studio/src/lib/localVideo.ts`, `shared/src/fileKind.ts` |
| Root `typecheck` skips `server/` | `dashboard/package.json` |

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
