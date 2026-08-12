# Deploying to Cloudflare Pages

**One bundle deploys: the Studio.** It is public.

| Bundle | Build | Output | Project | Audience |
|---|---|---|---|---|
| Studio | `npm run deploy -w @dashboard/studio` | `studio/dist/` | `neural-bridge-studio` | Prospective customers, investors |

Live at **https://neural-bridge-studio.pages.dev**.

The Studio is a **separate workspace**, not a second entry point of `web/`
([ADR-0006](docs/adr/0006-internal-external-separation.md)). It cannot import the programme's
data, and two gates in its build enforce that rather than trusting it.

## Why it is public, and why that is safe

There is deliberately **no Cloudflare Access in front of it**. It is for investors and
prospective customers, who should not have to be added to an email allowlist to open a link.

That is only defensible because of what the bundle contains. Every claim on the site comes
from `studio/src/content/claims.ts`, which holds the disclosure line, and two checks fail the
build if anything else gets in: `verify-boundary.mjs` on the source imports and
`verify-bundle.mjs` on the built output. A visitor sees a product demo over invented fixture
data, labelled as such via `report.isSample`.

**This makes the gates load-bearing rather than tidy.** Anything that trips them is now a
public disclosure. Fix the copy, never the gate.

`X-Robots-Tag: noindex, nofollow` stays: a link is meant to be shared, but a demo running on
sample data should not be indexed as the product.

## The evidence scorecard is gone

The `neural-bridge-scorecard` Pages project was **deleted in Phase E**. It carried the
internal research vocabulary — metric names, baseline and control design, the four landmark
results, and internal study paths — and was serving publicly with no gate.

The internal evidence dashboard is now **local-only and shelved**. Its Express server stays
bound to `127.0.0.1` and is never published — see
[ADR-0001](docs/adr/0001-node-only-read-only-dashboard.md). Nothing is lost: the code is in
git, and `npm run build:scorecard -w web` still produces the bundle if it is ever needed
again. Publishing it again would mean recreating the project **and** putting Access in front
of it first.

## What actually ships

Both bundles are fully self-contained: no third-party script, no remote font, and **no runtime
network request of any kind**. Verified by grep against the built output — re-run this after any
change to either build:

| symbol | dist-scorecard | dist-studio |
|---|---|---|
| `/api/tree`, `/api/file`, `/api/raw`, `/api/health` | 0 | 0 |
| `fetchTree`, `fetchFile`, `rawUrl` | 0 | 0 |
| `papaparse`, `react-markdown`, `json-view-lite` | 0 | 0 |
| `safePath` | 0 | 0 |
| `fetch(`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `sendBeacon` | 0 | 0 |

**You do not run this by hand.** Both checks are wired into the builds and fail them:

```bash
npm run build -w @dashboard/studio    # boundary gate, then vite, then bundle gate
```

```bash
npm run build:scorecard -w web        # vite, then bundle gate (internal profile)
```

`scripts/verify-bundle.mjs` takes a profile. `internal` checks only the API surface and network
primitives; `external` adds the whole programme vocabulary, dataset names as whole words, and
the raw metric values. The scorecard is *supposed* to say "PR-AUC" — the Studio is not.

`scripts/verify-boundary.mjs` runs first on the Studio and rejects any import that leaves the
package. It exists because the obvious assumption is wrong: **npm hoists every workspace into
the root `node_modules`**, so `import "dashboard-web/…"` resolves even though the Studio does
not depend on it, and `../../web/src/…` is an ordinary relative path. A workspace is not a wall.

Raw values are written without the leading zero (`.2536`, not `0.2536`) because that is how a
minifier emits them — matching `0.2536` gives a silent false pass, which is exactly what hid
them the first time.

The network-primitive row reads zero only because both configs set
`build.modulePreload.polyfill = false`. Vite otherwise injects a polyfill that calls `fetch()`
on the page's own asset hrefs — harmless, never executed in a browser that supports
modulepreload, and blocked by `connect-src 'none'` regardless. It was removed so this table is
checkable rather than needing a footnote.

### Deep links

The Studio uses real URLs, which needs three things in step. Break any one and the site works
until someone follows a link or presses refresh:

- `BrowserRouter` in `studio/src/App.tsx`
- `studio/public/_redirects` containing `/*  /index.html  200`
- `base: "/"` in `studio/vite.config.ts`, **and** an absolute `/theme-init.js` in `index.html` —
  with a relative path, `/results` requests `/results/theme-init.js`, which 404s and silently
  stops honouring a pinned theme on exactly the pages people are sent links to.

The file browser is **absent** from both bundles, not merely unreachable — so none of the
path-traversal surface exists in what gets published. The Studio additionally contains no
scorecard, no story and no evidence links: it reaches only the client product.

### The Studio's one policy difference

`studio/public/_headers` matches the scorecard's policy except for one directive:

```
media-src 'self' blob:
```

The report previews the visitor's own cut by handing a `<video>` an object URL built from the
File they picked. Without `blob:` listed, media falls back to `default-src 'self'` and the
browser blocks it — the frame silently goes blank on the deployed build while working fine in
dev. It widens **media only**; `script-src`, `connect-src` and `object-src` are unchanged.

This is not a hole in the read-only contract. A blob URL can only address bytes the page
already holds, and `connect-src 'none'` still means the Studio cannot send that file, or
anything else, anywhere. The uploaded video never leaves the browser — see
[ADR-0003](docs/adr/0003-studio-behind-an-analysis-client.md).

### What the Studio shows is sample data

Every report in the deployed Studio renders the same fixture: the numbers are invented, the
shape is real. The page says so via `AnalysisReport.isSample`. **Do not remove that notice** —
without it the demo reads as a measurement, which is precisely the false precision the
programme refuses everywhere else.

Provenance links resolve to `github.com/MaxSartini/neural-bridge` (a **private** repo), so
they work for repo members and 404 for everyone else. See `web/src/lib/evidenceLink.ts`.

## The gate: Cloudflare Access

Zero Trust Free (50 users) is active on this account. Access gates the site at Cloudflare's
edge, so it applies to an existing deployment the moment you switch it on — no redeploy needed.

An earlier revision of this bundle carried an HTTP Basic Auth worker
(`public-scorecard/_worker.js`) as a workaround for Zero Trust's card-on-file requirement. It
was removed once Zero Trust was working: running both would stack two login prompts, and Access
is strictly better here (per-person identity, audit log, revoke one person without disturbing
the rest). Removing it also returns Pages to normal mode, so `_headers` supplies the security
headers again.

### 1. Authenticate (opens a browser)

```bash
npx wrangler login
```

### 2. Create the project

```bash
npx wrangler pages project create neural-bridge-scorecard --production-branch main
```

Reserves `neural-bridge-scorecard.pages.dev` while it still serves nothing.

### 3. Add the Access application

Cloudflare dashboard -> **Zero Trust** -> **Access** -> **Applications** -> *Add an application*
-> **Self-hosted**:

| Field | Value |
|---|---|
| Application name | `Neural Bridge Scorecard` |
| Session duration | `1 week` (avoids re-auth every glance on a phone) |
| Subdomain | `neural-bridge-scorecard` |
| Domain | `pages.dev` |

Then *Add policy*: name `Team`, action **Allow**, include **Emails** -> each teammate's address.
Include your own address or you will lock yourself out.

An application with no Allow policy is the usual mistake — add the policy before saving.

### 4. Deploy

```bash
npm run deploy:scorecard -w web
```

Builds and uploads in one step. Re-run it for every update.

### 5. Confirm the gate is live

Open the URL in a private window. You must get the Access login screen, **not** the scorecard.
If the scorecard appears, the policy is not attached — fix it before sharing the link. Access
takes effect immediately, with no redeploy.

## Deploying the Studio

Identical shape, different project. Steps 1 and 3–5 above apply unchanged; substitute
`neural-bridge-studio` for the project and subdomain, and `Neural Bridge Studio` for the Access
application name.

```bash
npx wrangler pages project create neural-bridge-studio --production-branch main
npm run deploy -w @dashboard/studio
```

Give it its own Access application rather than reusing the scorecard's. The two have different
audiences — the scorecard is internal evidence, the Studio is what a prospective customer or
investor is shown — and separate applications mean you can grant someone the demo without
handing over the programme's results.

## Local preview, exactly as Cloudflare will serve it

```bash
npm run preview:scorecard -w web
```

```bash
npm run preview -w @dashboard/studio
```

Run `wrangler pages dev` on `http://127.0.0.1:4321` and `http://127.0.0.1:4322`, which apply
`_headers` — so the CSP is live locally and a violation shows up as a console error before it
ever ships. Preview the Studio this way after any change to the report's frame preview: a
`media-src` mistake is invisible in the Vite dev server and only appears under the real policy.

## Security headers

Set by `web/public-scorecard/_headers`, copied to the build root. The policy is strict because
the bundle genuinely needs nothing external:

```
default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';
img-src 'self' data:; font-src 'self'; connect-src 'none';
frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'
```

Plus `nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`, a `Permissions-Policy`
that switches off geolocation/camera/microphone, and `X-Robots-Tag: noindex, nofollow`.

Two notes on this:

- `style-src` needs `'unsafe-inline'` because React and Recharts set inline `style` attributes.
  That is a style-only relaxation; script stays fully locked down.
- The theme-init script lives in `public-scorecard/theme-init.js` rather than inline in the HTML
  *specifically* so `script-src 'self'` can hold without `'unsafe-inline'` or a fragile hash.
  If you ever move it back inline, the page will silently stop honouring a pinned theme.
- `noindex` is a request to well-behaved crawlers, not enforcement. **Cloudflare Access is the
  actual boundary.**

## Updating the numbers

`web/src/data/scorecard.ts` is a hand transcription of `results/README.md`
([ADR-0002](docs/adr/0002-hand-curated-scorecard-data.md)). There is no checker. If a phase
concludes and that table changes, edit `scorecard.ts` by hand or the deployed scorecard will
silently show stale results.
