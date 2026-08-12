/**
 * Where the client-facing Studio lives, if it has been deployed.
 *
 * The Studio is a separate package and a separate Cloudflare Pages project
 * (ADR-0006), so linking to it needs an absolute URL rather than a route. Set
 * it at build time:
 *
 *   VITE_STUDIO_URL=https://neural-bridge-studio.pages.dev npm run build:scorecard -w web
 *
 * When it is unset, the call to action does not render at all. That is
 * deliberate: the design's "New analysis" button pointed at `/app/new`, which
 * stopped existing when the Studio moved out, and shipping a button that 404s
 * on the internal evidence dashboard is worse than shipping no button.
 */
const raw = import.meta.env.VITE_STUDIO_URL;

/** Absolute URL, or null when the Studio has not been wired up yet. */
export const studioUrl: string | null =
  typeof raw === "string" && /^https?:\/\//.test(raw) ? raw : null;
