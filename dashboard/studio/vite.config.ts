import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Build config for Neural Bridge Studio — the client-facing product.
 *
 * This package cannot reach the internal evidence code. It does not depend on
 * `dashboard-web`, so `../../data/scorecard` is unresolvable by construction
 * rather than by convention — which is the point of the split. Everything the
 * Studio renders comes from its own `src/content/` or from `@dashboard/ui`,
 * which is presentational and knows nothing about the programme.
 *
 * There is no `VITE_EVIDENCE_LINKS` define because nothing here links to
 * evidence: the programme's internal document tree is not part of this surface.
 */
export default defineConfig({
  plugins: [react()],
  /**
   * Absolute, not "./". The site uses real URLs and a Cloudflare Pages SPA
   * rewrite, so `/results` is served the same index.html as `/`. With a
   * relative base, that page would resolve its assets against its own path and
   * request `/results/assets/…`, which 404s. Every deep link would break while
   * the home page kept working — the worst possible failure mode for a link you
   * hand to someone.
   */
  base: "/",
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Vite otherwise injects a modulepreload polyfill that calls fetch() on the
    // page's own asset hrefs. It never runs in a browser that supports
    // modulepreload, and `connect-src 'none'` would block it if it did.
    // Dropping it lets the bundle audit return a clean zero for network calls.
    modulePreload: { polyfill: false },
  },
});
