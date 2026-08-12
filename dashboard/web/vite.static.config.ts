import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { renameSync } from "node:fs";
import { resolve } from "node:path";

const OUT_DIR = "dist-scorecard";

/**
 * Cloudflare Pages serves `index.html` as the directory index. Our entry is
 * named `static.html` so it doesn't collide with the dashboard's own
 * `index.html` in source, so rename it on the way out. Safe because
 * `base: "./"` makes every asset reference relative to the document.
 */
function emitIndexHtml(): Plugin {
  // Taken from Vite's resolved config (already absolute) rather than __dirname,
  // which doesn't exist when the config loads as ESM.
  let outDir = "";
  return {
    name: "emit-index-html",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    closeBundle() {
      renameSync(resolve(outDir, "static.html"), resolve(outDir, "index.html"));
    },
  };
}

/**
 * Build config for the standalone scorecard bundle.
 *
 * Deliberately separate from vite.config.ts (the local dashboard dev server).
 * The entry reaches only ScorecardPage — so the API client, the file-browser
 * pages, and every route that reads the filesystem are absent from the output
 * rather than merely unreachable.
 */
export default defineConfig({
  plugins: [react(), emitIndexHtml()],
  base: "./",
  // Scorecard-only static assets: the CSP headers file and the pre-paint theme
  // script. Kept out of the dashboard's own public dir, which this build never
  // reads.
  publicDir: "public-scorecard",
  define: {
    "import.meta.env.VITE_EVIDENCE_LINKS": JSON.stringify("github"),
  },
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
    // Vite otherwise injects a modulepreload polyfill that calls fetch() on the
    // page's own asset hrefs. It never runs in a browser that supports
    // modulepreload (all current ones do), and `connect-src 'none'` would block
    // it if it did. Dropping it lets DEPLOY.md's verification grep return a
    // clean zero for network calls rather than one hit needing an explanation.
    modulePreload: { polyfill: false },
    rollupOptions: {
      // Relative to Vite's `root`, which defaults to this config's directory.
      input: "static.html",
    },
  },
});
