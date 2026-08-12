/**
 * One owner for "where does a piece of evidence point".
 *
 * The scorecard's provenance links had `/doc/${sourceDoc}` hardcoded at four
 * call sites (LandmarkCard, GainBarChart, ProgressionBarChart, ScorecardPage). That
 * only works inside the local dashboard, where an Express server is there to
 * serve the file. The static scorecard bundle has no server and no `/doc`
 * route, so the same links have to resolve somewhere else — the private GitHub
 * repo, which exactly the intended readers can already open.
 *
 * Callers ask for a target and render it; they no longer know the URL scheme.
 */

/** Set at build time. `app` = local dashboard, `github` = static bundle. */
const MODE: "app" | "github" =
  import.meta.env.VITE_EVIDENCE_LINKS === "github" ? "github" : "app";

const GITHUB_BLOB_BASE = "https://github.com/MaxSartini/neural-bridge/blob/main";

export type EvidenceTarget =
  /** An in-app route — render with react-router's <Link>. */
  | { mode: "app"; to: string }
  /** An off-site URL — render with a plain <a>, new tab. */
  | { mode: "external"; href: string };

/**
 * Resolves a repo-relative evidence path (e.g. `studies/again/.../README.md`)
 * to something the current build can actually navigate to.
 */
export function evidenceTarget(sourceDoc: string): EvidenceTarget {
  const clean = sourceDoc.replace(/^\/+/, "");
  if (MODE === "github") {
    return { mode: "external", href: `${GITHUB_BLOB_BASE}/${clean}` };
  }
  return { mode: "app", to: `/doc/${clean}` };
}

/** True when provenance links leave the app — used to label them for the reader. */
export const evidenceLinksAreExternal = MODE === "github";
