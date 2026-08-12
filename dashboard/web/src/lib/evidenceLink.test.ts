import { describe, it, expect, afterEach, vi } from "vitest";

/**
 * `evidenceTarget` is the seam that lets the same scorecard components ship in
 * two bundles: the local dashboard (where an Express server can serve the doc)
 * and the static Cloudflare bundle (where it cannot, so provenance links go to
 * the private GitHub repo instead).
 *
 * MODE is read once at module load from a build-time define, so exercising the
 * other branch means resetting the module registry and re-importing.
 */

const GITHUB_BASE = "https://github.com/MaxSartini/neural-bridge/blob/main";

async function loadWithMode(mode: string | undefined) {
  vi.resetModules();
  if (mode === undefined) {
    vi.unstubAllEnvs();
  } else {
    vi.stubEnv("VITE_EVIDENCE_LINKS", mode);
  }
  return import("./evidenceLink");
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("evidenceTarget — local dashboard build", () => {
  it("points at the in-app doc route", async () => {
    const { evidenceTarget } = await loadWithMode(undefined);
    expect(evidenceTarget("studies/again/phase-05/README.md")).toEqual({
      mode: "app",
      to: "/doc/studies/again/phase-05/README.md",
    });
  });

  it("strips leading slashes so the route never doubles up", async () => {
    const { evidenceTarget } = await loadWithMode(undefined);
    expect(evidenceTarget("///results/README.md")).toEqual({
      mode: "app",
      to: "/doc/results/README.md",
    });
  });

  it("reports links as internal", async () => {
    const { evidenceLinksAreExternal } = await loadWithMode(undefined);
    expect(evidenceLinksAreExternal).toBe(false);
  });

  it("falls back to app mode for any unrecognized flag value", async () => {
    // The check is `=== "github"`, so a typo in the build define fails safe:
    // links stay in-app rather than pointing somewhere unintended.
    const { evidenceTarget, evidenceLinksAreExternal } = await loadWithMode("githbu");
    expect(evidenceLinksAreExternal).toBe(false);
    expect(evidenceTarget("results/README.md")).toEqual({
      mode: "app",
      to: "/doc/results/README.md",
    });
  });
});

describe("evidenceTarget — static scorecard build", () => {
  it("points at the private GitHub blob", async () => {
    const { evidenceTarget } = await loadWithMode("github");
    expect(evidenceTarget("studies/again/phase-05/README.md")).toEqual({
      mode: "external",
      href: `${GITHUB_BASE}/studies/again/phase-05/README.md`,
    });
  });

  it("strips leading slashes so the blob URL never doubles up", async () => {
    const { evidenceTarget } = await loadWithMode("github");
    expect(evidenceTarget("/results/README.md")).toEqual({
      mode: "external",
      href: `${GITHUB_BASE}/results/README.md`,
    });
  });

  it("reports links as external so callers can label them", async () => {
    const { evidenceLinksAreExternal } = await loadWithMode("github");
    expect(evidenceLinksAreExternal).toBe(true);
  });

  it("never emits an in-app route", async () => {
    // The whole point: the static bundle has no /doc route and no server, so an
    // "app" target there is a broken link on a deployed page.
    const { evidenceTarget } = await loadWithMode("github");
    expect(evidenceTarget("docs/plan.md").mode).toBe("external");
  });
});
