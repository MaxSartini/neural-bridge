import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import fs from "node:fs";
import path from "node:path";

/**
 * Tests for the traversal guard. This is the security boundary of the whole
 * dashboard, so these assert *which* rejection happens (400 = malformed input,
 * 403 = a well-formed path we refuse to serve), not merely that something threw.
 *
 * The guard reads REPO_ROOT and ALLOWED_TOP_LEVEL from ../config.js at call
 * time, so mocking that module moves the whole boundary into a throwaway temp
 * directory. safePath.ts itself is untouched by these tests — it has to stay
 * exactly as written for the audit that gates committing this directory.
 */

const sandbox = await vi.hoisted(async () => {
  const os = await import("node:os");
  const nodeFs = await import("node:fs");
  const nodePath = await import("node:path");
  // realpath the temp dir: on Windows and macOS os.tmpdir() is itself commonly
  // a link, and the guard's post-realpath check compares against REPO_ROOT
  // literally. An unresolved root would make every path look like an escape.
  const root = nodeFs.realpathSync(
    nodeFs.mkdtempSync(nodePath.join(os.tmpdir(), "nb-safepath-root-")),
  );
  const outside = nodeFs.realpathSync(
    nodeFs.mkdtempSync(nodePath.join(os.tmpdir(), "nb-safepath-outside-")),
  );
  return { root, outside };
});

vi.mock("../config.js", () => ({
  REPO_ROOT: sandbox.root,
  ALLOWED_TOP_LEVEL: new Set(["README.md", "results", "studies", "docs"]),
  PORT: 4319,
  MAX_INLINE_BYTES: 5 * 1024 * 1024,
}));

const { resolveSafePath } = await import("./safePath.js");
const { HttpError } = await import("./httpError.js");

/** Asserts the call throws an HttpError with exactly this status, and returns it. */
function expectRejection(userPath: string, status: number) {
  let thrown: unknown;
  try {
    resolveSafePath(userPath);
  } catch (err) {
    thrown = err;
  }
  expect(thrown, `expected resolveSafePath(${JSON.stringify(userPath)}) to throw`).toBeInstanceOf(
    HttpError,
  );
  expect((thrown as InstanceType<typeof HttpError>).status).toBe(status);
  return thrown as InstanceType<typeof HttpError>;
}

/** Windows file symlinks need Developer Mode; directory junctions do not. */
function canLink(): boolean {
  const probe = path.join(sandbox.outside, "__link_probe__");
  try {
    fs.symlinkSync(sandbox.outside, probe, "junction");
    fs.rmSync(probe, { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}

const linksWork = canLink();

beforeAll(() => {
  fs.mkdirSync(path.join(sandbox.root, "results"), { recursive: true });
  fs.mkdirSync(path.join(sandbox.root, "studies", "again"), { recursive: true });
  fs.mkdirSync(path.join(sandbox.root, "docs"), { recursive: true });
  fs.mkdirSync(path.join(sandbox.root, "internal"), { recursive: true });
  fs.writeFileSync(path.join(sandbox.root, "README.md"), "# root\n");
  fs.writeFileSync(path.join(sandbox.root, "pyproject.toml"), "[project]\n");
  fs.writeFileSync(path.join(sandbox.root, "results", "README.md"), "# results\n");
  fs.writeFileSync(path.join(sandbox.root, "internal", "secret.md"), "# secret\n");
  fs.writeFileSync(path.join(sandbox.outside, "loot.md"), "# loot\n");
});

afterAll(() => {
  fs.rmSync(sandbox.root, { recursive: true, force: true });
  fs.rmSync(sandbox.outside, { recursive: true, force: true });
});

describe("resolveSafePath — malformed input is a 400", () => {
  it("rejects an empty path", () => {
    expectRejection("", 400);
  });

  it("rejects a non-string path", () => {
    // The route passes `typeof req.query.path === "string" ? … : ""`, but the
    // guard defends itself rather than trusting that.
    expectRejection(undefined as unknown as string, 400);
  });

  it("rejects a NUL byte", () => {
    expectRejection("results/README.md\0.png", 400);
  });

  it("rejects a Windows drive letter", () => {
    // Double-covered on Windows (path.isAbsolute already says yes) but
    // single-covered on Linux, where isAbsolute("C:\\...") is false and the
    // colon check is the only thing standing between this and a resolve().
    // Mutation-checked: deleting the colon check leaves this test green on
    // Windows and only the ADS case below goes red.
    expectRejection("C:\\Windows\\win.ini", 400);
  });

  it("rejects an NTFS alternate data stream", () => {
    expectRejection("results/README.md::$DATA", 400);
  });

  it("rejects a POSIX absolute path", () => {
    expectRejection("/etc/passwd", 400);
  });

  it("rejects a backslash-rooted path", () => {
    expectRejection("\\results\\README.md", 400);
  });

  it("rejects a path that is only separators", () => {
    expectRejection("///", 400);
  });
});

describe("resolveSafePath — traversal is a 400", () => {
  it("rejects a trailing escape", () => {
    expectRejection("results/../../pyproject.toml", 400);
  });

  it("rejects a leading escape", () => {
    expectRejection("../AGENTS.md", 400);
  });

  it("rejects '..' buried mid-path", () => {
    expectRejection("studies/again/../../../internal/secret.md", 400);
  });

  it("rejects a backslash-separated escape", () => {
    expectRejection("results\\..\\..\\pyproject.toml", 400);
  });

  it("rejects a bare '.' segment", () => {
    // Harmless in isolation; refused because the guard normalizes nothing and
    // treats any relative segment as a signal rather than something to resolve.
    expectRejection("results/./README.md", 400);
  });

  it("rejects '..' even when the result would land back inside the root", () => {
    expectRejection("results/../results/README.md", 400);
  });
});

describe("resolveSafePath — outside the allowlist is a 403", () => {
  it.each([
    ["internal/secret.md"],
    ["pyproject.toml"],
    ["AGENTS.md"],
    ["src/neural_bridge/again/metrics.py"],
    [".git/config"],
  ])("refuses %s", (userPath) => {
    expectRejection(userPath, 403);
  });

  it("refuses a name that merely starts with an allowlisted one", () => {
    // The check is set membership on the first segment, not a prefix match, so
    // "README.md.bak" and "results-backup" get no free ride.
    expectRejection("README.md.bak", 403);
    expectRejection("results-backup/README.md", 403);
  });

  it("refuses a case variant of an allowlisted root", () => {
    // Windows would happily open "Results\README.md". The allowlist is
    // case-sensitive, so the guard refuses before the filesystem is consulted.
    expectRejection("Results/README.md", 403);
  });
});

describe("resolveSafePath — accepted paths resolve where they should", () => {
  it("resolves an allowlisted directory", () => {
    expect(resolveSafePath("results")).toBe(path.join(sandbox.root, "results"));
  });

  it("resolves an allowlisted file", () => {
    expect(resolveSafePath("results/README.md")).toBe(
      path.join(sandbox.root, "results", "README.md"),
    );
  });

  it("resolves the allowlisted root file", () => {
    expect(resolveSafePath("README.md")).toBe(path.join(sandbox.root, "README.md"));
  });

  it("accepts backslashes as separators", () => {
    expect(resolveSafePath("studies\\again")).toBe(path.join(sandbox.root, "studies", "again"));
  });

  it("collapses repeated separators", () => {
    expect(resolveSafePath("results//README.md")).toBe(
      path.join(sandbox.root, "results", "README.md"),
    );
  });

  it("returns a path that does not exist rather than 404ing", () => {
    // Documented contract: callers stat afterwards and decide. The guard only
    // answers "may this path be touched at all".
    expect(resolveSafePath("results/nope.md")).toBe(
      path.join(sandbox.root, "results", "nope.md"),
    );
  });
});

describe("resolveSafePath — symlink boundary", () => {
  it.skipIf(!linksWork)("refuses a link inside the root pointing outside it", () => {
    const link = path.join(sandbox.root, "studies", "escape");
    fs.symlinkSync(sandbox.outside, link, "junction");
    try {
      const err = expectRejection("studies/escape", 403);
      expect(err.message).toContain("symlink");
    } finally {
      fs.rmSync(link, { recursive: true, force: true });
    }
  });

  it.skipIf(!linksWork)("refuses a file reached through such a link", () => {
    const link = path.join(sandbox.root, "studies", "escape");
    fs.symlinkSync(sandbox.outside, link, "junction");
    try {
      const err = expectRejection("studies/escape/loot.md", 403);
      expect(err.message).toContain("symlink");
    } finally {
      fs.rmSync(link, { recursive: true, force: true });
    }
  });

  it.skipIf(!linksWork)("allows a link that stays inside the root", () => {
    const link = path.join(sandbox.root, "studies", "inside");
    fs.symlinkSync(path.join(sandbox.root, "docs"), link, "junction");
    try {
      expect(resolveSafePath("studies/inside")).toBe(link);
    } finally {
      fs.rmSync(link, { recursive: true, force: true });
    }
  });

  it("reports whether the symlink cases actually ran", () => {
    // A guard branch that silently never executes is worse than a failing test.
    // This surfaces the skip in the report rather than hiding it.
    expect(typeof linksWork).toBe("boolean");
    if (!linksWork) {
      console.warn("symlink creation unavailable — the realpath branch was NOT exercised");
    }
  });
});
