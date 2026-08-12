import { describe, it, expect } from "vitest";
import { resolveLink } from "./linkRewrite";

/**
 * `resolveLink` decides what every link inside a rendered evidence doc points
 * at. It is the reason a Markdown file written for GitHub navigates correctly
 * inside this app, and the reason a link can quietly become a dead end.
 *
 * The last block pins behaviour that is arguably wrong. Those tests exist to
 * make the behaviour visible and to make a future change deliberate — not to
 * bless it.
 */

describe("resolveLink — links that leave the app", () => {
  it.each([
    ["https://pudding.cool/2025/04/wine-animals"],
    ["http://example.com/a/b"],
    ["HTTPS://Example.COM"],
  ])("treats %s as external", (href) => {
    expect(resolveLink("results/README.md", href)).toEqual({ kind: "external", href });
  });

  it("treats mailto: as external", () => {
    // No "//" in a mailto URL, so the protocol regex misses it and an explicit
    // check catches it. Both halves matter.
    expect(resolveLink("README.md", "mailto:someone@example.com")).toEqual({
      kind: "external",
      href: "mailto:someone@example.com",
    });
  });

  it("treats a bare fragment as external so the browser handles the jump", () => {
    expect(resolveLink("results/README.md", "#results-table")).toEqual({
      kind: "external",
      href: "#results-table",
    });
  });

  it("trims surrounding whitespace before deciding", () => {
    expect(resolveLink("README.md", "  https://example.com  ")).toEqual({
      kind: "external",
      href: "https://example.com",
    });
  });
});

describe("resolveLink — resolving relative to the containing doc", () => {
  it("resolves a sibling", () => {
    expect(resolveLink("studies/again/README.md", "notes.md")).toEqual({
      kind: "doc",
      path: "studies/again/notes.md",
    });
  });

  it("resolves a child directory", () => {
    expect(resolveLink("studies/README.md", "again/phase-05/README.md")).toEqual({
      kind: "doc",
      path: "studies/again/phase-05/README.md",
    });
  });

  it("resolves '..' against the doc's own directory", () => {
    expect(
      resolveLink("studies/again/phase-05/README.md", "../phase-04/README.md"),
    ).toEqual({ kind: "doc", path: "studies/again/phase-04/README.md" });
  });

  it("resolves against a doc at the repo root", () => {
    expect(resolveLink("README.md", "results/README.md")).toEqual({
      kind: "doc",
      path: "results/README.md",
    });
  });

  it("normalizes a redundant '.' segment", () => {
    expect(resolveLink("results/README.md", "./sub/x.md")).toEqual({
      kind: "doc",
      path: "results/sub/x.md",
    });
  });

  it("strips a fragment before resolving the path", () => {
    expect(resolveLink("README.md", "results/README.md#headline-numbers")).toEqual({
      kind: "doc",
      path: "results/README.md",
    });
  });

  it("accepts a Windows-style doc path", () => {
    // DocPage gets its path from the URL, but the server echoes back whatever
    // separator the caller used.
    expect(resolveLink("studies\\again\\README.md", "notes.md")).toEqual({
      kind: "doc",
      path: "studies/again/notes.md",
    });
  });
});

describe("resolveLink — classifying the target", () => {
  it("sends Markdown to the rendered doc view", () => {
    expect(resolveLink("README.md", "docs/plan.md")).toEqual({
      kind: "doc",
      path: "docs/plan.md",
    });
  });

  it("matches the extension case-insensitively", () => {
    expect(resolveLink("README.md", "docs/PLAN.MD")).toEqual({
      kind: "doc",
      path: "docs/PLAN.MD",
    });
  });

  it.each([[".svg"], [".png"], [".jpg"], [".jpeg"], [".gif"]])(
    "sends %s straight to the raw byte stream",
    (ext) => {
      expect(resolveLink("results/README.md", `figure${ext}`)).toEqual({
        kind: "raw",
        path: `results/figure${ext}`,
      });
    },
  );

  it("matches an image extension case-insensitively", () => {
    expect(resolveLink("results/README.md", "FIGURE.SVG")).toEqual({
      kind: "raw",
      path: "results/FIGURE.SVG",
    });
  });

  it.each([[".json"], [".csv"], [".jsonl"], [".py"], [".toml"]])(
    "sends %s to the file viewer",
    (ext) => {
      expect(resolveLink("results/README.md", `data${ext}`)).toEqual({
        kind: "view",
        path: `results/data${ext}`,
      });
    },
  );

  it("reads an extension-less link as a directory and lands on its README", () => {
    expect(resolveLink("README.md", "studies")).toEqual({
      kind: "doc",
      path: "studies/README.md",
    });
  });

  it("reads a trailing-slash link as a directory too", () => {
    expect(resolveLink("README.md", "studies/again/")).toEqual({
      kind: "doc",
      path: "studies/again/README.md",
    });
  });

  it("resolves a link back to the repo root as the root README", () => {
    expect(resolveLink("results/README.md", "../")).toEqual({
      kind: "doc",
      path: "README.md",
    });
  });
});

describe("resolveLink — pinned behaviour that is not obviously right", () => {
  it("defangs a javascript: URL into an in-app route instead of an href", () => {
    // Load-bearing: `javascript:` matches neither the "://" protocol regex nor
    // the mailto check, so it falls through to path handling and comes back as
    // a route, which Markdown.tsx hands to navigate(). It never reaches an
    // <a href>. Loosening the external check would turn this into a real hole.
    const target = resolveLink("README.md", "javascript:alert(1)");
    expect(target.kind).not.toBe("external");
  });

  it("lets a link escape the repo root, producing a path the server will refuse", () => {
    // posixNormalize pops on an empty stack, so surplus "../" segments are
    // silently absorbed. The server is the real boundary and answers 403 — but
    // the reader gets a dead link rather than a message.
    expect(resolveLink("results/README.md", "../../../../pyproject.toml")).toEqual({
      kind: "view",
      path: "pyproject.toml",
    });
  });

  it("misreads a leading-dot filename as a directory", () => {
    // `dotIndex > 0` is false for ".gitignore", so it is classified as having
    // no extension and rewritten to a README that does not exist.
    expect(resolveLink("README.md", ".gitignore")).toEqual({
      kind: "doc",
      path: ".gitignore/README.md",
    });
  });

  it("returns an empty external href for an empty link", () => {
    expect(resolveLink("README.md", "")).toEqual({ kind: "external", href: "" });
  });

  it("keeps a query string as part of the filename", () => {
    // Only "#" is split off, so "?" ends up inside the last segment and the
    // extension check sees ".md?v=2". Harmless today because evidence docs do
    // not carry query strings.
    expect(resolveLink("README.md", "results/README.md?v=2")).toEqual({
      kind: "view",
      path: "results/README.md?v=2",
    });
  });
});
