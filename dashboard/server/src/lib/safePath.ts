import fs from "node:fs";
import path from "node:path";
import { REPO_ROOT, ALLOWED_TOP_LEVEL } from "../config.js";
import { HttpError } from "./httpError.js";

/**
 * Resolves a repo-relative path from an untrusted query param into an absolute
 * filesystem path, guaranteed to stay inside REPO_ROOT and under an allowlisted
 * top-level entry. Throws HttpError on anything suspicious. Does not require the
 * path to exist (callers stat/readdir afterward); if it exists, also re-checks
 * the boundary after resolving symlinks.
 */
export function resolveSafePath(userPath: string): string {
  if (typeof userPath !== "string" || userPath.length === 0) {
    throw new HttpError(400, "path is required");
  }
  if (userPath.includes("\0")) {
    throw new HttpError(400, "invalid path");
  }
  if (userPath.includes(":")) {
    // blocks drive letters (C:\...) and NTFS alternate data streams (file::$DATA)
    throw new HttpError(400, "invalid path");
  }
  if (path.isAbsolute(userPath) || userPath.startsWith("/") || userPath.startsWith("\\")) {
    throw new HttpError(400, "invalid path");
  }

  const segments = userPath.split(/[\\/]+/).filter(Boolean);
  if (segments.length === 0) {
    throw new HttpError(400, "invalid path");
  }
  if (segments.some((s) => s === "." || s === "..")) {
    throw new HttpError(400, "invalid path");
  }
  if (!ALLOWED_TOP_LEVEL.has(segments[0])) {
    throw new HttpError(403, "path outside allowed roots");
  }

  const resolved = path.resolve(REPO_ROOT, ...segments);
  const rootWithSep = REPO_ROOT + path.sep;
  if (resolved !== REPO_ROOT && !resolved.startsWith(rootWithSep)) {
    throw new HttpError(403, "path escapes repo root");
  }

  if (fs.existsSync(resolved)) {
    const real = fs.realpathSync(resolved);
    if (real !== REPO_ROOT && !real.startsWith(rootWithSep)) {
      throw new HttpError(403, "path escapes repo root (symlink)");
    }
  }

  return resolved;
}
