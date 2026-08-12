import { Router } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { ALLOWED_TOP_LEVEL } from "../config.js";
import { resolveSafePath } from "../lib/safePath.js";
import { asyncHandler } from "../lib/asyncHandler.js";
import { HttpError } from "../lib/httpError.js";
import { fileKindForPath, type FileKind } from "@dashboard/shared";

export const treeRouter = Router();

interface TreeEntry {
  name: string;
  path: string;
  /** Directory or file. Named apart from `kind` so "kind" means exactly one thing. */
  entryType: "dir" | "file";
  ext: string;
  /** Viewer classification for files; null for directories and unknown extensions. */
  kind: FileKind | null;
  size: number;
  mtimeMs: number;
}

function entryFor(name: string, relativePath: string, stat: { isDirectory(): boolean; size: number; mtimeMs: number }): TreeEntry {
  const isDir = stat.isDirectory();
  return {
    name,
    path: relativePath,
    entryType: isDir ? "dir" : "file",
    ext: path.extname(name),
    kind: isDir ? null : fileKindForPath(name),
    size: stat.size,
    mtimeMs: stat.mtimeMs,
  };
}

treeRouter.get(
  "/tree",
  asyncHandler(async (req, res) => {
    const rawPath = typeof req.query.path === "string" ? req.query.path : "";

    if (rawPath === "") {
      // Root listing is synthesized, never a real readdir of the repo root —
      // internal/, src/, tests/, .git/, pyproject.toml stay structurally unreachable.
      const entries: TreeEntry[] = [];
      for (const name of ALLOWED_TOP_LEVEL) {
        const abs = resolveSafePath(name);
        const stat = await fs.stat(abs).catch(() => null);
        if (!stat) continue;
        entries.push(entryFor(name, name, stat));
      }
      sortEntries(entries);
      res.json({ path: "", entries });
      return;
    }

    const abs = resolveSafePath(rawPath);
    const stat = await fs.stat(abs).catch(() => null);
    if (!stat) throw new HttpError(404, "not found");
    if (!stat.isDirectory()) throw new HttpError(400, "not a directory");

    const dirents = await fs.readdir(abs, { withFileTypes: true });
    const entries: TreeEntry[] = [];
    for (const dirent of dirents) {
      const childRelative = path.posix.join(rawPath.replace(/\\/g, "/"), dirent.name);
      const childAbs = path.join(abs, dirent.name);
      const childStat = await fs.stat(childAbs).catch(() => null);
      if (!childStat) continue;
      entries.push(entryFor(dirent.name, childRelative, childStat));
    }
    sortEntries(entries);
    res.json({ path: rawPath, entries });
  }),
);

function sortEntries(entries: TreeEntry[]): void {
  entries.sort((a, b) => {
    if (a.entryType !== b.entryType) return a.entryType === "dir" ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}
