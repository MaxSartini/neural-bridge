import { Router } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { MAX_INLINE_BYTES } from "../config.js";
import { resolveSafePath } from "../lib/safePath.js";
import { textKindForExt } from "@dashboard/shared";
import { asyncHandler } from "../lib/asyncHandler.js";
import { HttpError } from "../lib/httpError.js";

export const fileRouter = Router();

fileRouter.get(
  "/file",
  asyncHandler(async (req, res) => {
    const rawPath = typeof req.query.path === "string" ? req.query.path : "";
    const abs = resolveSafePath(rawPath);

    const ext = path.extname(abs);
    const kind = textKindForExt(ext);
    if (!kind) {
      throw new HttpError(415, `unsupported file type: ${ext || "(no extension)"}`);
    }

    const stat = await fs.stat(abs).catch(() => null);
    if (!stat) throw new HttpError(404, "not found");
    if (!stat.isFile()) throw new HttpError(400, "not a file");

    const truncated = stat.size > MAX_INLINE_BYTES;
    let content: string;
    if (truncated) {
      const handle = await fs.open(abs, "r");
      try {
        const buffer = Buffer.alloc(MAX_INLINE_BYTES);
        await handle.read(buffer, 0, MAX_INLINE_BYTES, 0);
        content = buffer.toString("utf8");
      } finally {
        await handle.close();
      }
    } else {
      content = await fs.readFile(abs, "utf8");
    }

    res.json({
      path: rawPath,
      ext,
      kind,
      size: stat.size,
      truncated,
      mtimeMs: stat.mtimeMs,
      content,
    });
  }),
);
