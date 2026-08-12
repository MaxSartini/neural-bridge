import { Router } from "express";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { resolveSafePath } from "../lib/safePath.js";
import { mimeTypeForExt } from "@dashboard/shared";
import { asyncHandler } from "../lib/asyncHandler.js";
import { HttpError } from "../lib/httpError.js";

export const rawRouter = Router();

rawRouter.get(
  "/raw",
  asyncHandler(async (req, res) => {
    const rawPath = typeof req.query.path === "string" ? req.query.path : "";
    const abs = resolveSafePath(rawPath);

    const stat = await fsp.stat(abs).catch(() => null);
    if (!stat) throw new HttpError(404, "not found");
    if (!stat.isFile()) throw new HttpError(400, "not a file");

    const ext = path.extname(abs);
    res.setHeader("Content-Type", mimeTypeForExt(ext));
    res.setHeader("Content-Length", String(stat.size));

    // Evidence is authored elsewhere, some of it by third-party tooling, and
    // this route serves it inline from the app's own origin — the Vite dev
    // proxy puts /api on the same origin as the dashboard. An SVG opened as a
    // top-level document (FileViewerPage's "open it directly" link does exactly
    // that) would otherwise run any script inside it with full access to the
    // dashboard, and from there to every file the allowlist permits.
    //
    // `sandbox` with no allow-* tokens drops the response into an opaque origin
    // with scripting disabled. It does not affect <img> rendering, because
    // scripts in an SVG never run in image context anyway.
    res.setHeader("Content-Security-Policy", "sandbox");
    res.setHeader("X-Content-Type-Options", "nosniff");

    // Quote-strip the filename: it is attacker-influenceable on POSIX, where a
    // double quote is a legal filename character that would break the header.
    const safeName = path.basename(abs).replace(/["\\]/g, "");
    res.setHeader("Content-Disposition", `inline; filename="${safeName}"`);

    const stream = fs.createReadStream(abs);
    stream.on("error", (err) => res.destroy(err));
    stream.pipe(res);
  }),
);
