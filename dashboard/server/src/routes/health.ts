import { Router } from "express";

export const healthRouter = Router();

/**
 * Liveness only. This used to return `repoRoot`, which handed out an absolute
 * path — username and directory layout included — from the one endpoint most
 * likely to be pasted into a bug report or a screenshot. Nothing consumed it.
 */
healthRouter.get("/health", (_req, res) => {
  res.json({ ok: true });
});
