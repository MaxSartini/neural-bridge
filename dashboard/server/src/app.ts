import express, { type NextFunction, type Request, type Response } from "express";
import { healthRouter } from "./routes/health.js";
import { treeRouter } from "./routes/tree.js";
import { fileRouter } from "./routes/file.js";
import { rawRouter } from "./routes/raw.js";
import { HttpError } from "./lib/httpError.js";

/**
 * Hosts this server will answer to. Binding to 127.0.0.1 stops a remote machine
 * connecting directly, but it does not stop DNS rebinding: a page the user
 * visits can point an attacker-controlled name at 127.0.0.1 and then reach this
 * server *same-origin*, which sidesteps the absence of CORS entirely. Checking
 * Host is what closes that, and it is cheap.
 */
const ALLOWED_HOSTS = /^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$/;

export function createApp() {
  const app = express();
  app.disable("x-powered-by");

  app.use((req: Request, res: Response, next: NextFunction) => {
    if (!ALLOWED_HOSTS.test(req.headers.host ?? "")) {
      res.status(403).json({ error: "host not allowed" });
      return;
    }
    next();
  });

  const api = express.Router();
  api.use(healthRouter);
  api.use(treeRouter);
  api.use(fileRouter);
  api.use(rawRouter);
  app.use("/api", api);

  // Defense in depth: this is a read-only tool, no route ever writes. Reject
  // any non-GET method under /api explicitly rather than relying on routers
  // simply not defining one.
  app.all("/api/*", (req: Request, res: Response) => {
    res.status(405).json({ error: `method ${req.method} not allowed` });
  });

  app.use((req: Request, res: Response) => {
    res.status(404).json({ error: "not found" });
  });

  // `_next` is unused but required: Express identifies error middleware by
  // arity, so dropping it turns this into an ordinary handler that never runs.
  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof HttpError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
    console.error(err);
    res.status(500).json({ error: "internal error" });
  });

  return app;
}
