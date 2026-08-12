import fs from "node:fs";
import path from "node:path";
import { createApp } from "./app.js";
import { REPO_ROOT, PORT } from "./config.js";

// Cheap sanity check that we're pointed at the right repo before serving anything.
if (!fs.existsSync(path.join(REPO_ROOT, "pyproject.toml"))) {
  console.error(`REPO_ROOT does not look like the neural-bridge repo: ${REPO_ROOT}`);
  process.exit(1);
}

const app = createApp();

app.listen(PORT, "127.0.0.1", () => {
  console.log(`dashboard server listening on http://127.0.0.1:${PORT}`);
  console.log(`serving from repo root: ${REPO_ROOT}`);
});
