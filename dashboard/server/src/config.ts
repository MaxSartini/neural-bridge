import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// dashboard/server/src -> dashboard/server -> dashboard -> repo root
export const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

export const ALLOWED_TOP_LEVEL = new Set(["README.md", "results", "studies", "docs"]);

export const PORT = 4319;

export const MAX_INLINE_BYTES = 5 * 1024 * 1024; // 5MB
