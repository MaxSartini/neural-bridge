/**
 * The one owner of "what kind of file is this".
 *
 * This decision used to live in three places — `server/src/lib/fileType.ts`,
 * `web/src/lib/linkRewrite.ts` and `web/src/pages/FileViewerPage.tsx` — with the
 * image extension set written out verbatim twice and `.pdf` known to the server
 * alone. Adding an extension meant remembering all three.
 *
 * This module has no dependencies, not even `node:path`, so both the Express
 * server and the browser bundle can import it. It is the only reason the
 * `shared` workspace exists.
 */

/** Kinds `/api/file` will inline. Parsing happens client-side; this only labels. */
export type TextFileKind = "markdown" | "json" | "jsonl" | "csv" | "code" | "text";

/** Every kind the dashboard can display, including the ones streamed as bytes. */
export type FileKind = TextFileKind | "image";

const KIND_BY_EXT: Record<string, FileKind> = {
  ".md": "markdown",
  ".json": "json",
  ".jsonl": "jsonl",
  ".csv": "csv",
  ".py": "code",
  ".tex": "code",
  ".mmd": "code",
  ".txt": "text",
  ".svg": "image",
  ".png": "image",
  ".jpg": "image",
  ".jpeg": "image",
  ".gif": "image",
};

/**
 * Content-Type for `/api/raw` byte streaming.
 *
 * Deliberately a separate table from KIND_BY_EXT rather than a field on it:
 * `.pdf` has a real MIME type but no viewer, so it appears here and not there.
 * That asymmetry used to be invisible — it was the `.pdf` divergence between the
 * server's table and the two frontend copies. Now it is one line you can see.
 *
 * `.py` / `.tex` / `.mmd` are intentionally absent, so they keep falling back to
 * `application/octet-stream` exactly as before.
 */
const MIME_BY_EXT: Record<string, string> = {
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".csv": "text/csv",
  ".md": "text/markdown",
  ".txt": "text/plain",
  ".pdf": "application/pdf",
};

/**
 * The extension of a path, including the leading dot, or `""`.
 *
 * Matches `node:path`'s `extname` semantics — notably `".gitignore"` has *no*
 * extension — so the server keeps behaving identically after switching to this,
 * and the browser gets the same answer without pulling in a path polyfill.
 * Accepts either separator.
 */
export function extname(pathOrName: string): string {
  const name = pathOrName.replace(/\\/g, "/").split("/").pop() ?? "";
  if (/^\.+$/.test(name)) return "";
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot) : "";
}

/** The viewer classification for an extension, or `null` if nothing displays it. */
export function fileKindForExt(ext: string): FileKind | null {
  return KIND_BY_EXT[ext.toLowerCase()] ?? null;
}

/** As `fileKindForExt`, from a whole path. */
export function fileKindForPath(pathOrName: string): FileKind | null {
  return fileKindForExt(extname(pathOrName));
}

/**
 * The kind for `/api/file`, which inlines text only. Images are served as bytes
 * by `/api/raw`, so they are not a text kind and this returns `null` for them.
 */
export function textKindForExt(ext: string): TextFileKind | null {
  const kind = fileKindForExt(ext);
  return kind === null || kind === "image" ? null : kind;
}

/** Content-Type for `/api/raw`, falling back to a generic binary type. */
export function mimeTypeForExt(ext: string): string {
  return MIME_BY_EXT[ext.toLowerCase()] ?? "application/octet-stream";
}

/** True when the extension names something the browser renders as an image. */
export function isImageExt(ext: string): boolean {
  return fileKindForExt(ext) === "image";
}
