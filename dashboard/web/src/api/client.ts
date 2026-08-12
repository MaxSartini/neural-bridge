import type { FileKind, TextFileKind } from "@dashboard/shared";

export type { FileKind };

export interface TreeEntry {
  name: string;
  path: string;
  /** Directory or file. Distinct from `kind`, which classifies the contents. */
  entryType: "dir" | "file";
  ext: string;
  /** Viewer classification for files; null for directories and unknown extensions. */
  kind: FileKind | null;
  size: number;
  mtimeMs: number;
}

export interface TreeResponse {
  path: string;
  entries: TreeEntry[];
}

export interface FileResponse {
  path: string;
  ext: string;
  /** /api/file inlines text only, so an image kind never appears here. */
  kind: TextFileKind;
  size: number;
  truncated: boolean;
  mtimeMs: number;
  content: string;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new ApiError(res.status, body.error ?? res.statusText);
  }
  return (await res.json()) as T;
}

export function fetchTree(path: string): Promise<TreeResponse> {
  return getJson<TreeResponse>(`/api/tree?path=${encodeURIComponent(path)}`);
}

export function fetchFile(path: string): Promise<FileResponse> {
  return getJson<FileResponse>(`/api/file?path=${encodeURIComponent(path)}`);
}

export function rawUrl(path: string): string {
  return `/api/raw?path=${encodeURIComponent(path)}`;
}
