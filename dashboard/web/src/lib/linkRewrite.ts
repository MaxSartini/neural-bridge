import { extname, fileKindForExt } from "@dashboard/shared";

export type LinkTarget =
  | { kind: "external"; href: string }
  | { kind: "doc"; path: string }
  | { kind: "raw"; path: string }
  | { kind: "view"; path: string };

function posixDirname(p: string): string {
  const idx = p.lastIndexOf("/");
  return idx === -1 ? "" : p.slice(0, idx);
}

function posixNormalize(p: string): string {
  const out: string[] = [];
  for (const part of p.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      out.pop();
      continue;
    }
    out.push(part);
  }
  return out.join("/");
}

function posixJoin(dir: string, href: string): string {
  if (dir === "") return posixNormalize(href);
  return posixNormalize(`${dir}/${href}`);
}

/**
 * Resolves a link found inside a rendered Markdown doc (repo-relative, possibly with
 * "../" segments) against the doc's own repo-relative path, into a target this app
 * knows how to navigate to. Used by Markdown.tsx's custom link/image renderers.
 */
export function resolveLink(currentDocPath: string, hrefRaw: string): LinkTarget {
  const href = hrefRaw.trim();
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(href) || href.startsWith("mailto:") || href.startsWith("#")) {
    return { kind: "external", href };
  }

  const [pathPart] = href.split("#");
  if (!pathPart) {
    return { kind: "external", href };
  }

  const dir = posixDirname(currentDocPath.replace(/\\/g, "/"));
  const resolved = posixJoin(dir, pathPart);

  const ext = extname(resolved);

  if (resolved.endsWith("/") || ext === "") {
    const dirPath = resolved.endsWith("/") ? resolved.slice(0, -1) : resolved;
    return { kind: "doc", path: dirPath === "" ? "README.md" : `${dirPath}/README.md` };
  }

  const kind = fileKindForExt(ext);
  if (kind === "markdown") {
    return { kind: "doc", path: resolved };
  }
  if (kind === "image") {
    return { kind: "raw", path: resolved };
  }
  return { kind: "view", path: resolved };
}
