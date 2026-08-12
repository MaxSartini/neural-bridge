/**
 * @dashboard/ui — the Industry design language as code.
 *
 * Everything here is presentational and knows nothing about the programme: no
 * evidence data, no API client, no routing. That is what lets both the internal
 * evidence dashboard and the external Studio depend on it without the Studio
 * gaining a path to internal material.
 *
 * Raw TS/TSX with no build step, following `@dashboard/shared`. Vite resolves
 * the workspace symlink to real source and transforms it normally — verified
 * against a production build, not assumed.
 */
export { default as Blueprint } from "./Blueprint";
export { default as ProgressBar } from "./ProgressBar";
export { default as Skeleton } from "./Skeleton";
export { default as Tag } from "./Tag";
export type { TagVariant } from "./Tag";
export { default as ThemeToggle } from "./ThemeToggle";
