/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute URL of the deployed Studio, or undefined. See lib/studioLink.ts. */
  readonly VITE_STUDIO_URL?: string;
  /** "github" in the standalone scorecard bundle; unset in the local dashboard. */
  readonly VITE_EVIDENCE_LINKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
