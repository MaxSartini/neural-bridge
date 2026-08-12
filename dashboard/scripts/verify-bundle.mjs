#!/usr/bin/env node
/**
 * Bundle audit. Fails the build if a published artifact contains something it
 * must not.
 *
 * ## Why this exists
 *
 * The Studio bundle shipped internal programme material to a customer-facing
 * deployment: upstream model names as visible UI text, metric and control
 * vocabulary, raw metric values, and internal study paths. Every one of those
 * came from a single import plus a handful of string literals, and every
 * document in the repo already asserted the separation that import broke.
 *
 * Prose did not catch it. A grep run by hand would not have either, because
 * DEPLOY.md's table only checked for network symbols. So the check runs in the
 * build.
 *
 * ## What it cannot do
 *
 * It greps a minified bundle. It catches vocabulary and known constants, not
 * paraphrase — someone can still describe the architecture in their own words
 * and this will pass. It is a backstop for regressions, not a substitute for
 * reading what you publish.
 *
 * Usage: node scripts/verify-bundle.mjs <dist-dir> [--profile external|internal]
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, extname } from "node:path";

/** Literal substrings. Case-sensitive unless listed in CASE_INSENSITIVE. */
const INTERNAL_VOCABULARY = [
  // Upstream models — the most valuable thing to withhold.
  "V-JEPA",
  "TRIBE",
  // Metric names.
  "PR-AUC",
  "Spearman",
  // Baseline and control design.
  "frozen AR",
  "Frozen AR",
  "Trained AR",
  "trained AR",
  "Shuffled features",
  "false-signal",
  "no-video control",
  "matched control",
  "positive groups",
  "held-out",
  // Substrate.
  "cortical",
  "annotators",
  // Internal document structure.
  "studies/",
  "results/README",
  "phase-0",
  "zero-label",
  "evidence-summary",
];

/**
 * Dataset names are ordinary English words, so they need word boundaries and
 * case sensitivity. "again" appears in normal prose; "AGAIN" is the corpus.
 */
const INTERNAL_WORDS = ["VEATIC", "AGAIN"];

/**
 * Raw metric values from the transcribed record. Written without the leading
 * zero because that is how a minifier emits them — checking for "0.2536" gives
 * a false pass, which is exactly the mistake that hid these the first time.
 */
const INTERNAL_VALUES = [
  ".2536",
  ".1969",
  ".178513",
  ".100488",
  ".234368",
  ".21805",
  ".217972",
  ".260301",
  ".240537",
  ".136579",
  ".238341",
  ".203622",
];

/** Both published bundles are static and must reach nothing at runtime. */
const NETWORK_PRIMITIVES = [
  "fetch(",
  "XMLHttpRequest",
  "WebSocket",
  "EventSource",
  "sendBeacon",
];

/** The internal API surface, absent from every published bundle. */
const SERVER_SURFACE = [
  "/api/tree",
  "/api/file",
  "/api/raw",
  "/api/health",
  "fetchTree",
  "fetchFile",
  "rawUrl",
  "safePath",
];

function collectFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...collectFiles(p));
    else if ([".js", ".css", ".html"].includes(extname(p))) out.push(p);
  }
  return out;
}

const [, , distDir, ...rest] = process.argv;
if (!distDir) {
  console.error("usage: verify-bundle.mjs <dist-dir> [--profile external|internal]");
  process.exit(2);
}
const profileArg = rest.indexOf("--profile");
const profile = profileArg === -1 ? "external" : rest[profileArg + 1];

const files = collectFiles(distDir);
if (files.length === 0) {
  console.error(`verify-bundle: no .js/.css/.html found under ${distDir} — did the build run?`);
  process.exit(2);
}

const corpus = files.map((f) => ({ file: f, text: readFileSync(f, "utf8") }));
const failures = [];

function checkLiteral(term, label) {
  for (const { file, text } of corpus) {
    let n = 0;
    let i = text.indexOf(term);
    while (i !== -1) {
      n += 1;
      i = text.indexOf(term, i + term.length);
    }
    if (n > 0) failures.push({ term, label, file, count: n });
  }
}

function checkWord(term, label) {
  const re = new RegExp(`\\b${term}\\b`, "g");
  for (const { file, text } of corpus) {
    const n = (text.match(re) ?? []).length;
    if (n > 0) failures.push({ term, label, file, count: n });
  }
}

// The server surface and network primitives are forbidden in every bundle.
for (const t of SERVER_SURFACE) checkLiteral(t, "internal API surface");
for (const t of NETWORK_PRIMITIVES) checkLiteral(t, "runtime network call");

// Programme vocabulary is forbidden only in the external bundle. The internal
// scorecard is supposed to say "PR-AUC" — that is its whole job.
if (profile === "external") {
  for (const t of INTERNAL_VOCABULARY) checkLiteral(t, "internal vocabulary");
  for (const t of INTERNAL_WORDS) checkWord(t, "dataset name");
  for (const t of INTERNAL_VALUES) checkLiteral(t, "raw metric value");
}

if (failures.length > 0) {
  console.error(`\n  BUNDLE AUDIT FAILED — ${distDir} (profile: ${profile})\n`);
  for (const f of failures) {
    console.error(`    ${f.label.padEnd(22)} ${JSON.stringify(f.term).padEnd(22)} x${f.count}  ${f.file}`);
  }
  console.error(
    `\n  This artifact is published to people outside the programme.\n` +
      `  Move the claim into studio/src/content/claims.ts and phrase it without\n` +
      `  method vocabulary, or remove it. Do not add an exception here.\n`,
  );
  process.exit(1);
}

const checked =
  profile === "external"
    ? SERVER_SURFACE.length +
      NETWORK_PRIMITIVES.length +
      INTERNAL_VOCABULARY.length +
      INTERNAL_WORDS.length +
      INTERNAL_VALUES.length
    : SERVER_SURFACE.length + NETWORK_PRIMITIVES.length;
console.log(
  `  bundle audit: ${distDir} clean — ${checked} terms checked across ${files.length} files (profile: ${profile})`,
);
