#!/usr/bin/env node
/**
 * Source-level package boundary check.
 *
 * ## Why this exists, and what it corrects
 *
 * ADR-0006 was first drafted claiming that moving the Studio into its own npm
 * workspace made internal imports "unresolvable by construction". That is
 * **false**, and a probe proved it before the claim shipped. Two holes:
 *
 * 1. **npm hoists every workspace into the root `node_modules`**, whether or
 *    not you depend on it. `import "dashboard-web/src/data/scorecard"` resolves
 *    from inside `studio/` even though `dashboard-web` is not in its
 *    `dependencies`.
 * 2. **Relative paths escape upward.** `../../web/src/data/scorecard` is a
 *    perfectly ordinary path that happens to leave the package.
 *
 * A package boundary is a convention npm does not enforce. So this does.
 *
 * The model is the one the repo already trusts for the evidence server: an
 * **allowlist**, checked before anything else happens. A specifier is legal
 * only if it is relative and stays inside the package, or bare and declared in
 * this package's own `dependencies`. Everything else fails the build, which is
 * how the class of regression that leaked the programme's internals to a
 * customer-facing bundle stops being possible rather than merely discouraged.
 *
 * Usage: node scripts/verify-boundary.mjs <package-dir>
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, extname, resolve, relative, dirname, sep } from "node:path";
import { builtinModules } from "node:module";

const [, , pkgDirArg] = process.argv;
if (!pkgDirArg) {
  console.error("usage: verify-boundary.mjs <package-dir>");
  process.exit(2);
}

const pkgRoot = resolve(pkgDirArg);
const pkg = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8"));

const allowedBare = new Set([
  ...Object.keys(pkg.dependencies ?? {}),
  ...Object.keys(pkg.peerDependencies ?? {}),
  ...Object.keys(pkg.devDependencies ?? {}),
  ...builtinModules,
]);

/** `react-dom/client` is a subpath of an allowed package; take the package name. */
function packageNameOf(spec) {
  const parts = spec.split("/");
  return spec.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
}

function sourceFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...sourceFiles(p));
    else if ([".ts", ".tsx", ".js", ".jsx", ".css"].includes(extname(p))) out.push(p);
  }
  return out;
}

// Covers `import x from "y"`, `import "y"`, `export * from "y"`,
// `import("y")`, and CSS `@import "y"`.
const SPECIFIER = /(?:\bfrom\s*|\bimport\s*\(?\s*|@import\s+(?:url\()?)["']([^"']+)["']/g;

const violations = [];

for (const file of sourceFiles(join(pkgRoot, "src"))) {
  const text = readFileSync(file, "utf8");
  for (const m of text.matchAll(SPECIFIER)) {
    const spec = m[1];
    if (spec.startsWith(".")) {
      const target = resolve(dirname(file), spec);
      const rel = relative(pkgRoot, target);
      if (rel.startsWith("..") || rel.startsWith(`..${sep}`)) {
        violations.push({ file, spec, why: "relative path escapes the package" });
      }
    } else if (!spec.startsWith("\0") && !spec.startsWith("data:")) {
      const name = packageNameOf(spec);
      if (!allowedBare.has(name)) {
        violations.push({
          file,
          spec,
          why: `"${name}" is not a declared dependency (npm hoists every workspace, so this would still resolve)`,
        });
      }
    }
  }
}

if (violations.length > 0) {
  console.error(`\n  PACKAGE BOUNDARY VIOLATION — ${pkg.name}\n`);
  for (const v of violations) {
    console.error(`    ${relative(pkgRoot, v.file)}`);
    console.error(`      imports ${JSON.stringify(v.spec)}`);
    console.error(`      ${v.why}\n`);
  }
  console.error(
    `  This package is published to people outside the programme. It may only\n` +
      `  import its own source and its declared dependencies. If you need\n` +
      `  something from the internal packages, copy the value into\n` +
      `  src/content/ and phrase it for an external reader — see ADR-0006.\n`,
  );
  process.exit(1);
}

console.log(`  package boundary: ${pkg.name} clean — no import leaves the package`);
