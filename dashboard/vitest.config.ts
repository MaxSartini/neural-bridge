import { defineConfig } from "vitest/config";

/**
 * One test run for both workspaces.
 *
 * The traversal guard lives in `server/`, the link/provenance logic lives in
 * `web/`, and each is tested where it lives — no cross-workspace imports, which
 * the server's NodeNext resolution would fight anyway.
 *
 * Both projects run in `node`. Every module under test is pure logic (string
 * and path manipulation), so there is no DOM to emulate and jsdom is not a
 * dependency. If a component test ever lands, give the `web` project its own
 * `environment: "jsdom"` then, not before.
 */
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "server",
          root: "./server",
          environment: "node",
          include: ["src/**/*.test.ts"],
        },
      },
      {
        test: {
          name: "web",
          root: "./web",
          environment: "node",
          include: ["src/**/*.test.ts"],
        },
      },
      {
        test: {
          name: "studio",
          root: "./studio",
          environment: "node",
          include: ["src/**/*.test.ts"],
        },
      },
    ],
  },
});
