import { MemoryRouter, NavLink, Route, Routes } from "react-router-dom";
import ScorecardPage from "../pages/ScorecardPage";
import ScrollyStory from "../scrolly/ScrollyStory";
import { ThemeToggle } from "@dashboard/ui";

/**
 * Shell for the standalone bundle.
 *
 * No API client — this build never talks to a server, so the file browser (and
 * with it the whole path-guard surface) is absent from the bundle rather than
 * merely unreachable. Provenance links resolve to the private GitHub repo via
 * `lib/evidenceLink`.
 *
 * MemoryRouter, not BrowserRouter: navigation stays in memory so the two views
 * work from a static host without needing an SPA rewrite rule. (Phase A5 adds
 * that rule for the full dashboard, which does need real URLs.)
 */
export default function StaticScorecard() {
  return (
    <MemoryRouter>
      <div className="app-shell">
        {/* Same green band as the dashboard shell, minus the "New analysis"
            call to action: this bundle has no Studio to route to. */}
        <header className="nav nav-inverse">
          <span className="nav-brand">Neural Bridge</span>
          <nav className="nav-links">
            <NavLink to="/" end>
              Scorecard
            </NavLink>
            <NavLink to="/story">Story</NavLink>
          </nav>
          <ThemeToggle />
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<ScorecardPage />} />
            <Route path="/story" element={<ScrollyStory />} />
          </Routes>
          <p className="static-footnote">
            Evidence links open the private{" "}
            <a
              href="https://github.com/MaxSartini/neural-bridge"
              target="_blank"
              rel="noopener noreferrer"
            >
              neural-bridge
            </a>{" "}
            repository and require access to it.
          </p>
        </main>
      </div>
    </MemoryRouter>
  );
}
