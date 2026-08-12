import { NavLink, Outlet } from "react-router-dom";
import { Blueprint, ThemeToggle } from "@dashboard/ui";
import { studioUrl } from "../lib/studioLink";

/**
 * "Results" is deliberately absent: the design routes readers to it from the
 * Overview page's evidence table rather than the band, which keeps the nav to
 * six items at the width the design was drawn for.
 */
const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/scorecard", label: "Scorecard", end: false },
  { to: "/story", label: "Story", end: false },
  { to: "/doc/docs/README.md", label: "Docs", end: false },
  { to: "/doc/studies/README.md", label: "Studies", end: false },
  { to: "/browse", label: "Browse", end: false },
];

export default function Layout() {
  return (
    <div className="app-shell">
      {/* The sliding pill from Phase A3 is gone. The design carries the current
          route with the green field plus paper text, so the pill's measured
          offsets — a useLayoutEffect and a ResizeObserver — had nothing left to
          express. NavLink's own `active` class and aria-current do the work. */}
      <header className="nav nav-inverse">
        <span className="nav-brand">Neural Bridge</span>
        <nav className="nav-links">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <ThemeToggle />
        {/* The Studio is a separate deployment (ADR-0006), so this needs an
            absolute URL. It renders only when VITE_STUDIO_URL is set, because a
            button that 404s is worse than no button. */}
        {studioUrl && (
          <Blueprint
            as="a"
            href={studioUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary nav-cta"
          >
            New analysis
          </Blueprint>
        )}
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
