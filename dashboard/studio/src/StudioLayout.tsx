import { Link, NavLink, Outlet } from "react-router-dom";
import { Blueprint, ThemeToggle } from "@dashboard/ui";

/**
 * The client-facing shell. Same green band as the evidence dashboard, different
 * identity: this is the product a customer uses, not the programme's evidence.
 * The two share tokens and primitives but not navigation — nothing here routes
 * into the internal evidence tree, and nothing can.
 *
 * The design's band also carried "Docs" and "Help". Both stay out until they
 * lead somewhere: a dead link is worse than a missing one, especially on the
 * surface a prospective customer sees first.
 */
export default function StudioLayout() {
  return (
    <div className="app-shell">
      <header className="nav nav-inverse">
        <Link to="/" className="nav-brand nav-brand-link">
          Neural Bridge Studio
        </Link>
        <nav className="nav-links">
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/how-it-works">How it works</NavLink>
          <NavLink to="/results">Results</NavLink>
          <NavLink to="/sample">Sample report</NavLink>
        </nav>
        <ThemeToggle />
        <Blueprint as={Link} to="/analyze" className="btn btn-primary nav-cta">
          Analyze video
        </Blueprint>
      </header>
      <main className="app-main studio-main">
        <Outlet />
      </main>
    </div>
  );
}
