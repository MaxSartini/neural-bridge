import { Link } from "react-router-dom";

/**
 * Every unknown path lands here rather than on a blank page, because the SPA
 * rewrite means Cloudflare serves index.html for *anything* — a typo in a
 * shared link reaches the app, not a 404 from the edge.
 */
export default function NotFoundPage() {
  return (
    <div className="studio-column studio-column-narrow">
      <section className="hero">
        <div className="card-kicker">Not found</div>
        <h1 className="hero-title">There&apos;s nothing at this address</h1>
        <p className="hero-technical studio-subtitle">
          The link may be mistyped, or it may point at an analysis that has finished its
          session. Runs live in the browser tab that started them.
        </p>
        <div className="hero-actions">
          <Link to="/" className="btn btn-secondary">
            Back to the start
          </Link>
          <Link to="/sample" className="btn btn-ghost">
            See a sample report
          </Link>
        </div>
      </section>
    </div>
  );
}
