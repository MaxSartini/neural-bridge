import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Blueprint } from "@dashboard/ui";
import ImageSlot from "../components/ImageSlot";
import Reveal from "../components/Reveal";
import { FrameStripArt, RegistrationFieldArt } from "../components/Artwork";
import { heroClaim, validatedClaims, whatYouGet } from "../content/claims";

/**
 * The Studio's front door.
 *
 * Every claim comes from `content/claims.ts`, which holds the disclosure line.
 * This page previously imported the programme's landmark records directly,
 * which put metric names, control names, raw values and internal study paths
 * into the bundle shipped to customers — see ADR-0006. Nothing here reaches
 * outside this package, and `verify-boundary.mjs` fails the build if it tries.
 *
 * Home stays short on purpose: one claim, what you get back, proof, and two
 * ways in. The detail lives on /how-it-works and /results so a visitor can go
 * as deep as they want rather than scrolling past everything.
 */
export default function HomePage() {
  return (
    <div className="studio-column">
      <section className="hero hero-with-texture">
        <RegistrationFieldArt className="hero-texture" />
        <div className="card-kicker">Neural Bridge Studio</div>
        <h1 className="hero-title">{heroClaim.plain}</h1>
        <p className="hero-technical studio-subtitle">{heroClaim.supporting}</p>
        <div className="hero-actions">
          <Blueprint as={Link} to="/analyze" className="btn btn-primary">
            Analyze a video
            <ArrowRight size={14} strokeWidth={1.5} aria-hidden="true" />
          </Blueprint>
          <Link to="/sample" className="btn btn-ghost">
            See a sample report
          </Link>
        </div>
      </section>

      <Reveal>
        <ImageSlot
          className="hero-figure"
          ratio="12 / 5"
          fallback={<FrameStripArt />}
          /* Drop a photograph into public/media/ and pass src here — no other
             change needed. See public/media/README.md. */
        />
      </Reveal>

      <Reveal>
        <section className="overview-section">
          <div className="section-head">
            <span className="card-kicker">What you get back</span>
            <h3>A response map for the cut you upload</h3>
          </div>
          <div className="claim-grid claim-grid-3">
            {whatYouGet.map((item) => (
              <Blueprint key={item.id} className="how-card">
                <div className="how-title">{item.title}</div>
                <p className="how-body">{item.body}</p>
              </Blueprint>
            ))}
          </div>
        </section>
      </Reveal>

      <Reveal>
        <section className="overview-section">
          <div className="section-head">
            <span className="card-kicker">Proof</span>
            <h3>Measured, not asserted</h3>
            <Link to="/results" className="section-link">
              All results
            </Link>
          </div>
          <div className="claim-grid claim-grid-2">
            {validatedClaims.slice(0, 2).map((c) => (
              <Blueprint key={c.id} className="claim-card">
                <div className="claim-plain">{c.plain}</div>
                <div className="claim-figure">{c.figure}</div>
                <div className="claim-context">{c.context}</div>
              </Blueprint>
            ))}
          </div>
        </section>
      </Reveal>
    </div>
  );
}
