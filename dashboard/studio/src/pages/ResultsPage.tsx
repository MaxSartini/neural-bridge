import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Blueprint } from "@dashboard/ui";
import ImageSlot from "../components/ImageSlot";
import Reveal from "../components/Reveal";
import { ResponseMapArt } from "../components/Artwork";
import {
  boundariesIntro,
  boundariesOutro,
  notClaimed,
  validatedClaims,
} from "../content/claims";

/**
 * The credibility page.
 *
 * Every figure here is real and every framing is method-free: what improved,
 * by how much, against what it was compared to in plain words. No metric
 * names, no dataset names, no control vocabulary — see the disclosure line in
 * `content/claims.ts`.
 *
 * The boundaries block is not a disclaimer tucked at the bottom. It sits on
 * the same page as the numbers because it is what makes them worth believing,
 * and it mirrors the "Honest boundaries" section of the programme's own README.
 */
export default function ResultsPage() {
  return (
    <div className="studio-column">
      <section className="hero">
        <div className="card-kicker">Results</div>
        <h1 className="hero-title">What has actually been measured</h1>
        <p className="hero-technical studio-subtitle">
          Each figure below compares Neural Bridge against the strongest alternative we could
          build, on video the system had never seen. Nothing here is a projection.
        </p>
      </section>

      <Reveal>
        <ImageSlot className="hero-figure" ratio="16 / 5" fallback={<ResponseMapArt />} />
      </Reveal>

      <div className="claim-grid claim-grid-2 results-grid">
        {validatedClaims.map((c, i) => (
          <Reveal key={c.id} delay={i * 60}>
            <Blueprint className="claim-card claim-card-large">
              <div className="claim-plain">{c.plain}</div>
              <div className="claim-figure claim-figure-large">{c.figure}</div>
              <div className="claim-context">{c.context}</div>
            </Blueprint>
          </Reveal>
        ))}
      </div>

      <Reveal>
        <Blueprint className="chart-card confidence-card">
          <div className="card-kicker">What we do not claim</div>
          <h3 className="confidence-title">The boundaries are part of the product</h3>
          <p className="confidence-body">{boundariesIntro}</p>
          <ul className="boundaries-list">
            {notClaimed.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p className="confidence-body">{boundariesOutro}</p>
        </Blueprint>
      </Reveal>

      <div className="hero-actions section-cta">
        <Blueprint as={Link} to="/analyze" className="btn btn-primary">
          Analyze a video
          <ArrowRight size={14} strokeWidth={1.5} aria-hidden="true" />
        </Blueprint>
        <Link to="/sample" className="btn btn-ghost">
          See a sample report
        </Link>
      </div>
    </div>
  );
}
