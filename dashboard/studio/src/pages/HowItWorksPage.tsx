import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Blueprint } from "@dashboard/ui";
import Reveal from "../components/Reveal";
import { howItWorks } from "../content/claims";

/**
 * The pipeline, one step per row, described by what each stage does rather
 * than what it runs on. The upstream models are deliberately unnamed — see the
 * disclosure line in `content/claims.ts`.
 */
export default function HowItWorksPage() {
  return (
    <div className="studio-column">
      <section className="hero">
        <div className="card-kicker">How it works</div>
        <h1 className="hero-title">Four steps, no audience data</h1>
        <p className="hero-technical studio-subtitle">
          You give us a cut. We give you back a map of where response is most likely to move.
          Nothing in between needs a survey, a panel, or a single viewer.
        </p>
      </section>

      <ol className="step-list">
        {howItWorks.map((s, i) => (
          <li key={s.step}>
            {/* A small stagger so the four steps land in order rather than
                arriving as one block — it reads as a sequence, which is what
                they are. */}
            <Reveal delay={i * 70}>
              <Blueprint className="step-row">
                <div className="step-number">{s.step}</div>
                <div className="step-body">
                  <div className="how-title">{s.title}</div>
                  <p className="how-body">{s.body}</p>
                </div>
              </Blueprint>
            </Reveal>
          </li>
        ))}
      </ol>

      <div className="hero-actions section-cta">
        <Blueprint as={Link} to="/analyze" className="btn btn-primary">
          Analyze a video
          <ArrowRight size={14} strokeWidth={1.5} aria-hidden="true" />
        </Blueprint>
        <Link to="/sample" className="btn btn-ghost">
          See what a report looks like
        </Link>
      </div>
    </div>
  );
}
