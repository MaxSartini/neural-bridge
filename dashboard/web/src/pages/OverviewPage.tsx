import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Blueprint, Tag } from "@dashboard/ui";
import EvidenceLink from "../components/EvidenceLink";
import LandmarkCard from "../components/LandmarkCard";
import { datasets, landmarks } from "../data/scorecard";
import {
  datasetPlain,
  datasetTechnicalLine,
  heroGloss,
  landmarkPlain,
  landmarkTechnicalLine,
} from "../data/plainLanguage";

/**
 * Where the evidence lives. The design's mock carried an "Updated" column
 * reading "Today" / "This week"; it is dropped here rather than invented.
 * Freshness we cannot actually measure is exactly the false precision the
 * programme's README refuses elsewhere, and the server does not report a
 * modification time for a synthesized section.
 */
const EVIDENCE_SECTIONS = [
  {
    label: "Results",
    to: "/doc/results/README.md",
    contains: "The concluded scorecard, with exact values",
  },
  {
    label: "Studies",
    to: "/doc/studies/README.md",
    contains: "The phase-by-phase record, including what failed",
  },
  {
    label: "Docs",
    to: "/doc/docs/README.md",
    contains: "Methods, controls, and how to reproduce a result",
  },
  { label: "Browse", to: "/browse", contains: "Every evidence file, as a tree" },
];

export default function OverviewPage() {
  return (
    <div className="overview">
      <section className="hero">
        <div className="card-kicker">Evidence Dashboard</div>
        <h1 className="hero-title">{heroGloss.plain}</h1>
        <p className="hero-technical">{heroGloss.technical}</p>
        <div className="hero-actions">
          {/* "Start new analysis" pointed at the Studio, which is a separate
              deployment now — C7 restores it as an external link. */}
          <Blueprint as={Link} to="/doc/docs/README.md" className="btn btn-primary">
            Read the methodology
            <ArrowRight size={14} strokeWidth={1.5} aria-hidden="true" />
          </Blueprint>
          <Link to="/doc/results/README.md" className="btn btn-ghost">
            See the results table
          </Link>
        </div>
      </section>

      <section className="overview-section">
        <div className="section-head">
          <span className="card-kicker">Programme</span>
          <h3>Four landmark results</h3>
        </div>
        <div className="landmark-grid">
          {landmarks.map((l) => (
            <LandmarkCard
              key={l.id}
              kicker={`${l.romanNumeral} · ${l.metricLabel}`}
              plain={landmarkPlain[l.id]}
              value={l.baselineGainPct}
              technical={landmarkTechnicalLine(l)}
              sourceDoc={l.sourceDoc}
            />
          ))}
        </div>
      </section>

      <section className="overview-section">
        <div className="section-head">
          <span className="card-kicker">Datasets</span>
          <h3>What the bridge learned from</h3>
        </div>
        <div className="dataset-grid">
          {datasets.map((d) => (
            <Blueprint
              as={EvidenceLink}
              key={d.id}
              sourceDoc={d.sourceDoc}
              className="dataset-card"
            >
              <div className="card-kicker">{d.category}</div>
              <div className="dataset-headline">
                <span className="dataset-stat">{d.stat}</span>
                <span className="dataset-title">{d.title}</span>
              </div>
              <div className="dataset-plain">{datasetPlain[d.id]}</div>
              <div className="landmark-technical">{datasetTechnicalLine(d)}</div>
            </Blueprint>
          ))}
        </div>
      </section>

      <section className="overview-section">
        <div className="section-head">
          <span className="card-kicker">Explore</span>
          <h3>Jump into the evidence</h3>
        </div>
        <table className="table evidence-table">
          <thead>
            <tr>
              <th>Section</th>
              <th>Contains</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {EVIDENCE_SECTIONS.map((s) => (
              <tr key={s.to}>
                <td>
                  <Link to={s.to}>{s.label}</Link>
                </td>
                <td className="text-muted">{s.contains}</td>
                <td>
                  <Tag variant="accent">Live</Tag>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
