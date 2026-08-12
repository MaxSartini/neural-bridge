import type { ReactNode } from "react";
import { Blueprint, Tag } from "@dashboard/ui";
import EvidenceLink from "../components/EvidenceLink";
import LandmarkCard from "../components/LandmarkCard";
import GainBarChart from "../components/charts/GainBarChart";
import ProgressionBarChart from "../components/charts/ProgressionBarChart";
import {
  landmarks,
  againBridgeProgression,
  againBridgeProgressionSource,
  rawFeatureFailure,
  rawFeatureFailureSource,
} from "../data/scorecard";
import { chartGlosses, landmarkPlain, landmarkTechnicalLine } from "../data/plainLanguage";

const prAuc = (v: number) => v.toFixed(4);

interface ChartCardProps {
  num: string;
  glossKey: keyof typeof chartGlosses;
  badge: string;
  children: ReactNode;
}

/**
 * A chart, framed. The plain gloss is the card's title and the precise wording
 * becomes the subtitle beneath it — the same order every other surface uses.
 */
function ChartCard({ num, glossKey, badge, children }: ChartCardProps) {
  const gloss = chartGlosses[glossKey];
  return (
    <Blueprint className="chart-card">
      <div className="section-head">
        <span className="card-kicker">{num}</span>
        <h3>{gloss.plain}</h3>
        <Tag variant="outline">{badge}</Tag>
      </div>
      <p className="chart-subtitle">{gloss.technical}</p>
      {children}
    </Blueprint>
  );
}

export default function ScorecardPage() {
  return (
    <div>
      <section className="hero">
        <div className="card-kicker">Scorecard</div>
        <h1 className="hero-title">Every result, side by side</h1>
        <p className="hero-technical">
          Neural Bridge against frozen AR and the strongest matched control, across the
          programme&apos;s four landmark wins, plus the AGAIN event-bridge development story.
          Every card and chart links back to its source evidence — start from{" "}
          <EvidenceLink sourceDoc="results/README.md">results/README.md</EvidenceLink> or the{" "}
          <EvidenceLink sourceDoc="README.md">full write-up</EvidenceLink>.
        </p>
      </section>

      <div className="landmark-grid landmark-grid-compact">
        {landmarks.map((l) => (
          <LandmarkCard
            key={l.id}
            variant="value-first"
            kicker={`${l.romanNumeral} · ${l.metricLabel}`}
            plain={landmarkPlain[l.id]}
            value={l.baselineGainPct}
            technical={landmarkTechnicalLine(l)}
            sourceDoc={l.sourceDoc}
          />
        ))}
      </div>

      <ChartCard num="01" glossKey="gain" badge="4 landmarks">
        <GainBarChart data={landmarks} />
      </ChartCard>

      <ChartCard num="02" glossKey="progression" badge="6 stages">
        <ProgressionBarChart
          data={againBridgeProgression}
          sourceDoc={againBridgeProgressionSource}
          seriesName="PR-AUC"
          valueFormat={prAuc}
        />
      </ChartCard>

      <ChartCard num="03" glossKey="failure" badge="motivating failure">
        <ProgressionBarChart
          data={rawFeatureFailure}
          sourceDoc={rawFeatureFailureSource}
          seriesName="PR-AUC"
          valueFormat={prAuc}
          baselineValue={rawFeatureFailure[rawFeatureFailure.length - 1].value}
          baselineLabel="AR baseline"
        />
      </ChartCard>
    </div>
  );
}
