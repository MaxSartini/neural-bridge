import { Blueprint } from "@dashboard/ui";
import EvidenceLink from "./EvidenceLink";
import { useCountUp } from "../hooks/useCountUp";

interface Props {
  /** Roman numeral and primary metric, e.g. "I · Blocked event PR-AUC". */
  kicker: string;
  /** The plain gloss. Leads the card, or sits under the numeral. */
  plain: string;
  /** Raw number so it can animate — formatting happens here, not at the call site. */
  value: number;
  /** The precise line: metric, gain, baseline, group count. */
  technical: string;
  /** Repo-relative evidence path; EvidenceLink decides where it resolves. */
  sourceDoc: string;
  /**
   * `titled` leads with the plain gloss, for a reader meeting the programme for
   * the first time. `value-first` leads with the numeral, for the scorecard,
   * where someone is scanning four figures side by side. Both keep the gloss
   * and both keep the technical line — only the order changes.
   */
  variant?: "titled" | "value-first";
}

const formatGain = (v: number) => `+${v.toFixed(2)}%`;

/**
 * One landmark, framed as a blueprint object. The whole card links to that
 * landmark's source evidence, so a plain-English claim is always one click from
 * the document that backs it.
 */
export default function LandmarkCard({
  kicker,
  plain,
  value,
  technical,
  sourceDoc,
  variant = "titled",
}: Props) {
  const animated = useCountUp(value);

  const gloss = <div className="landmark-plain">{plain}</div>;
  const figure = <div className="landmark-value">{formatGain(animated)}</div>;

  return (
    <Blueprint as={EvidenceLink} sourceDoc={sourceDoc} className="landmark-card">
      <div className="card-kicker">{kicker}</div>
      {variant === "titled" ? (
        <>
          {gloss}
          {figure}
        </>
      ) : (
        <>
          {figure}
          {gloss}
        </>
      )}
      <div className="landmark-technical">{technical}</div>
    </Blueprint>
  );
}
