import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { evidenceTarget } from "../lib/evidenceLink";

interface Props {
  sourceDoc: string;
  className?: string;
  children: ReactNode;
}

/**
 * Renders a provenance link without the caller knowing where evidence lives.
 * In the local dashboard that's an in-app route; in the static bundle it's the
 * private GitHub repo, opened in a new tab.
 */
export default function EvidenceLink({ sourceDoc, className, children }: Props) {
  const target = evidenceTarget(sourceDoc);

  if (target.mode === "external") {
    return (
      <a href={target.href} className={className} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  }
  return (
    <Link to={target.to} className={className}>
      {children}
    </Link>
  );
}
