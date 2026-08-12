import { useNavigate } from "react-router-dom";
import { evidenceTarget } from "../lib/evidenceLink";

/**
 * Chart click-through to a piece of evidence. Same decision as EvidenceLink,
 * but for the imperative case (Recharts hands us a click, not an anchor).
 */
export function useEvidenceNavigate(): (sourceDoc: string) => void {
  const navigate = useNavigate();
  return (sourceDoc: string) => {
    const target = evidenceTarget(sourceDoc);
    if (target.mode === "external") {
      window.open(target.href, "_blank", "noopener,noreferrer");
      return;
    }
    navigate(target.to);
  };
}
