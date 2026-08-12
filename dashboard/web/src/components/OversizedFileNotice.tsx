import { rawUrl } from "../api/client";
import { formatBytes } from "../lib/formatBytes";

interface Props {
  path: string;
  size: number;
}

export default function OversizedFileNotice({ path, size }: Props) {
  return (
    <div className="oversized-notice">
      <strong>Showing a truncated preview.</strong> This file is {formatBytes(size)}, above the
      5&nbsp;MB inline preview limit.{" "}
      <a href={rawUrl(path)} target="_blank" rel="noopener noreferrer">
        Open the full file
      </a>
      .
    </div>
  );
}
