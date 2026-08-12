import { Link } from "react-router-dom";
import type { TreeEntry } from "../api/client";
import { formatBytes } from "../lib/formatBytes";

interface Props {
  entries: TreeEntry[];
}

function targetFor(entry: TreeEntry): string {
  if (entry.entryType === "dir") return `/browse/${entry.path}`;
  // The server labels; this only routes. Re-deriving ".md is markdown" here was
  // the fourth copy of that decision.
  if (entry.kind === "markdown") return `/doc/${entry.path}`;
  return `/view/${entry.path}`;
}

export default function FileTree({ entries }: Props) {
  if (entries.length === 0) {
    return <p className="empty">Empty directory.</p>;
  }
  return (
    <ul className="file-tree">
      {entries.map((entry) => (
        <li key={entry.path} className={entry.entryType}>
          <Link to={targetFor(entry)}>
            <span className="icon">{entry.entryType === "dir" ? "\u{1F4C1}" : "\u{1F4C4}"}</span>
            {entry.name}
          </Link>
          {entry.entryType === "file" && <span className="size">{formatBytes(entry.size)}</span>}
        </li>
      ))}
    </ul>
  );
}
