import { Link } from "react-router-dom";

interface Props {
  /** Repo-relative path of the thing being shown, file or directory. */
  path: string;
}

/**
 * Ancestor trail for whatever is on screen.
 *
 * Every segment before the last one is a **directory**, whatever the current
 * page is, so every ancestor link goes to `/browse`. This used to take a `base`
 * prop and mirror it into the link, which meant a doc page emitted `/doc/<dir>`
 * — and `/api/file` answers 415 for a directory, so every intermediate crumb on
 * every `/doc/*` page was a dead end.
 *
 * The last segment is the current location and renders as plain text.
 */
export default function Breadcrumbs({ path }: Props) {
  const segments = path.split(/[\\/]/).filter(Boolean);
  let acc = "";
  return (
    <nav className="breadcrumbs">
      {segments.length === 0 ? <span>root</span> : <Link to="/browse">root</Link>}
      {segments.map((seg, i) => {
        acc = acc ? `${acc}/${seg}` : seg;
        const isLast = i === segments.length - 1;
        return (
          <span key={acc}>
            {" / "}
            {isLast ? <span>{seg}</span> : <Link to={`/browse/${acc}`}>{seg}</Link>}
          </span>
        );
      })}
    </nav>
  );
}
