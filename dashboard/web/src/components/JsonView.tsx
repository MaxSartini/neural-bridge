import { JsonView as RjvJsonView, collapseAllNested } from "react-json-view-lite";
import type { StyleProps } from "react-json-view-lite/dist/DataRenderer";

interface Props {
  content: string;
  jsonl?: boolean;
}

/**
 * The library's own `defaultStyles`/`darkStyles` bake colors into hashed CSS
 * module classes, which can't follow a `data-theme` flip. Supplying our own
 * class names instead puts every color in global.css where the theme tokens
 * live — so the tree re-themes with the rest of the page.
 */
const styles: StyleProps = {
  container: "json-view",
  basicChildStyle: "jv-row",
  label: "jv-label",
  clickableLabel: "jv-label jv-label-clickable",
  nullValue: "jv-null",
  undefinedValue: "jv-null",
  numberValue: "jv-number",
  stringValue: "jv-string",
  booleanValue: "jv-bool",
  otherValue: "jv-other",
  punctuation: "jv-punct",
  expandIcon: "jv-icon jv-icon-expand",
  collapseIcon: "jv-icon jv-icon-collapse",
  collapsedContent: "jv-collapsed",
  noQuotesForStringValues: false,
};

export default function JsonView({ content, jsonl }: Props) {
  let data: unknown;
  try {
    data = jsonl
      ? content
          .split("\n")
          .filter((line) => line.trim().length > 0)
          .map((line) => JSON.parse(line))
      : JSON.parse(content);
  } catch {
    return (
      <div>
        <p className="notice">
          Couldn&apos;t parse as JSON (the file may have been cut off by the size limit). Showing
          raw text.
        </p>
        <pre className="raw-fallback">{content}</pre>
      </div>
    );
  }
  return (
    <RjvJsonView data={data as object} shouldExpandNode={collapseAllNested} style={styles} />
  );
}
