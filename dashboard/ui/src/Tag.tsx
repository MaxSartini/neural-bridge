import type { ReactNode } from "react";

/**
 * A small label. `accent` is a tinted fill (Done, Live, Complete, a peak
 * timestamp), `outline` is a hairline box for a state that is not yet
 * settled (Processing, In progress, a weak-moment timestamp), and `neutral`
 * recedes for work that has not started (Queued).
 */
export type TagVariant = "accent" | "outline" | "neutral";

interface Props {
  variant?: TagVariant;
  className?: string;
  children: ReactNode;
}

export default function Tag({ variant = "accent", className, children }: Props) {
  const classes = ["tag", `tag-${variant}`];
  if (className) classes.push(className);
  return <span className={classes.join(" ")}>{children}</span>;
}
