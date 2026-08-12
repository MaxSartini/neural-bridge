import type { ComponentPropsWithoutRef, ElementType, ReactNode } from "react";

/**
 * The Industry blueprint frame: a square, hairline-bordered box carrying "+"
 * registration marks 6px outside each corner.
 *
 * This is a component rather than a bare class because the frame is four
 * separate elements it cannot render without — "never drop the registration
 * marks" is one of the design system's two stated Don'ts — and hand-writing
 * them at every call site is how one eventually goes missing. The marks are
 * decorative, so they are aria-hidden and contribute nothing to the tree.
 *
 * `as` covers the elements the design actually frames: cards (div/section),
 * the primary button, and the report's figure.
 */
type Props<T extends ElementType> = {
  as?: T;
  className?: string;
  children?: ReactNode;
} & Omit<ComponentPropsWithoutRef<T>, "as" | "className" | "children">;

export default function Blueprint<T extends ElementType = "div">({
  as,
  className,
  children,
  ...rest
}: Props<T>) {
  const Element = (as ?? "div") as ElementType;
  return (
    <Element className={className ? `blueprint ${className}` : "blueprint"} {...rest}>
      <i className="corner tl" aria-hidden="true" />
      <i className="corner tr" aria-hidden="true" />
      <i className="corner bl" aria-hidden="true" />
      <i className="corner br" aria-hidden="true" />
      {children}
    </Element>
  );
}
