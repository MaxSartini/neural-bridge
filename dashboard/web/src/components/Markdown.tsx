import type { AnchorHTMLAttributes, ImgHTMLAttributes } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useNavigate } from "react-router-dom";
import { resolveLink } from "../lib/linkRewrite";
import { rawUrl } from "../api/client";

interface Props {
  content: string;
  docPath: string;
}

function LinkRenderer({
  docPath,
  href,
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { docPath: string }) {
  const navigate = useNavigate();
  if (!href) return <a {...rest}>{children}</a>;

  const target = resolveLink(docPath, href);
  if (target.kind === "external") {
    return (
      <a href={target.href} target="_blank" rel="noopener noreferrer" {...rest}>
        {children}
      </a>
    );
  }
  if (target.kind === "raw") {
    return (
      <a href={rawUrl(target.path)} target="_blank" rel="noopener noreferrer" {...rest}>
        {children}
      </a>
    );
  }
  const to = target.kind === "doc" ? `/doc/${target.path}` : `/view/${target.path}`;
  return (
    <a
      href={to}
      {...rest}
      onClick={(e) => {
        e.preventDefault();
        navigate(to);
      }}
    >
      {children}
    </a>
  );
}

function ImageRenderer({
  docPath,
  src,
  alt,
}: ImgHTMLAttributes<HTMLImageElement> & { docPath: string }) {
  if (!src) return null;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(src)) {
    return <img src={src} alt={alt ?? ""} loading="lazy" />;
  }
  const target = resolveLink(docPath, src);
  const resolvedSrc = target.kind === "raw" ? rawUrl(target.path) : src;
  return <img src={resolvedSrc} alt={alt ?? ""} loading="lazy" />;
}

export default function Markdown({ content, docPath }: Props) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => <LinkRenderer docPath={docPath} {...props} />,
          img: (props) => <ImageRenderer docPath={docPath} {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
