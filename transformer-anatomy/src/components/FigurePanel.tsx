import type { ReactNode } from 'react';

interface FigurePanelProps {
  tag: string;
  title: string;
  description: ReactNode;
  footer?: ReactNode;
  controls?: ReactNode;
  children: ReactNode;
}

export function FigurePanel({ tag, title, description, footer, controls, children }: FigurePanelProps) {
  return (
    <div className="fig">
      <div className="fig-tag">{tag}</div>
      <div className="fig-head">
        <h2>{title}</h2>
        <p>{description}</p>
        {controls}
      </div>
      <div className="fig-body">{children}</div>
      {footer && <div className="fig-foot">{footer}</div>}
    </div>
  );
}
