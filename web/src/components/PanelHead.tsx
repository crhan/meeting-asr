import type { ReactNode } from "react";

/** Section eyebrow + title, the instrument-panel header of the voiceprint pages. */
export function PanelHead(props: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="vq-panel-head">
      <div className="vq-panel-titles">
        <span className="vq-eyebrow">{props.eyebrow}</span>
        <h2>{props.title}</h2>
      </div>
      {props.children}
    </div>
  );
}
