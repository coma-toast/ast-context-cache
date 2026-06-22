import type { ReactNode } from "react";

/** Renders static dashboard HTML using production CSS class names. */
export function DashboardHtml({ html, className }: { html: string; className?: string }) {
  return (
    <div className={className ?? "dashboard-story-root"}>
      <div className="main-area" dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

export function StoryFrame({ children }: { children: ReactNode }) {
  return (
    <div className="dashboard-story-root">
      <div className="main-area">{children}</div>
    </div>
  );
}
