import type { ReactNode } from "react";

/** Renders static dashboard HTML using production CSS class names. */
export function DashboardHtml({ html, className }: { html: string; className?: string }) {
  return (
    <div
      className={className ?? "dashboard-story-root"}
      style={{ padding: 24, background: "var(--bg)", minHeight: "100vh" }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function StoryFrame({ children }: { children: ReactNode }) {
  return (
    <div style={{ padding: 24, background: "var(--bg)", minHeight: "100vh" }}>{children}</div>
  );
}
