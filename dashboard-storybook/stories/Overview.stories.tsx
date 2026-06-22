import type { Meta, StoryObj } from "@storybook/react-vite";
import { DashboardHtml } from "./DashboardHtml";
import {
  embedPanelDegradedFixture,
  embedPanelHealthyFixture,
  healthBarDegradedFixture,
  healthBarHealthyFixture,
  indexHealthHealthyFixture,
  indexHealthDegradedFixture,
  overviewFixture,
  overviewTopBarFixture,
  statsCardsFixture,
} from "./fixtures";

const meta = {
  title: "Dashboard/Overview",
  component: DashboardHtml,
  parameters: {
    docs: {
      description: {
        component:
          "Static HTML fixtures using production dashboard CSS from `internal/dashboard/static/styles.css`. Matches the operator dashboard served on port 7830.",
      },
    },
  },
} satisfies Meta<typeof DashboardHtml>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Representative overview: health bar, embeddings panel, and stat cards. */
export const Overview: Story = {
  render: (args) => <DashboardHtml html={args.html ?? overviewFixture} />,
  args: { html: overviewFixture },
};

export const EmbedPanel: Story = {
  render: (args) => <DashboardHtml html={args.html ?? embedPanelDegradedFixture} />,
  args: { html: embedPanelDegradedFixture },
};

export const EmbedPanelHealthy: Story = {
  name: "Embed panel (healthy)",
  render: (args) => <DashboardHtml html={args.html ?? embedPanelHealthyFixture} />,
  args: { html: embedPanelHealthyFixture },
};

export const HealthBar: Story = {
  render: (args) => (
    <DashboardHtml
      html={
        args.html ??
        overviewTopBarFixture(healthBarDegradedFixture)
      }
    />
  ),
  args: { html: overviewTopBarFixture(healthBarDegradedFixture) },
};

export const HealthBarOk: Story = {
  name: "Health bar (ok)",
  render: (args) => (
    <DashboardHtml
      html={args.html ?? overviewTopBarFixture(healthBarHealthyFixture)}
    />
  ),
  args: { html: overviewTopBarFixture(healthBarHealthyFixture) },
};

export const StatsRow: Story = {
  name: "Query activity cards",
  render: (args) => <DashboardHtml html={args.html ?? statsCardsFixture} />,
  args: { html: statsCardsFixture },
};

export const IndexHealth: Story = {
  render: (args) => <DashboardHtml html={args.html ?? indexHealthHealthyFixture} />,
  args: { html: indexHealthHealthyFixture },
};

export const IndexHealthDegraded: Story = {
  name: "Index health (degraded embed)",
  render: (args) => <DashboardHtml html={args.html ?? indexHealthDegradedFixture} />,
  args: { html: indexHealthDegradedFixture },
};
