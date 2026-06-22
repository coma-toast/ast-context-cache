import type { Meta, StoryObj } from "@storybook/react-vite";
import { DashboardHtml } from "./DashboardHtml";
import {
  embedPanelFixture,
  healthBarFixture,
  overviewFixture,
  statsRowFixture,
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
  render: (args) => <DashboardHtml html={args.html ?? embedPanelFixture} />,
  args: { html: embedPanelFixture },
};

export const HealthBar: Story = {
  render: (args) => <DashboardHtml html={args.html ?? healthBarFixture} />,
  args: { html: healthBarFixture },
};

export const StatsRow: Story = {
  render: (args) => <DashboardHtml html={args.html ?? statsRowFixture} />,
  args: { html: statsRowFixture },
};
