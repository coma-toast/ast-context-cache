import type { StorybookConfig } from "@storybook/react-vite";

const config: StorybookConfig = {
  stories: ["../stories/**/*.mdx", "../stories/**/*.stories.@(ts|tsx)"],
  addons: [
    "@storybook/addon-docs",
    "@storybook/addon-a11y",
    "@storybook/addon-mcp",
  ],
  framework: "@storybook/react-vite",
  staticDirs: [{ from: "../../internal/dashboard/static", to: "/static" }],
};

export default config;
