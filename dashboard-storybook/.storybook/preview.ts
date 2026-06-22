import type { Preview } from "@storybook/react-vite";
import "../../internal/dashboard/static/styles.css";

const preview: Preview = {
  parameters: {
    layout: "fullscreen",
    backgrounds: {
      default: "dashboard",
      values: [{ name: "dashboard", value: "#0d1117" }],
    },
  },
};

export default preview;
