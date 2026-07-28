import type { Preview } from '@storybook/react-vite'
import { CssBaseline, ThemeProvider } from '@mui/material'
import { dashboardTheme } from '../src/theme'
import { ToastProvider } from '../src/context/ToastContext'
import '../src/index.css'

const preview: Preview = {
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dashboard-dark',
      values: [{ name: 'dashboard-dark', value: '#0d1117' }],
    },
    options: {
      storySort: {
        order: ['Dashboard', ['Overview', 'Memory', 'Embeddings', 'Settings']],
      },
    },
  },
  decorators: [
    (Story) => (
      <ThemeProvider theme={dashboardTheme}>
        <CssBaseline />
        <ToastProvider>
          <Story />
        </ToastProvider>
      </ThemeProvider>
    ),
  ],
}

export default preview
