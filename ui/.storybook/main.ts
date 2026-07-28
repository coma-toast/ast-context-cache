import { fileURLToPath } from 'node:url'
import type { StorybookConfig } from '@storybook/react-vite'

const apiStub = fileURLToPath(new URL('../src/storybook/api-stub.ts', import.meta.url))

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.tsx'],
  addons: ['@storybook/addon-docs', '@storybook/addon-a11y'],
  framework: '@storybook/react-vite',
  viteFinal: async (viteConfig) => {
    viteConfig.resolve ??= {}
    const existing = Array.isArray(viteConfig.resolve.alias) ? viteConfig.resolve.alias : []
    viteConfig.resolve.alias = [
      ...existing,
      // Stub the real API client so idle stories never hit a live ast-mcp server.
      { find: '../api/client', replacement: apiStub },
      { find: './api/client', replacement: apiStub },
    ]
    return viteConfig
  },
}

export default config
