# TODO

## Priority: High

- [ ] Task 1: React setup - Create React+Vite app in ui/ directory, set up package.json, vite.config.ts, update Makefile to build static and embed in Go binary
  - **Context:** Current dashboard uses Go templ. We need React+Vite for MUI-style dashboard.
  - **Validation Criteria:** npm run build creates static files, Go serves them at /dashboard route
  - **Created:** 2026-04-22

- [ ] Task 2: Material UI components - Install MUI core + icons, replace templ components with MUI (Card, Button, Stat, etc.)
  - **Context:** MUI matches the dashboard template we referenced. Use @mui/material and @mui/icons-material.
  - **Validation Criteria:** Dashboard uses MUI components for cards, stats, buttons
  - **Created:** 2026-04-22

- [ ] Task 3: Dashboard UI design - Redesign header, sidebar, main content area with MUI dashboard styling
  - **Context:** Match MUI dashboard template style: clean stat cards, icons, organized layout
  - **Validation Criteria:** Dashboard visually similar to MUI template, cleaner than current templ version
  - **Created:** 2026-04-22

- [ ] Task 4: Real-time SSE integration - React app connects to existing SSE /events endpoint, health bar and stats update in real-time
  - **Context:** Current SSE works. Need to keep for React but also enable React SSE client
  - **Validation Criteria:** Health bar updates every few seconds without page refresh
  - **Created:** 2026-04-22

- [ ] Task 5: Remove old templ files - Delete old templ files from internal/dashboard/components/ after React is fully working
  - **Context:** Clean upGo templ files after React replaces them
  - **Validation Criteria:** Old .templ files removed, new React dashboard works
  - **Created:** 2026-04-22

## Priority: Medium

## Priority: Low