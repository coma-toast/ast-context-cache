import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider } from '@mui/material'
import App from './App'
import { dashboardTheme } from './theme'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={dashboardTheme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
