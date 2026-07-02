import { createTheme } from '@mui/material/styles'

export const dashboardTheme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#0d1117',
      paper: '#161b22',
    },
    primary: { main: '#58a6ff' },
    secondary: { main: '#bc8cff' },
    success: { main: '#3fb950' },
    warning: { main: '#d29922' },
    error: { main: '#f85149' },
    text: {
      primary: '#e6edf3',
      secondary: '#8b949e',
    },
    divider: '#30363d',
  },
  typography: {
    fontFamily: '"Inter", -apple-system, sans-serif',
    overline: {
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: '0.06em',
      lineHeight: 1.4,
    },
    h5: {
      fontSize: 22,
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        html: { scrollBehavior: 'smooth' },
        body: { backgroundColor: '#0d1117' },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none' },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          color: '#8b949e',
          fontWeight: 500,
          fontSize: 13,
          py: 1.25,
          px: 1.5,
          marginBottom: 4,
          '&.Mui-selected': {
            bgcolor: 'rgba(88,166,255,0.12)',
            color: '#58a6ff',
            border: '1px solid rgba(88,166,255,0.35)',
            boxShadow: 'inset 3px 0 0 #58a6ff',
            '&:hover': { bgcolor: 'rgba(88,166,255,0.16)' },
          },
          '&:hover': { bgcolor: 'rgba(88,166,255,0.08)' },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundImage: 'linear-gradient(180deg, #161b22 0%, #0d1117 100%)',
          borderRight: '1px solid #30363d',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: { backgroundImage: 'none' },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: '#8b949e',
          py: 1,
          px: 1.5,
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: 36,
          textTransform: 'none',
          fontSize: 13,
          borderRadius: 8,
          mr: 0.5,
          '&.Mui-selected': {
            bgcolor: 'rgba(88,166,255,0.15)',
            color: '#58a6ff',
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: { display: 'none' },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { height: 24 },
        outlined: { borderColor: '#30363d' },
      },
    },
  },
})
