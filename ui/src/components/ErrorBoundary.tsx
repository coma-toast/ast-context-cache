import { Component, type ReactNode } from 'react'
import { Alert, Box, Button } from '@mui/material'

export class ErrorBoundary extends Component<
  { children: ReactNode; label?: string },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={() => this.setState({ error: null })}>
              Retry
            </Button>
          }
        >
          <Box component="span" sx={{ display: 'block', fontWeight: 600 }}>
            {this.props.label || 'Panel'} failed to render
          </Box>
          {this.state.error.message}
        </Alert>
      )
    }
    return this.props.children
  }
}
