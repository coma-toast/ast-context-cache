import { Component, type ReactNode } from 'react'
import { Alert, Box, Button } from '@mui/material'

export class ErrorBoundary extends Component<
  { children: ReactNode; label?: string; onRetry?: () => void },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  // Retry used to only clear the boundary's own error state — if the crash was
  // caused by the currently-loaded data (not a one-off render glitch), the
  // panel re-rendered that same data and crashed again immediately. onRetry
  // lets the caller re-fetch first.
  handleRetry = () => {
    this.props.onRetry?.()
    this.setState({ error: null })
  }

  render() {
    if (this.state.error) {
      return (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={this.handleRetry}>
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
