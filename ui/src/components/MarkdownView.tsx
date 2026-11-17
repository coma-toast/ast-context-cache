import { Box, type SxProps, type Theme } from '@mui/material'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'

export interface MarkdownViewProps {
  content?: string | null
  sx?: SxProps<Theme>
}

const markdownComponents: Components = {
  a: ({ node: _node, href, children, ...rest }) => {
    const isExternal = Boolean(href && /^(https?:)?\/\//i.test(href))
    return (
      <a
        href={href}
        {...(isExternal ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
        {...rest}
      >
        {children}
      </a>
    )
  },
  table: ({ node: _node, children, ...rest }) => (
    <Box sx={{ overflowX: 'auto', my: 1.5 }}>
      <table {...rest}>{children}</table>
    </Box>
  ),
}

const markdownStyles: SxProps<Theme> = {
  color: '#e6edf3',
  fontSize: 14,
  lineHeight: 1.6,
  wordBreak: 'break-word',
  '& > :first-of-type': { mt: 0 },
  '& > :last-of-type': { mb: 0 },
  '& h1, & h2, & h3, & h4, & h5, & h6': {
    color: '#e6edf3',
    fontWeight: 600,
    lineHeight: 1.3,
    mt: 2,
    mb: 1,
  },
  '& h1': {
    fontSize: '1.35rem',
    pb: 0.5,
    borderBottom: '1px solid #30363d',
  },
  '& h2': {
    fontSize: '1.15rem',
    pb: 0.5,
    borderBottom: '1px solid #30363d',
  },
  '& h3': { fontSize: '1rem' },
  '& h4': { fontSize: '0.925rem' },
  '& h5': { fontSize: '0.875rem' },
  '& h6': { fontSize: '0.85rem', color: '#8b949e' },
  '& p': {
    my: 1,
    lineHeight: 1.6,
  },
  '& ul, & ol': {
    pl: 3,
    my: 1,
  },
  '& li': {
    my: 0.35,
    '& > p': { my: 0.35 },
  },
  '& li input[type="checkbox"]': {
    mr: 0.75,
    accentColor: '#58a6ff',
  },
  '& a': {
    color: '#58a6ff',
    textDecoration: 'none',
    '&:hover': { textDecoration: 'underline' },
  },
  '& blockquote': {
    my: 1.25,
    mx: 0,
    pl: 1.75,
    py: 0.25,
    borderLeft: '3px solid #30363d',
    color: '#8b949e',
  },
  '& hr': {
    border: 0,
    borderTop: '1px solid #30363d',
    my: 2,
  },
  '& code': {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    fontSize: '0.85em',
    bgcolor: 'rgba(110, 118, 129, 0.2)',
    border: '1px solid rgba(48, 54, 61, 0.7)',
    borderRadius: '4px',
    px: 0.6,
    py: 0.15,
    color: '#e6edf3',
  },
  '& pre': {
    bgcolor: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 1,
    p: 1.5,
    my: 1.25,
    overflowX: 'auto',
    lineHeight: 1.5,
    fontSize: '0.85em',
  },
  '& pre code': {
    bgcolor: 'transparent',
    border: 'none',
    p: 0,
    borderRadius: 0,
    fontSize: 'inherit',
    color: '#e6edf3',
  },
  '& table': {
    borderCollapse: 'collapse',
    width: '100%',
    fontSize: '0.875rem',
  },
  '& th, & td': {
    border: '1px solid #30363d',
    px: 1.25,
    py: 0.75,
    textAlign: 'left',
  },
  '& th': {
    bgcolor: '#161b22',
    color: '#e6edf3',
    fontWeight: 600,
  },
  '& tbody tr:nth-of-type(even)': {
    bgcolor: 'rgba(255, 255, 255, 0.02)',
  },
  '& img': {
    maxWidth: '100%',
    borderRadius: 1,
  },
}

export function MarkdownView({ content, sx }: MarkdownViewProps) {
  if (!content || content.trim() === '') {
    return null
  }

  return (
    <Box
      data-testid="markdown-view"
      sx={[markdownStyles, ...(Array.isArray(sx) ? sx : sx ? [sx] : [])]}
    >
      <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </Markdown>
    </Box>
  )
}
