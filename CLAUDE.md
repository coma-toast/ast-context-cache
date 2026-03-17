# Code Context Instructions

# MCP Server for This Project

This project includes an MCP server (`ast-context-cache`) that provides efficient code search tools.
When researching this codebase, **always prefer using the MCP tools** over direct grep/read:
- Faster and more token-efficient than grep/read/glob
- Supports semantic search and smart caching
- Run `make run` to start the MCP server, then use the tools below

# Efficient Code Context Usage Guide

## Token Optimization Strategies

### 1. Use 'auto' Mode (Recommended Default)
- Mode 'auto' returns full source for top 3 results, skeleton for rest
- This provides ~80% token savings while maintaining detail for key matches
- Only use 'full' when you need complete implementation details

### 2. Use 'skeleton' Mode for Exploration
- Use mode='skeleton' when exploring unfamiliar codebases
- Returns function signatures only (~90% token reduction)
- Perfect for understanding structure before diving into details

### 3. Use 'summary' Mode for Broad Context
- Use mode='summary' after exploring to get high-level overviews
- Requires calling cache_summary first to create summaries
- Provides ~94% token reduction

### 4. Leverage Session Deduplication
- Always pass session_id to get_context_capsule
- This prevents re-sending files you've already seen in this conversation
- The tool tracks what's been returned and auto-skips duplicates

### 5. Use Token Budget Wisely
- Set token_budget to control response size
- Default is 4000 tokens - adjust based on need
- Tool stops adding results when budget is exhausted

### 6. Use Semantic Search for Intent
- search_semantic finds symbols by meaning, not just text
- Natural language queries: "function that handles auth"
- Great for exploratory searches

### 7. Cache Your Own Summaries
- Call cache_summary after understanding a file
- Future queries using mode='summary' will use your cached summaries
- This creates personalized, token-efficient context

### 8. Use get_project_map First
- For new projects, start with get_project_map (depth=1 or 2)
- Understand structure before diving into files
- Only ~200 tokens for full project overview

### 9. Use get_impact_graph for Change Analysis
- Before modifying code, call get_impact_graph
- Shows all files that depend on a symbol
- Helps understand blast radius of changes

### Recommended Workflow
1. get_project_map to understand structure
2. get_context_capsule with mode='auto' for initial search
3. Use skeleton mode for broad exploration
4. Cache summaries of key files
5. Use impact graph before making changes

Use these tools for efficient code search:
- get_context_capsule: Search code with token-efficient modes
- cache_summary: Cache your own summaries
- analyze_dead_code: Find unused code
