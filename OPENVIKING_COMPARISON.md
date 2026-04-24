# OpenViking vs ast-context-cache Comparison

## Overview

This document compares two context management systems for AI agents: **OpenViking** (a general-purpose context database) and **ast-context-cache** (a specialized AST context engine for coding agents).

## Core Purpose

| Aspect | OpenViking | ast-context-cache |
|--------|------------|-------------------|
| **Primary Goal** | Unified context management for AI Agents (memories, resources, skills) | Precise code context retrieval for coding agents |
| **Scope** | General agent context (beyond just code) | Code-specific context for development agents |
| **Target Users** | AI agent developers building complex agents | Developers using coding assistants (Cursor, OpenCode, etc.) |

## Architecture Approach

| Aspect | OpenViking | ast-context-cache |
|--------|------------|-------------------|
| **Paradigm** | File system paradigm with Viking URI scheme | Tree-sitter AST parsing + SQLite database |
| **Context Organization** | Hierarchical virtual filesystem (`viking://resources/`, `viking://user/`, etc.) | Symbol-based indexing with dependency graphs |
| **Retrieval Mechanism** | Tiered loading (L0/L1/L2), directory recursive search, visualized trajectories | BM25 full-text + optional semantic search, dependency graph traversal |
| **Storage** | Workspace-based with model dependencies | Local SQLite database (~/.astcache/usage.db) |

## Technical Implementation

| Aspect | OpenViking | ast-context-cache |
|--------|------------|-------------------|
| **Languages** | Python/Go/C++ | Go with CGO |
| **Dependencies** | Requires external VLM and embedding models | Self-contained (bundles ONNX Runtime) |
| **Deployment** | Client-server architecture | Local-first MCP server |
| **Configuration** | Requires model setup (API keys for external services) | Minimal setup (`make setup`) |
| **Data Privacy** | May send data to external model providers | 100% local, no data leaves machine |

## Features Comparison

### Context Types Managed
| Feature | OpenViking | ast-context-cache |
|---------|------------|-------------------|
| **Memories** | ✅ Agent experiences and interactions | ❌ Not focused on agent memory |
| **Resources** | ✅ Project docs, repos, web pages | ❌ Limited to codebase files |
| **Skills** | ✅ Agent capabilities and tools | ❌ Not designed for skill management |
| **Code Symbols** | ❌ Indirect through resources | ✅ Functions, classes, variables with full source |
| **Dependencies** | ❌ Limited | ✅ Import/call edges with impact analysis |
| **File Context** | ❌ Indirect | ✅ Complete file symbol overview |

### Retrieval Capabilities
| Feature | OpenViking | ast-context-cache |
|---------|------------|-------------------|
| **Full-text Search** | ❌ Relies on external models | ✅ BM25-ranked FTS5 |
| **Semantic Search** | ✅ Via configured embedding models | ✅ Optional ONNX vector search |
| **Path-based Search** | ✅ Filesystem hierarchy navigation | ❌ No direct path search |
| **Dependency Analysis** | ❌ Not a core feature | ✅ Blast radius via `get_impact_graph` |
| **Context Visualization** | ✅ Visualized retrieval trajectories | ✅ Dashboard with stats and charts |
| **Incremental Updates** | ❌ Not specified | ✅ fsnotify-based file watcher |

### Integration & Usage
| Feature | OpenViking | ast-context-cache |
|---------|------------|-------------------|
| **Agent Integration** | Viking URI scheme (`viking://...`) | MCP Server (:7821/mcp) |
| **Editor Support** | Plugin-specific (OpenClaw, etc.) | Cursor, OpenCode via MCP |
| **CLI Tools** | ✅ `ov` command suite | ✅ `ast-mcp` shell functions |
| **Dashboard** | ❌ Not mentioned | ✅ Web UI on port 7830 |
| **Setup Complexity** | Higher (model configuration) | Lower (`make setup`) |
| **Cross-platform** | ✅ Linux/macOS/Windows | ✅ Linux/macOS (with notes) |

## Strengths & Trade-offs

### OpenViking Advantages
1. **Unified Context Management** - Handles memories, resources, and skills in one system
2. **Observable Retrieval** - Visualized trajectories help debug and optimize context acquisition
3. **Context Self-iteration** - Automatic session management extracts long-term memory
4. **Production-Ready** - Includes cloud deployment guidance and plugin ecosystem
5. **Framework Agnostic** - Designed to work with various agent frameworks

### ast-context-cache Advantages
1. **Code Specialization** - Purpose-built for code understanding in coding agents
2. **Zero Configuration** - Local-first with no external API dependencies
3. **Precision** - AST-aware search returns actual source code, not just file pointers
4. **Token Efficiency** - Optimized results for LLM context windows
5. **Dependency Intelligence** - Impact analysis helps understand code change effects
6. **Transparency** - MIT licensed, simple architecture

## Ideal Use Cases

### Choose OpenViking when:
- Building complex AI agents that need to manage diverse context types
- Requiring observable context retrieval for debugging/optimization
- Planning production deployment with cloud infrastructure
- Wanting a unified system for memories, skills, and resources
- Needing framework-specific plugins (OpenClaw, etc.)

### Choose ast-context-cache when:
- Enhancing coding agents (Cursor, OpenCode, etc.) with code intelligence
- Wanting zero-setup, local-only solution for code context
- Needing precise symbol search with dependency tracking
- Prioritizing token efficiency for LLM interactions
- Preferring MIT-licensed, transparent local tool

## Conclusion

While both systems address the challenge of providing context to AI agents, they serve different niches:

- **OpenViking** is a **general-purpose context platform** for sophisticated AI agents needing unified management of all context types (beyond just code).
- **ast-context-cache** is a **specialized code intelligence tool** optimized specifically for helping coding agents understand and navigate codebases efficiently.

For coding agent enhancement specifically, ast-context-cache offers a more focused, lightweight, and private solution. OpenViking provides broader context management capabilities suitable for complex agent architectures that extend beyond code understanding.
