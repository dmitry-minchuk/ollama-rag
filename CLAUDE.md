# Graph-Enhanced RAG MCP Server

use-mcp ollama-rag
# Use ollama-rag to understand codebase in depth using Vector and Graph embeddings
use-mcp playwright
# Use it for more UI understanding
use-mcp context7
# Use context7 for searching documentation

A Model Context Protocol (MCP) server that provides graph-enhanced code search capabilities by combining vector similarity search with code relationship analysis.

## Status: TESTED & WORKING
✅ MCP server correctly returns both [VECTOR] and [GRAPH] search results
✅ Data is properly isolated to this project only (no external codebase data)
✅ Graph visualization interface available at http://localhost:5001
✅ Full MCP integration tested and verified

## Features

- **Hybrid Search**: Combines vector similarity with code relationship graphs
- **Code Understanding**: Analyzes imports, inheritance, and method calls
- **Multi-hop Relations**: Finds related code through structural connections
- **Java Support**: Specialized parsing for Java codebases
- **MCP Integration**: Full Model Context Protocol server implementation
- **Graph Visualization**: Interactive web interface for exploring code relationships
- **Real-time Search**: Instant search across nodes and relationships

## Setup

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Index Your Codebase**:
   ```bash
   python ingest-code.py
   ```

3. **Start MCP Server**:
   ```bash
   python mcp_server.py
   ```

## Usage

The server provides a single `code_search` tool that performs graph-enhanced code search:

```bash
# Search for code with both vector similarity and graph relationships
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "code_search", "arguments": {"query": "AdminTab user validation", "limit": 3}}, "id": 1}' | python mcp_server.py
```

## Architecture

- **Vector Store**: ChromaDB with HuggingFace embeddings for semantic search
- **Code Graph**: NetworkX-based relationship graph for structural analysis
- **Java Parser**: Extracts imports, inheritance, and method calls
- **Hybrid Search**: Combines vector similarity with graph traversal

## Configuration

Edit paths in `ingest-code.py` to index multiple codebases:

```python
CODEBASE_PATHS = [
    "/path/to/your/codebase1",
    "/path/to/your/codebase2",
]
```

## Files

- `mcp_server.py` - Main MCP server with graph-enhanced search
- `ingest-code.py` - Codebase indexing script
- `requirements.txt` - Python dependencies
- `chroma_db/` - Vector database storage
- `code_graph.pkl` - Serialized code relationship graph

## Search Results

Results include both:
- **[VECTOR]** - Semantically similar code chunks
- **[GRAPH]** - Structurally related code through imports/calls/inheritance

This provides comprehensive code understanding that goes beyond simple text similarity.