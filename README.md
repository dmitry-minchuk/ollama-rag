# Graph-Enhanced RAG MCP Server

A Model Context Protocol (MCP) server that provides graph-enhanced code search capabilities by combining vector similarity search with code relationship analysis.

## Features

- **Hybrid Search**: Combines vector similarity with code relationship graphs
- **Code Understanding**: Analyzes imports, inheritance, and method calls
- **Multi-hop Relations**: Finds related code through structural connections
- **Java Support**: Specialized parsing for Java codebases

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