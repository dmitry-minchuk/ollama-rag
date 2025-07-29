# Graph-Enhanced RAG MCP Server

A Model Context Protocol (MCP) server that provides graph-enhanced code search capabilities by combining vector similarity search with code relationship analysis.

## Features

- **Hybrid Search**: Combines vector similarity with code relationship graphs
- **Code Understanding**: Analyzes imports, inheritance, and method calls
- **Multi-hop Relations**: Finds related code through structural connections
- **Java Support**: Specialized parsing for Java codebases
- **MCP Integration**: Full Model Context Protocol server implementation
- **Graph Visualization**: Interactive web interface for exploring code relationships

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

### MCP Server Integration

The server provides a single `code_search` tool that performs graph-enhanced code search. Configure in your MCP client:

```json
{
  "mcpServers": {
    "ollama-rag": {
      "command": "python",
      "args": ["/path/to/ollama-rag/mcp_server.py"]
    }
  }
}
```

### Direct Usage

```bash
# Search for code with both vector similarity and graph relationships
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "code_search", "arguments": {"query": "AdminTab user validation", "limit": 3}}, "id": 1}' | python mcp_server.py
```

## Architecture

- **Vector Store**: ChromaDB with HuggingFace embeddings for semantic search
- **Code Graph**: NetworkX-based relationship graph for structural analysis
- **Java Parser**: Extracts imports, inheritance, and method calls
- **Hybrid Search**: Combines vector similarity with graph traversal
- **MCP Protocol**: Standard Model Context Protocol for AI assistant integration
- **Web Interface**: Flask-based visualization server for graph exploration

## Configuration

Edit paths in `ingest-code.py` to index multiple codebases:

```python
CODEBASE_PATHS = [
    "/path/to/your/codebase1",
    "/path/to/your/codebase2",
]
```

## Graph Visualization

Interactive web interface to explore code relationships:

```bash
# Start the graph visualizer
python graph_visualizer.py
```

Open http://localhost:5001 in your browser to access the **Graph Explorer**:

### Features:
- 🔍 **Real-time Search**: Instant search across 5,616 nodes (files, classes, methods)
- 📊 **Interactive Visualization**: D3.js powered graph with 19,113 relationships
- 🎯 **Smart Navigation**: Click any node to center and explore its connections
- 📈 **Detailed Analytics**: View node statistics and relationship breakdowns
- 🔧 **Interactive Controls**: Adjust exploration radius, zoom, pan, and drag nodes
- 🎨 **Color-coded Nodes**: Red (center), Teal (files), Blue (classes/methods)

### Graph Statistics:
- **Total Nodes**: 5,616 (files, classes, methods)
- **Total Edges**: 19,113 (relationships)
- **Relationship Types**: 7,265 imports, 557 inheritance, 11,291 method calls
- **Classes**: 361 across 678 files

### How to Use:
1. **Search**: Type in the search box to find files or classes
2. **Explore**: Click on any search result to visualize its relationships
3. **Navigate**: Use radius slider to control how many connection hops to show
4. **Details**: Click nodes to see detailed incoming/outgoing relationships
5. **Control**: Drag nodes, zoom, and pan to explore the graph structure

## Files

- `mcp_server.py` - Main MCP server with graph-enhanced search
- `ingest-code.py` - Codebase indexing script
- `graph_visualizer.py` - Interactive graph visualization tool
- `requirements.txt` - Python dependencies
- `chroma_db/` - Vector database storage
- `code_graph.pkl` - Serialized code relationship graph
- `templates/` - HTML templates for web interface

## Search Results

Results include both:
- **[VECTOR]** - Semantically similar code chunks
- **[GRAPH]** - Structurally related code through imports/calls/inheritance

This provides comprehensive code understanding that goes beyond simple text similarity.