# Graph-Enhanced RAG System

## Overview

This is a graph-enhanced RAG (Retrieval-Augmented Generation) system that combines vector similarity search with code relationship graphs for more contextually aware code search.

## Architecture

### Core Components

1. **Vector Store (ChromaDB)**
   - Stores 10,195 text chunks with embeddings
   - Uses HuggingFace `all-MiniLM-L6-v2` model
   - Provides semantic similarity search

2. **Code Graph (NetworkX)**
   - Analyzes 223,890 code relationships
   - Tracks imports, inheritance, method calls
   - Enables structural code navigation

3. **Hybrid Search Engine**
   - Combines vector similarity with graph traversal
   - Provides both semantic and structural results
   - Re-ranks results based on multiple signals

## Features

### Enhanced Search Tools

1. **`hybrid_search`** - Primary search combining vector + graph
   - Finds semantically similar code chunks
   - Expands results with related files through code relationships
   - Provides both VECTOR and GRAPH result types

2. **`method_search`** - Find method callers across codebase
   - Discovers all places where a method is called
   - Follows method call relationships in the graph

3. **`class_hierarchy`** - Class inheritance analysis
   - Shows extends/implements relationships
   - Finds parent and child classes

4. **`code_search`** - Legacy vector-only search
   - Backwards compatibility with original system
   - Pure semantic similarity search

### Code Relationship Types

- **`import`** - Package/class imports
- **`extends`** - Class inheritance
- **`implements`** - Interface implementation
- **`calls`** - Method invocations
- **`uses`** - Field/variable references

## Usage

### Switch Between Systems

```bash
# Switch to graph-enhanced system
python switch_server.py graph

# Switch back to original vector-only system
python switch_server.py original
```

### Search Examples

```bash
# Hybrid search (vector + graph)
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "hybrid_search", "arguments": {"query": "AdminTab email validation", "limit": 3}}, "id": 1}' | python mcp_server.py

# Method search
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "method_search", "arguments": {"method_name": "verifyUserInfoInTable", "limit": 2}}, "id": 2}' | python mcp_server.py

# Class hierarchy
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "class_hierarchy", "arguments": {"class_name": "BaseTest"}}, "id": 3}' | python mcp_server.py
```

## Files

### Core Implementation
- `graph_enhanced_rag.py` - Main graph-enhanced RAG system
- `mcp_server_graph.py` - Graph-enhanced MCP server
- `switch_server.py` - Utility to switch between systems

### Data Files
- `chroma_db/` - Vector database storage
- `code_graph.pkl` - Serialized code relationship graph
- `mcp_server_original.py` - Backup of original server

### Configuration
- `ingest-code.py` - Updated to support multiple source paths
- `requirements.txt` - Dependencies including NetworkX

## Benefits Over Vector-Only Search

1. **Structural Awareness**
   - Finds related code through imports and inheritance
   - Discovers method call chains and dependencies
   - Understanding of code architecture

2. **Better Context**
   - Related files are included in results
   - Multi-hop relationships reveal broader context
   - Distinguishes between semantic similarity and structural relevance

3. **Precise Results**
   - Combines semantic matching with code structure
   - Filters noise by actual code relationships
   - Provides type-specific searches (method callers, class hierarchy)

## Performance

- **Graph Building**: 223,890 relationships processed
- **Search Speed**: Near-instant for hybrid queries
- **Memory Usage**: Graph cached in memory for fast access
- **Scalability**: Linear with codebase size

## Future Enhancements

1. **Advanced Graph Features**
   - Control flow analysis
   - Data flow tracking
   - Cross-references and usage patterns

2. **Improved Parsing**
   - Better Java AST parsing
   - Support for more languages
   - Annotation and metadata extraction

3. **Query Optimization**
   - Graph-based query planning
   - Caching of common searches
   - Parallel graph traversal

## Technical Details

### Graph Schema
```python
# Node types: 'file', 'class', 'method'
# Edge types: 'import', 'extends', 'implements', 'calls', 'uses'

# Example relationship:
CodeRelationship(
    source_file="/path/to/TestClass.java",
    target_file="/path/to/BaseTest.java", 
    relationship_type="extends",
    source_element="TestClass",
    target_element="BaseTest",
    line_number=15,
    context="public class TestClass extends BaseTest {"
)
```

### Search Algorithm
1. **Vector Search**: Find semantically similar chunks
2. **Graph Expansion**: Discover related files via code relationships
3. **Scoring**: Combine semantic similarity with structural relevance
4. **Ranking**: Sort by composite score and return top results

This system provides a significant improvement in code search capabilities, especially for large codebases where understanding code relationships is crucial for effective development and maintenance.