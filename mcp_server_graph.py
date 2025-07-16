#!/usr/bin/env python3
"""
Graph-Enhanced MCP Server for RAG
Provides both vector similarity search and graph-based code relationship search
"""

import sys
import json
import os
from typing import Dict, Any, Optional

# Auto-activate virtual environment if not already active
def ensure_venv():
    """Ensure we're running in the virtual environment"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(script_dir, "venv")
    venv_python = os.path.join(venv_path, "bin", "python")
    
    # Check if we're already in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in venv
    
    # Check if venv exists and re-run with venv python
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])
    else:
        print(f"Error: Virtual environment not found at {venv_path}", file=sys.stderr)
        print("Please run setup_mcp.sh first", file=sys.stderr)
        sys.exit(1)

# Ensure virtual environment before importing dependencies
ensure_venv()

from graph_enhanced_rag import GraphEnhancedRAG
from langchain.schema import Document

# Paths
CHROMA_DB_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/chroma_db"
GRAPH_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/code_graph.pkl"

# Initialize the enhanced RAG system
print("Initializing Graph-Enhanced RAG system...", file=sys.stderr)
rag_system = GraphEnhancedRAG(CHROMA_DB_PATH, GRAPH_PATH)

# Build graph if it doesn't exist
if not os.path.exists(GRAPH_PATH):
    print("Building code graph from documents...", file=sys.stderr)
    docs = rag_system.vector_store.get()
    documents = []
    for i, content in enumerate(docs.get('documents', [])):
        metadata = docs.get('metadatas', [])[i] if i < len(docs.get('metadatas', [])) else {}
        documents.append(Document(page_content=content, metadata=metadata))
    
    rag_system.build_graph_from_documents(documents)
    rag_system.save_graph(GRAPH_PATH)
    print("Code graph built and saved!", file=sys.stderr)

print("Graph-Enhanced RAG system ready!", file=sys.stderr)


def handle_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP JSON-RPC request"""
    try:
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "hybrid_search",
                        "description": "Search the codebase using both vector similarity and code relationship graphs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for code"
                                },
                                "limit": {
                                    "type": "number",
                                    "description": "Maximum number of results to return",
                                    "default": 5
                                },
                                "use_graph": {
                                    "type": "boolean",
                                    "description": "Whether to use graph enhancement",
                                    "default": True
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "method_search",
                        "description": "Find all places where a specific method is called",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "method_name": {
                                    "type": "string",
                                    "description": "Name of the method to search for"
                                },
                                "limit": {
                                    "type": "number",
                                    "description": "Maximum number of results to return",
                                    "default": 5
                                }
                            },
                            "required": ["method_name"]
                        }
                    },
                    {
                        "name": "class_hierarchy",
                        "description": "Get inheritance hierarchy information for a class",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "class_name": {
                                    "type": "string",
                                    "description": "Name of the class to analyze"
                                }
                            },
                            "required": ["class_name"]
                        }
                    },
                    {
                        "name": "code_search",
                        "description": "Search the codebase for relevant code snippets using semantic similarity (legacy)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for code"
                                },
                                "limit": {
                                    "type": "number",
                                    "description": "Maximum number of results to return",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
            return {"jsonrpc": "2.0", "result": result, "id": request_id}

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "hybrid_search":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)
                use_graph = arguments.get("use_graph", True)
                
                if not query:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing query parameter"},
                        "id": request_id
                    }
                
                # Perform hybrid search
                results = rag_system.hybrid_search(query, limit, use_graph)
                
                if not results:
                    content = [{"type": "text", "text": f"No results found for query: {query}"}]
                else:
                    content = []
                    for result in results:
                        source = result['metadata'].get('source', 'unknown')
                        search_type = result.get('source', 'vector')
                        score = result.get('score', 0.0)
                        
                        content.append({
                            "type": "text",
                            "text": f"**Result {result['rank']}** [{search_type.upper()}] (score: {score:.3f}) from {source}:\n\n```java\n{result['content']}\n```\n\n**Metadata:** {result['metadata']}\n\n---\n"
                        })
                
                return {
                    "jsonrpc": "2.0",
                    "result": {"content": content, "isError": False},
                    "id": request_id
                }
            
            elif tool_name == "method_search":
                method_name = arguments.get("method_name")
                limit = arguments.get("limit", 5)
                
                if not method_name:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing method_name parameter"},
                        "id": request_id
                    }
                
                # Search for method usage
                results = rag_system.method_search(method_name, limit)
                
                if not results:
                    content = [{"type": "text", "text": f"No callers found for method: {method_name}"}]
                else:
                    content = []
                    for i, result in enumerate(results, 1):
                        source = result['metadata'].get('source', 'unknown')
                        content.append({
                            "type": "text",
                            "text": f"**Caller {i}** from {source}:\n\n```java\n{result['content']}\n```\n\n**Metadata:** {result['metadata']}\n\n---\n"
                        })
                
                return {
                    "jsonrpc": "2.0",
                    "result": {"content": content, "isError": False},
                    "id": request_id
                }
            
            elif tool_name == "class_hierarchy":
                class_name = arguments.get("class_name")
                
                if not class_name:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing class_name parameter"},
                        "id": request_id
                    }
                
                # Get class hierarchy
                hierarchy = rag_system.class_hierarchy_search(class_name)
                
                content = []
                content.append({
                    "type": "text",
                    "text": f"# Class Hierarchy for {class_name}\n\n"
                })
                
                for rel_type, classes in hierarchy.items():
                    if classes:
                        content.append({
                            "type": "text",
                            "text": f"## {rel_type.replace('_', ' ').title()}\n\n"
                        })
                        
                        for cls_info in classes:
                            content.append({
                                "type": "text",
                                "text": f"**{cls_info['class']}** (from {cls_info['file']})\n\n```java\n{cls_info['content']}\n```\n\n---\n"
                            })
                
                if not any(hierarchy.values()):
                    content.append({
                        "type": "text",
                        "text": f"No hierarchy information found for class: {class_name}"
                    })
                
                return {
                    "jsonrpc": "2.0",
                    "result": {"content": content, "isError": False},
                    "id": request_id
                }
            
            elif tool_name == "code_search":
                # Legacy code search (vector only)
                query = arguments.get("query")
                limit = arguments.get("limit", 5)
                
                if not query:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing query parameter"},
                        "id": request_id
                    }
                
                # Use vector search only
                results = rag_system.hybrid_search(query, limit, use_graph=False)
                
                if not results:
                    content = [{"type": "text", "text": f"No results found for query: {query}"}]
                else:
                    content = []
                    for result in results:
                        source = result['metadata'].get('source', 'unknown')
                        content.append({
                            "type": "text",
                            "text": f"**Result {result['rank']}** from {source}:\n\n```java\n{result['content']}\n```\n\n**Metadata:** {result['metadata']}\n\n---\n"
                        })
                
                return {
                    "jsonrpc": "2.0",
                    "result": {"content": content, "isError": False},
                    "id": request_id
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Tool {tool_name} not found"},
                    "id": request_id
                }

        elif method == "resources/list":
            try:
                docs = rag_system.vector_store.get()
                sources = set()
                metadatas = docs.get("metadatas", [])

                for metadata in metadatas:
                    if metadata and "source" in metadata:
                        source = metadata["source"]
                        if source:
                            sources.add(source)

                result = {
                    "resources": [
                        {
                            "uri": f"file://{source}",
                            "name": os.path.basename(source),
                            "description": f"Code file: {source}",
                            "mimeType": "text/plain"
                        } for source in sorted(sources)
                    ]
                }
                return {"jsonrpc": "2.0", "result": result, "id": request_id}

            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": f"Failed to list resources: {str(e)}"},
                    "id": request_id
                }

        elif method == "resources/read":
            uri = params.get("uri")
            if not uri:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Missing uri parameter"},
                    "id": request_id
                }

            # Extract file path from URI
            if uri.startswith("file://"):
                file_path = uri[7:]  # Remove "file://" prefix
            else:
                file_path = uri

            try:
                docs = rag_system.vector_store.get()
                documents = docs.get("documents", [])
                metadatas = docs.get("metadatas", [])

                # Find all chunks from this file
                file_chunks = []
                for i, metadata in enumerate(metadatas):
                    if metadata and metadata.get("source") == file_path:
                        file_chunks.append({
                            "content": documents[i],
                            "metadata": metadata
                        })

                if not file_chunks:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": f"Resource {uri} not found"},
                        "id": request_id
                    }

                # Combine all chunks from the file
                combined_content = "\n\n".join([chunk["content"] for chunk in file_chunks])

                result = {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": combined_content
                        }
                    ]
                }
                return {"jsonrpc": "2.0", "result": result, "id": request_id}

            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": f"Failed to read resource: {str(e)}"},
                    "id": request_id
                }

        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method {method} not found"},
                "id": request_id
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": f"Server error: {str(e)}"},
            "id": request_id
        }


def main():
    """Main stdio loop"""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = handle_mcp_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                    "id": None
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": f"Server error: {str(e)}"},
                    "id": None
                }
                print(json.dumps(error_response), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()