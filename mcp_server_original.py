#!/usr/bin/env python3
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

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB path
CHROMA_DB_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/chroma_db"
COLLECTION_NAME = "codebase"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH
)

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
                        "name": "code_search",
                        "description": "Search the codebase for relevant code snippets using semantic similarity",
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
            if tool_name != "code_search":
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Tool {tool_name} not found"},
                    "id": request_id
                }

            arguments = params.get("arguments", {})
            query = arguments.get("query")
            limit = arguments.get("limit", 5)

            if not query:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32602, "message": "Missing query parameter in arguments"},
                    "id": request_id
                }

            # Search chunks in ChromaDB
            docs = vector_store.similarity_search(query, k=limit)

            if not docs:
                content = [{"type": "text", "text": f"No results found for query: {query}"}]
            else:
                content = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "unknown")
                    content.append({
                        "type": "text",
                        "text": f"**Result {i}** (from {source}):\n\n```\n{doc.page_content}\n```\n\n**Metadata:** {doc.metadata}\n\n---\n"
                    })

            result = {
                "content": content,
                "isError": False
            }
            return {"jsonrpc": "2.0", "result": result, "id": request_id}

        elif method == "resources/list":
            try:
                docs = vector_store.get()
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
                docs = vector_store.get()
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
