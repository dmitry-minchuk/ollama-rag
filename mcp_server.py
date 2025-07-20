#!/usr/bin/env python3
"""
Graph-Enhanced MCP Server for RAG
Combines vector similarity search with code relationship graphs
"""

import sys
import json
import os
import pickle
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass

# Auto-activate virtual environment if not already active
def ensure_venv():
    """Ensure we're running in the virtual environment"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(script_dir, "venv")
    venv_python = os.path.join(venv_path, "bin", "python")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in venv
    
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python, __file__] + sys.argv[1:])
    else:
        print(f"Error: Virtual environment not found at {venv_path}", file=sys.stderr)
        print("Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# Ensure virtual environment before importing dependencies
ensure_venv()

import networkx as nx
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Configuration
CHROMA_DB_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/chroma_db"
GRAPH_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/code_graph.pkl"


@dataclass
class CodeRelationship:
    """Represents a relationship between code elements"""
    source_file: str
    target_file: str
    relationship_type: str
    source_element: str
    target_element: str
    line_number: int
    context: str


class CodeGraph:
    """Manages the code relationship graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_to_classes = {}
        self.class_to_file = {}
    
    def find_related_files(self, file_path: str, max_hops: int = 2) -> Set[str]:
        """Find files related to the given file within max_hops"""
        if file_path not in self.graph:
            return set()
        
        related = set()
        visited = set()
        queue = [(file_path, 0)]
        
        while queue:
            current_file, hops = queue.pop(0)
            if current_file in visited or hops > max_hops:
                continue
                
            visited.add(current_file)
            related.add(current_file)
            
            for neighbor in self.graph.neighbors(current_file):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
            
            for predecessor in self.graph.predecessors(current_file):
                if predecessor not in visited:
                    queue.append((predecessor, hops + 1))
        
        return related
    
    def load_graph(self, file_path: str):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.file_to_classes = data['file_to_classes']
            self.class_to_file = data['class_to_file']


class GraphEnhancedRAG:
    """Graph-Enhanced RAG System"""
    
    def __init__(self, chroma_db_path: str, graph_path: str = None):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="codebase",
            embedding_function=self.embeddings,
            persist_directory=chroma_db_path
        )
        
        self.code_graph = CodeGraph()
        if graph_path and os.path.exists(graph_path):
            self.code_graph.load_graph(graph_path)
    
    
    def hybrid_search(self, query: str, limit: int = 5, use_graph: bool = True) -> List[Dict]:
        """Perform hybrid vector + graph search"""
        vector_results = self.vector_store.similarity_search_with_score(query, k=limit * 2)
        
        if not use_graph:
            return self._format_results(vector_results[:limit])
        
        expanded_results = []
        seen_files = set()
        
        for doc, score in vector_results:
            source_file = doc.metadata.get('source', '')
            if source_file in seen_files:
                continue
            
            seen_files.add(source_file)
            expanded_results.append((doc, score, 'vector'))
            
            related_files = self.code_graph.find_related_files(source_file, max_hops=2)
            
            for related_file in related_files:
                if related_file != source_file and related_file not in seen_files:
                    related_docs = self._find_documents_by_file(related_file)
                    for related_doc in related_docs:
                        expanded_results.append((related_doc, score * 0.8, 'graph'))
                        seen_files.add(related_file)
                        break
        
        expanded_results.sort(key=lambda x: x[1], reverse=True)
        return self._format_results(expanded_results[:limit])
    
    def _find_documents_by_file(self, file_path: str) -> List[Document]:
        """Find all documents from a specific file"""
        try:
            docs = self.vector_store.get()
            documents = docs.get('documents', [])
            metadatas = docs.get('metadatas', [])
            
            file_docs = []
            for i, metadata in enumerate(metadatas):
                if metadata and metadata.get('source') == file_path:
                    file_docs.append(Document(
                        page_content=documents[i],
                        metadata=metadata
                    ))
            
            return file_docs
        except Exception as e:
            print(f"Error finding documents for {file_path}: {e}", file=sys.stderr)
            return []
    
    def _format_results(self, results: List[Tuple]) -> List[Dict]:
        """Format search results for output"""
        formatted = []
        for i, result in enumerate(results):
            if len(result) == 3:
                doc, score, source = result
            else:
                doc, score = result
                source = 'vector'
            
            formatted.append({
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'source': source
            })
        
        return formatted


# Initialize the enhanced RAG system
print("Initializing Graph-Enhanced RAG system...", file=sys.stderr)
rag_system = GraphEnhancedRAG(CHROMA_DB_PATH, GRAPH_PATH)

# Check if graph exists
if not os.path.exists(GRAPH_PATH):
    print("Warning: Code graph not found. Please run 'python ingest-code.py' to build the graph.", file=sys.stderr)
else:
    print("Code graph loaded successfully!", file=sys.stderr)

print("Graph-Enhanced RAG system ready!", file=sys.stderr)


def handle_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP JSON-RPC request"""
    try:
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "initialize":
            # Handle MCP initialization - required for handshake
            client_info = params.get("clientInfo", {})
            protocol_version = params.get("protocolVersion", "2024-11-05")
            
            # Log initialization for debugging
            print(f"MCP Initialize: client={client_info.get('name', 'unknown')} protocol={protocol_version}", file=sys.stderr)
            
            result = {
                "protocolVersion": protocol_version,
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "ollama-rag",
                    "version": "1.0.0"
                }
            }
            return {"jsonrpc": "2.0", "result": result, "id": request_id}

        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "code_search",
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
            
            if tool_name == "code_search":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)
                
                if not query:
                    return {
                        "jsonrpc": "2.0",
                        "error": {"code": -32602, "message": "Missing query parameter"},
                        "id": request_id
                    }
                
                # Perform hybrid search
                results = rag_system.hybrid_search(query, limit, use_graph=True)
                
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

            file_path = uri[7:] if uri.startswith("file://") else uri

            try:
                docs = rag_system.vector_store.get()
                documents = docs.get("documents", [])
                metadatas = docs.get("metadatas", [])

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