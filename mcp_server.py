#!/usr/bin/env python3
"""
Graph-Enhanced MCP Server for RAG
Combines vector similarity search with code relationship graphs
"""

import sys
import json
import os
import re
import pickle
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

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


class JavaCodeParser:
    """Parses Java code to extract relationships"""
    
    def __init__(self):
        self.patterns = {
            'import': re.compile(r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?);'),
            'package': re.compile(r'package\s+([a-zA-Z_][a-zA-Z0-9_.]*);'),
            'class': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:implements\s+([a-zA-Z_][a-zA-Z0-9_,\s]*))?\s*\{'),
            'method': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>[\],\s]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]*)?[{;]'),
            'method_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\([^)]*\)'),
        }
    
    def parse_file(self, file_path: str, content: str) -> List[CodeRelationship]:
        """Parse a Java file and extract relationships"""
        relationships = []
        lines = content.split('\n')
        
        # Extract basic info
        package_name = self._extract_package(content)
        class_info = self._extract_class_info(content, lines)
        imports = self._extract_imports(content, lines)
        
        # Add import relationships
        for imp in imports:
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file=imp['import'].replace('.', '/') + '.java',
                relationship_type='import',
                source_element=class_info.get('name', ''),
                target_element=imp['import'].split('.')[-1],
                line_number=imp['line'],
                context=imp['context']
            ))
        
        # Add inheritance relationships
        if class_info.get('extends'):
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file='',
                relationship_type='extends',
                source_element=class_info['name'],
                target_element=class_info['extends'],
                line_number=class_info['line'],
                context=class_info['context']
            ))
        
        # Add method calls
        methods = self._extract_methods(content, lines)
        for method in methods:
            method_calls = self._extract_method_calls(method['content'])
            for call in method_calls:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='calls',
                    source_element=f"{class_info.get('name', '')}.{method['name']}",
                    target_element=call['method'],
                    line_number=method['line'] + call['line_offset'],
                    context=call['context']
                ))
        
        return relationships
    
    def _extract_package(self, content: str) -> str:
        match = self.patterns['package'].search(content)
        return match.group(1) if match else ''
    
    def _extract_imports(self, content: str, lines: List[str]) -> List[Dict]:
        imports = []
        for i, line in enumerate(lines):
            match = self.patterns['import'].search(line)
            if match:
                imports.append({
                    'import': match.group(1),
                    'line': i + 1,
                    'context': line.strip()
                })
        return imports
    
    def _extract_class_info(self, content: str, lines: List[str]) -> Dict:
        for i, line in enumerate(lines):
            match = self.patterns['class'].search(line)
            if match:
                extends = match.group(2) if match.group(2) else None
                implements = []
                if match.group(3):
                    implements = [impl.strip() for impl in match.group(3).split(',')]
                
                return {
                    'name': match.group(1),
                    'extends': extends,
                    'implements': implements,
                    'line': i + 1,
                    'context': line.strip()
                }
        return {}
    
    def _extract_methods(self, content: str, lines: List[str]) -> List[Dict]:
        methods = []
        for i, line in enumerate(lines):
            match = self.patterns['method'].search(line)
            if match and not line.strip().startswith('//'):
                method_content = self._extract_method_body(lines, i)
                methods.append({
                    'name': match.group(1),
                    'line': i + 1,
                    'content': method_content
                })
        return methods
    
    def _extract_method_body(self, lines: List[str], start_line: int) -> str:
        body_lines = []
        brace_count = 0
        started = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if '{' in line:
                started = True
                brace_count += line.count('{')
            if started:
                body_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    break
        
        return '\n'.join(body_lines)
    
    def _extract_method_calls(self, method_content: str) -> List[Dict]:
        calls = []
        lines = method_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
                
            matches = self.patterns['method_call'].finditer(line)
            for match in matches:
                method_call = match.group(1)
                if self._is_valid_method_call(method_call):
                    calls.append({
                        'method': method_call,
                        'line_offset': i,
                        'context': line.strip()
                    })
        
        return calls
    
    def _is_valid_method_call(self, method_call: str) -> bool:
        keywords = {'if', 'for', 'while', 'switch', 'catch', 'new', 'this', 'super'}
        first_part = method_call.split('.')[0]
        return first_part not in keywords and not first_part.isupper()


class CodeGraph:
    """Manages the code relationship graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_to_classes = {}
        self.class_to_file = {}
        
    def add_relationships(self, relationships: List[CodeRelationship]):
        """Add relationships to the graph"""
        for rel in relationships:
            self.graph.add_node(rel.source_file, type='file')
            if rel.target_file:
                self.graph.add_node(rel.target_file, type='file')
            
            self.graph.add_edge(
                rel.source_file,
                rel.target_file or rel.target_element,
                type=rel.relationship_type,
                source_element=rel.source_element,
                target_element=rel.target_element,
                line_number=rel.line_number,
                context=rel.context
            )
            
            if rel.source_element and '.' in rel.source_element:
                class_name = rel.source_element.split('.')[0]
                self.class_to_file[class_name] = rel.source_file
                if rel.source_file not in self.file_to_classes:
                    self.file_to_classes[rel.source_file] = set()
                self.file_to_classes[rel.source_file].add(class_name)
    
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
    
    def save_graph(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'file_to_classes': self.file_to_classes,
                'class_to_file': self.class_to_file
            }, f)
    
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
        
        self.parser = JavaCodeParser()
    
    def build_graph_from_documents(self, documents: List[Document]):
        """Build code graph from documents"""
        print("Building code relationship graph...", file=sys.stderr)
        
        all_relationships = []
        for doc in documents:
            if doc.metadata.get('source', '').endswith('.java'):
                relationships = self.parser.parse_file(
                    doc.metadata['source'],
                    doc.page_content
                )
                all_relationships.extend(relationships)
        
        self.code_graph.add_relationships(all_relationships)
        print(f"Built graph with {len(all_relationships)} relationships", file=sys.stderr)
    
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
    
    def save_graph(self, file_path: str):
        """Save the code graph"""
        self.code_graph.save_graph(file_path)


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