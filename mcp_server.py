#!/usr/bin/env python3
"""
Graph-Enhanced MCP Server for RAG
Combines vector similarity search with code relationship graphs
"""

import sys
import json
import os
import pickle
import re
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum

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


class QueryType(Enum):
    """Types of queries for specialized handling"""
    INSTANTIATION = "instantiation"
    INHERITANCE = "inheritance"
    GENERAL = "general"


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


class QueryClassifier:
    """Classifies queries by intent to enable specialized handling"""
    
    def __init__(self):
        self.instantiation_patterns = [
            r'\bnew\b',
            r'\binstantiate\b',
            r'\binstantiation\b', 
            r'\bconstructor\b',
            r'\bcreate\b',
            r'\bcreating\b',
            r'\bassign\b',
            r'\bassignment\b',
            r'\binitialize\b',
            r'[A-Z][a-zA-Z]*(?:Component|Page|Element|Class)',  # Class names
        ]
        
        self.inheritance_patterns = [
            r'\bextends\b',
            r'\bimplements\b',
            r'\binherits\b',
            r'\binheritance\b',
            r'\bparent\b',
            r'\bsuperclass\b',
            r'\bbase\s+class\b',
            r'\bderived\b',
            r'\bsubclass\b',
        ]
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query type based on content analysis"""
        query_lower = query.lower()
        
        # Check for instantiation patterns
        for pattern in self.instantiation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.INSTANTIATION
        
        # Check for inheritance patterns  
        for pattern in self.inheritance_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.INHERITANCE
        
        # Default to general query
        return QueryType.GENERAL


class RelationshipScoring:
    """Configurable relationship scoring weights for different query types"""
    
    def __init__(self):
        # Relationship importance weights by query type
        self.scoring_weights = {
            QueryType.INSTANTIATION: {
                'instantiates': 3.0,
                'assigns': 2.5,
                'calls': 1.0,
                'import': 0.8,
                'extends': 0.5,
                'implements': 0.5,
                'inherits': 0.5
            },
            QueryType.INHERITANCE: {
                'extends': 3.0,
                'implements': 2.5,
                'inherits': 2.5,
                'import': 1.0,
                'instantiates': 0.8,
                'assigns': 0.8,
                'calls': 0.5
            },
            QueryType.GENERAL: {
                'import': 1.0,
                'calls': 1.0,
                'extends': 1.0,
                'implements': 1.0,
                'inherits': 1.0,
                'instantiates': 1.0,
                'assigns': 1.0
            }
        }
    
    def get_relationship_weight(self, query_type: QueryType, relationship_type: str) -> float:
        """Get the scoring weight for a relationship type given the query type"""
        weights = self.scoring_weights.get(query_type, self.scoring_weights[QueryType.GENERAL])
        return weights.get(relationship_type, 1.0)
    
    def get_graph_boost_factor(self, query_type: QueryType) -> float:
        """Get the boost factor for graph results based on query type"""
        boost_factors = {
            QueryType.INSTANTIATION: 1.5,  # Boost graph results for instantiation queries
            QueryType.INHERITANCE: 1.4,    # Boost graph results for inheritance queries  
            QueryType.GENERAL: 1.0         # No boost for general queries
        }
        return boost_factors.get(query_type, 1.0)


class CodeGraph:
    """Manages the code relationship graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_to_classes = {}
        self.class_to_file = {}
        self.relationship_scorer = RelationshipScoring()
    
    def find_related_files(self, file_path: str, query_type: QueryType = QueryType.GENERAL, max_hops: int = 2) -> List[Tuple[str, float, str, List[Dict]]]:
        """Find files related to the given file with relationship-type-aware scoring"""
        if file_path not in self.graph:
            return []
        
        # Priority queue: (priority, current_file, hops, relationship_chain)
        from heapq import heappush, heappop
        
        related_results = []
        visited = set()
        priority_queue = [(0, file_path, 0, [])]  # Start with priority 0 (highest)
        
        while priority_queue:
            neg_priority, current_file, hops, relationship_chain = heappop(priority_queue)
            priority = -neg_priority  # Convert back to positive
            
            if current_file in visited or hops > max_hops:
                continue
                
            visited.add(current_file)
            
            # Add current file to results (skip the source file itself)
            if hops > 0:
                related_results.append((current_file, priority, 'graph', relationship_chain.copy()))
            
            # Explore neighbors (outgoing edges)
            for neighbor in self.graph.neighbors(current_file):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(current_file, neighbor, {})
                    relationship_type = edge_data.get('type', 'unknown')
                    
                    # Calculate relationship weight based on query type
                    weight = self.relationship_scorer.get_relationship_weight(query_type, relationship_type)
                    
                    # Build relationship chain info
                    chain_info = {
                        'from': current_file,
                        'to': neighbor,
                        'relationship': relationship_type,
                        'weight': weight,
                        'context': edge_data.get('context', ''),
                        'line_number': edge_data.get('line_number', 0)
                    }
                    
                    new_chain = relationship_chain + [chain_info]
                    new_priority = priority + weight
                    
                    heappush(priority_queue, (-new_priority, neighbor, hops + 1, new_chain))
            
            # Explore predecessors (incoming edges)  
            for predecessor in self.graph.predecessors(current_file):
                if predecessor not in visited:
                    edge_data = self.graph.get_edge_data(predecessor, current_file, {})
                    relationship_type = edge_data.get('type', 'unknown')
                    
                    # Calculate relationship weight based on query type
                    weight = self.relationship_scorer.get_relationship_weight(query_type, relationship_type)
                    
                    # Build relationship chain info
                    chain_info = {
                        'from': predecessor,
                        'to': current_file,
                        'relationship': relationship_type,
                        'weight': weight,
                        'context': edge_data.get('context', ''),
                        'line_number': edge_data.get('line_number', 0)
                    }
                    
                    new_chain = relationship_chain + [chain_info]
                    new_priority = priority + weight
                    
                    heappush(priority_queue, (-new_priority, predecessor, hops + 1, new_chain))
        
        # Sort by priority (highest first) and return
        related_results.sort(key=lambda x: x[1], reverse=True)
        return related_results
    
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
        
        self.query_classifier = QueryClassifier()
        self.relationship_scorer = RelationshipScoring()
    
    
    def hybrid_search(self, query: str, limit: int = 5, use_graph: bool = True) -> Dict[str, List[Dict]]:
        """Perform adaptive hybrid vector + graph search with multi-modal results"""
        
        # Step 1: Classify query type
        query_type = self.query_classifier.classify_query(query)
        
        # Step 2: Get vector results
        vector_results = self.vector_store.similarity_search_with_score(query, k=limit * 2)
        
        if not use_graph:
            return {
                'direct_matches': self._format_results(vector_results[:limit], 'vector'),
                'related_code': [],
                'query_type': query_type.value,
                'total_results': len(vector_results[:limit])
            }
        
        # Step 3: Prepare results containers
        direct_matches = []  # Vector similarity results
        related_code = []    # Graph relationship results
        seen_files = set()
        
        # Step 4: Process vector results and find graph relationships
        for doc, vector_score in vector_results[:limit]:
            source_file = doc.metadata.get('source', '')
            if source_file in seen_files:
                continue
            
            seen_files.add(source_file)
            
            # Add to direct matches
            direct_matches.append((doc, vector_score, 'vector', None))
            
            # Find related files using relationship-aware traversal
            related_files_data = self.code_graph.find_related_files(
                source_file, 
                query_type=query_type, 
                max_hops=2
            )
            
            # Step 5: Apply adaptive scoring for graph results
            graph_boost_factor = self.relationship_scorer.get_graph_boost_factor(query_type)
            
            for related_file, relationship_priority, source_type, relationship_chain in related_files_data:
                if related_file != source_file and related_file not in seen_files:
                    related_docs = self._find_documents_by_file(related_file)
                    
                    for related_doc in related_docs:
                        # Adaptive scoring: base vector score + relationship priority boost
                        adaptive_score = (vector_score * 0.7) + (relationship_priority * graph_boost_factor * 0.3)
                        
                        related_code.append((related_doc, adaptive_score, 'graph', {
                            'relationship_chain': relationship_chain,
                            'relationship_priority': relationship_priority,
                            'base_vector_score': vector_score,
                            'adaptive_score': adaptive_score
                        }))
                        seen_files.add(related_file)
                        break
        
        # Step 6: Sort and limit results
        related_code.sort(key=lambda x: x[1], reverse=True)
        related_code = related_code[:limit]
        
        # Step 7: Format and return multi-modal results
        return {
            'direct_matches': self._format_results(direct_matches, 'vector'),
            'related_code': self._format_results(related_code, 'graph'),
            'query_type': query_type.value,
            'total_results': len(direct_matches) + len(related_code)
        }
    
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
    
    def _format_results(self, results: List[Tuple], result_type: str) -> List[Dict]:
        """Format search results for output with enhanced relationship context"""
        formatted = []
        for i, result in enumerate(results):
            if len(result) >= 4:
                doc, score, source, extra_info = result[:4]
            elif len(result) == 3:
                doc, score, source = result
                extra_info = None
            else:
                doc, score = result
                source = 'vector'
                extra_info = None
            
            result_entry = {
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'source': source,
                'result_type': result_type
            }
            
            # Add relationship context for graph results
            if extra_info and result_type == 'graph':
                relationship_chain = extra_info.get('relationship_chain', [])
                if relationship_chain:
                    # Format relationship chain information
                    chain_summary = []
                    for chain_link in relationship_chain:
                        chain_summary.append({
                            'relationship': chain_link.get('relationship', 'unknown'),
                            'weight': chain_link.get('weight', 1.0),
                            'context': chain_link.get('context', ''),
                            'line_number': chain_link.get('line_number', 0)
                        })
                    
                    result_entry['relationship_context'] = {
                        'chain': chain_summary,
                        'priority': extra_info.get('relationship_priority', 0),
                        'base_vector_score': extra_info.get('base_vector_score', 0),
                        'adaptive_score': extra_info.get('adaptive_score', score)
                    }
            
            formatted.append(result_entry)
        
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
                search_results = rag_system.hybrid_search(query, limit, use_graph=True)
                
                if not search_results or (not search_results['direct_matches'] and not search_results['related_code']):
                    content = [{"type": "text", "text": f"No results found for query: {query}"}]
                else:
                    content = []
                    
                    # Add query type information
                    query_type = search_results.get('query_type', 'general')
                    content.append({
                        "type": "text",
                        "text": f"**Query Type:** {query_type.upper()}\n**Total Results:** {search_results.get('total_results', 0)}\n\n"
                    })
                    
                    # Add direct matches section
                    direct_matches = search_results.get('direct_matches', [])
                    if direct_matches:
                        content.append({
                            "type": "text", 
                            "text": "## 🎯 Direct Matches (Vector Similarity)\n\n"
                        })
                        
                        for result in direct_matches:
                            source = result['metadata'].get('source', 'unknown')
                            score = result.get('score', 0.0)
                            
                            content.append({
                                "type": "text",
                                "text": f"**Match {result['rank']}** (score: {score:.3f}) from {source}:\n\n```java\n{result['content']}\n```\n\n---\n"
                            })
                    
                    # Add related code section  
                    related_code = search_results.get('related_code', [])
                    if related_code:
                        content.append({
                            "type": "text",
                            "text": "\n## 🔗 Related Code (Graph Relationships)\n\n"
                        })
                        
                        for result in related_code:
                            source = result['metadata'].get('source', 'unknown')
                            score = result.get('score', 0.0)
                            
                            # Build relationship context string
                            relationship_info = ""
                            if 'relationship_context' in result:
                                rel_ctx = result['relationship_context']
                                chain = rel_ctx.get('chain', [])
                                if chain:
                                    relationships = [f"{link['relationship']} (weight: {link['weight']:.1f})" for link in chain]
                                    relationship_info = f"\n**Relationships:** {' → '.join(relationships)}"
                                    if rel_ctx.get('priority', 0) > 0:
                                        relationship_info += f"\n**Priority:** {rel_ctx['priority']:.2f}"
                            
                            content.append({
                                "type": "text",
                                "text": f"**Related {result['rank']}** (score: {score:.3f}) from {source}:{relationship_info}\n\n```java\n{result['content']}\n```\n\n---\n"
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