#!/usr/bin/env python3
"""
Graph-Enhanced RAG System for Code Analysis
Combines vector similarity search with code relationship graphs
"""

import os
import re
import json
import pickle
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


@dataclass
class CodeRelationship:
    """Represents a relationship between code elements"""
    source_file: str
    target_file: str
    relationship_type: str  # 'import', 'extends', 'calls', 'implements', 'uses'
    source_element: str     # class/method name
    target_element: str     # class/method name
    line_number: int
    context: str           # surrounding code context


class JavaCodeParser:
    """Parses Java code to extract relationships"""
    
    def __init__(self):
        # Java patterns for relationship extraction
        self.patterns = {
            'import': re.compile(r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?);'),
            'package': re.compile(r'package\s+([a-zA-Z_][a-zA-Z0-9_.]*);'),
            'class': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:implements\s+([a-zA-Z_][a-zA-Z0-9_,\s]*))?\s*\{'),
            'method': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>[\],\s]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]*)?[{;]'),
            'method_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\([^)]*\)'),
            'field_access': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)'),
            'annotation': re.compile(r'@([a-zA-Z_][a-zA-Z0-9_]*)')
        }
    
    def parse_file(self, file_path: str, content: str) -> List[CodeRelationship]:
        """Parse a Java file and extract relationships"""
        relationships = []
        lines = content.split('\n')
        
        # Extract package and class info
        package_name = self._extract_package(content)
        class_info = self._extract_class_info(content, lines)
        imports = self._extract_imports(content, lines)
        
        # Add import relationships
        for imp in imports:
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file=self._resolve_import_path(imp['import']),
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
                target_file='',  # Will be resolved later
                relationship_type='extends',
                source_element=class_info['name'],
                target_element=class_info['extends'],
                line_number=class_info['line'],
                context=class_info['context']
            ))
        
        # Add implementation relationships
        for interface in class_info.get('implements', []):
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file='',  # Will be resolved later
                relationship_type='implements',
                source_element=class_info['name'],
                target_element=interface.strip(),
                line_number=class_info['line'],
                context=class_info['context']
            ))
        
        # Extract method calls and field accesses
        methods = self._extract_methods(content, lines)
        for method in methods:
            method_calls = self._extract_method_calls(method['content'])
            for call in method_calls:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',  # Will be resolved later
                    relationship_type='calls',
                    source_element=f"{class_info.get('name', '')}.{method['name']}",
                    target_element=call['method'],
                    line_number=method['line'] + call['line_offset'],
                    context=call['context']
                ))
        
        return relationships
    
    def _extract_package(self, content: str) -> str:
        """Extract package name from content"""
        match = self.patterns['package'].search(content)
        return match.group(1) if match else ''
    
    def _extract_imports(self, content: str, lines: List[str]) -> List[Dict]:
        """Extract import statements"""
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
        """Extract class information"""
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
        """Extract method information"""
        methods = []
        for i, line in enumerate(lines):
            match = self.patterns['method'].search(line)
            if match and not line.strip().startswith('//'):
                # Extract method body (simplified)
                method_content = self._extract_method_body(lines, i)
                methods.append({
                    'name': match.group(1),
                    'line': i + 1,
                    'content': method_content
                })
        return methods
    
    def _extract_method_body(self, lines: List[str], start_line: int) -> str:
        """Extract method body content"""
        # Simple implementation - could be improved with proper parsing
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
        """Extract method calls from method content"""
        calls = []
        lines = method_content.split('\n')
        
        for i, line in enumerate(lines):
            # Skip comments and strings
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
                
            matches = self.patterns['method_call'].finditer(line)
            for match in matches:
                method_call = match.group(1)
                # Filter out common Java keywords and constructors
                if not self._is_valid_method_call(method_call):
                    continue
                    
                calls.append({
                    'method': method_call,
                    'line_offset': i,
                    'context': line.strip()
                })
        
        return calls
    
    def _is_valid_method_call(self, method_call: str) -> bool:
        """Check if method call is valid and not a keyword"""
        keywords = {'if', 'for', 'while', 'switch', 'catch', 'new', 'this', 'super'}
        first_part = method_call.split('.')[0]
        return first_part not in keywords and not first_part.isupper()
    
    def _resolve_import_path(self, import_name: str) -> str:
        """Resolve import to file path (simplified)"""
        # This is a simplified version - in a real implementation,
        # you'd need to resolve against the actual classpath
        return import_name.replace('.', '/') + '.java'


class CodeGraph:
    """Manages the code relationship graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_to_classes = {}  # file_path -> set of class names
        self.class_to_file = {}    # class name -> file_path
        self.method_to_class = {}  # method name -> class name
        
    def add_relationships(self, relationships: List[CodeRelationship]):
        """Add relationships to the graph"""
        for rel in relationships:
            # Add nodes
            self.graph.add_node(rel.source_file, type='file')
            if rel.target_file:
                self.graph.add_node(rel.target_file, type='file')
            
            # Add relationship edge
            self.graph.add_edge(
                rel.source_file,
                rel.target_file or rel.target_element,
                type=rel.relationship_type,
                source_element=rel.source_element,
                target_element=rel.target_element,
                line_number=rel.line_number,
                context=rel.context
            )
            
            # Update lookup tables
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
        
        # BFS to find related nodes
        visited = set()
        queue = [(file_path, 0)]
        
        while queue:
            current_file, hops = queue.pop(0)
            if current_file in visited or hops > max_hops:
                continue
                
            visited.add(current_file)
            related.add(current_file)
            
            # Add neighbors
            for neighbor in self.graph.neighbors(current_file):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
            
            # Add predecessors
            for predecessor in self.graph.predecessors(current_file):
                if predecessor not in visited:
                    queue.append((predecessor, hops + 1))
        
        return related
    
    def find_method_callers(self, method_name: str) -> List[str]:
        """Find all files that call the given method"""
        callers = []
        for u, v, data in self.graph.edges(data=True):
            if (data.get('type') == 'calls' and 
                data.get('target_element') == method_name):
                callers.append(u)
        return callers
    
    def find_class_hierarchy(self, class_name: str) -> Dict[str, List[str]]:
        """Find inheritance hierarchy for a class"""
        hierarchy = {
            'extends': [],
            'implements': [],
            'extended_by': [],
            'implemented_by': []
        }
        
        class_file = self.class_to_file.get(class_name)
        if not class_file:
            return hierarchy
        
        # Find what this class extends/implements
        for u, v, data in self.graph.edges(data=True):
            if data.get('source_element') == class_name:
                if data.get('type') == 'extends':
                    hierarchy['extends'].append(data.get('target_element'))
                elif data.get('type') == 'implements':
                    hierarchy['implements'].append(data.get('target_element'))
        
        # Find what extends/implements this class
        for u, v, data in self.graph.edges(data=True):
            if data.get('target_element') == class_name:
                if data.get('type') == 'extends':
                    hierarchy['extended_by'].append(data.get('source_element'))
                elif data.get('type') == 'implements':
                    hierarchy['implemented_by'].append(data.get('source_element'))
        
        return hierarchy
    
    def save_graph(self, file_path: str):
        """Save graph to file"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'file_to_classes': self.file_to_classes,
                'class_to_file': self.class_to_file,
                'method_to_class': self.method_to_class
            }, f)
    
    def load_graph(self, file_path: str):
        """Load graph from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.file_to_classes = data['file_to_classes']
            self.class_to_file = data['class_to_file']
            self.method_to_class = data['method_to_class']


class GraphEnhancedRAG:
    """Graph-Enhanced RAG System"""
    
    def __init__(self, chroma_db_path: str, graph_path: str = None):
        # Initialize vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="codebase",
            embedding_function=self.embeddings,
            persist_directory=chroma_db_path
        )
        
        # Initialize code graph
        self.code_graph = CodeGraph()
        if graph_path and os.path.exists(graph_path):
            self.code_graph.load_graph(graph_path)
        
        # Initialize parser
        self.parser = JavaCodeParser()
    
    def build_graph_from_documents(self, documents: List[Document]):
        """Build code graph from documents"""
        print("Building code relationship graph...")
        
        all_relationships = []
        for doc in documents:
            if doc.metadata.get('source', '').endswith('.java'):
                relationships = self.parser.parse_file(
                    doc.metadata['source'],
                    doc.page_content
                )
                all_relationships.extend(relationships)
        
        self.code_graph.add_relationships(all_relationships)
        print(f"Built graph with {len(all_relationships)} relationships")
    
    def hybrid_search(self, query: str, limit: int = 5, use_graph: bool = True) -> List[Dict]:
        """Perform hybrid vector + graph search"""
        # Step 1: Vector similarity search
        vector_results = self.vector_store.similarity_search_with_score(query, k=limit * 2)
        
        if not use_graph:
            return self._format_results(vector_results[:limit])
        
        # Step 2: Graph expansion
        expanded_results = []
        seen_files = set()
        
        for doc, score in vector_results:
            source_file = doc.metadata.get('source', '')
            if source_file in seen_files:
                continue
            
            seen_files.add(source_file)
            
            # Add original result
            expanded_results.append((doc, score, 'vector'))
            
            # Find related files through graph
            related_files = self.code_graph.find_related_files(source_file, max_hops=2)
            
            for related_file in related_files:
                if related_file != source_file and related_file not in seen_files:
                    # Find documents from related file
                    related_docs = self._find_documents_by_file(related_file)
                    for related_doc in related_docs:
                        expanded_results.append((related_doc, score * 0.8, 'graph'))
                        seen_files.add(related_file)
                        break  # Only add one document per related file
        
        # Step 3: Sort by score and return top results
        expanded_results.sort(key=lambda x: x[1], reverse=True)
        return self._format_results(expanded_results[:limit])
    
    def method_search(self, method_name: str, limit: int = 5) -> List[Dict]:
        """Search for method usage across codebase"""
        # Find files that call this method
        caller_files = self.code_graph.find_method_callers(method_name)
        
        results = []
        for file_path in caller_files[:limit]:
            docs = self._find_documents_by_file(file_path)
            for doc in docs:
                if method_name in doc.page_content:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source': 'method_search',
                        'score': 1.0
                    })
                    break
        
        return results
    
    def class_hierarchy_search(self, class_name: str) -> Dict:
        """Get class hierarchy information"""
        hierarchy = self.code_graph.find_class_hierarchy(class_name)
        
        # Find documents for each class in hierarchy
        hierarchy_docs = {}
        for rel_type, classes in hierarchy.items():
            hierarchy_docs[rel_type] = []
            for cls in classes:
                class_file = self.code_graph.class_to_file.get(cls)
                if class_file:
                    docs = self._find_documents_by_file(class_file)
                    if docs:
                        hierarchy_docs[rel_type].append({
                            'class': cls,
                            'file': class_file,
                            'content': docs[0].page_content[:500] + '...'
                        })
        
        return hierarchy_docs
    
    def _find_documents_by_file(self, file_path: str) -> List[Document]:
        """Find all documents from a specific file"""
        # This is a simplified implementation
        # In practice, you'd want to query ChromaDB with file path filter
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
            print(f"Error finding documents for {file_path}: {e}")
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


if __name__ == "__main__":
    # Example usage
    CHROMA_DB_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/chroma_db"
    GRAPH_PATH = "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag/code_graph.pkl"
    
    # Initialize the enhanced RAG system
    rag = GraphEnhancedRAG(CHROMA_DB_PATH)
    
    # Build graph from existing documents
    docs = rag.vector_store.get()
    documents = []
    for i, content in enumerate(docs.get('documents', [])):
        metadata = docs.get('metadatas', [])[i] if i < len(docs.get('metadatas', [])) else {}
        documents.append(Document(page_content=content, metadata=metadata))
    
    rag.build_graph_from_documents(documents)
    
    # Save the graph
    rag.save_graph(GRAPH_PATH)
    
    print("Graph-Enhanced RAG system initialized and graph built!")