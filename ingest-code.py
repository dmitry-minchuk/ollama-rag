import os
import re
import pickle
import networkx as nx
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

# Configuration - Paths to codebases for indexing
CODEBASE_PATHS = [
#    "/Users/dmitryminchuk/Projects/eis/openl-tests",
#    "/Users/dmitryminchuk/Projects/java-taf-template",
#    "/Users/dmitryminchuk/Projects/python-taf-bp",
#    "/Users/dmitryminchuk/Projects/ai/mcp/ollama-rag",
    "/Users/dmitryminchuk/Projects/trading-app",
]

# Folders to exclude from indexing
EXCLUDE_FOLDERS = [
    "node_modules",
    "venv",
    "target",
    "__pycache__",
    "dist",
    "build",
    ".git"
]


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


class PythonCodeParser:
    """Parses Python code to extract relationships"""
    
    def __init__(self):
        self.patterns = {
            'import': re.compile(r'(?:from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+)?import\s+([a-zA-Z_][a-zA-Z0-9_.,\s*]+)'),
            'class': re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s*:'),
            'function': re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->[^:]*)?:'),
            'function_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\([^)]*\)'),
        }
    
    def parse_file(self, file_path: str, content: str) -> List[CodeRelationship]:
        """Parse a Python file and extract relationships"""
        relationships = []
        lines = content.split('\n')
        
        # Extract imports
        imports = self._extract_imports(content, lines)
        for imp in imports:
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file=imp['module'].replace('.', '/') + '.py',
                relationship_type='import',
                source_element='',
                target_element=imp['name'],
                line_number=imp['line'],
                context=imp['context']
            ))
        
        # Extract class inheritance
        classes = self._extract_classes(content, lines)
        for cls in classes:
            if cls.get('inherits'):
                for parent in cls['inherits']:
                    relationships.append(CodeRelationship(
                        source_file=file_path,
                        target_file='',
                        relationship_type='inherits',
                        source_element=cls['name'],
                        target_element=parent,
                        line_number=cls['line'],
                        context=cls['context']
                    ))
        
        # Extract function calls
        functions = self._extract_functions(content, lines)
        for func in functions:
            function_calls = self._extract_function_calls(func['content'])
            for call in function_calls:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='calls',
                    source_element=func['name'],
                    target_element=call['function'],
                    line_number=func['line'] + call['line_offset'],
                    context=call['context']
                ))
        
        return relationships
    
    def _extract_imports(self, content: str, lines: List[str]) -> List[Dict]:
        imports = []
        for i, line in enumerate(lines):
            match = self.patterns['import'].search(line.strip())
            if match:
                module = match.group(1) or ''
                imported_items = match.group(2)
                for item in imported_items.split(','):
                    item = item.strip().split(' as ')[0]
                    imports.append({
                        'module': module,
                        'name': item,
                        'line': i + 1,
                        'context': line.strip()
                    })
        return imports
    
    def _extract_classes(self, content: str, lines: List[str]) -> List[Dict]:
        classes = []
        for i, line in enumerate(lines):
            match = self.patterns['class'].search(line.strip())
            if match:
                name = match.group(1)
                parents = []
                if match.group(2):
                    parents = [p.strip() for p in match.group(2).split(',')]
                classes.append({
                    'name': name,
                    'inherits': parents,
                    'line': i + 1,
                    'context': line.strip()
                })
        return classes
    
    def _extract_functions(self, content: str, lines: List[str]) -> List[Dict]:
        functions = []
        for i, line in enumerate(lines):
            match = self.patterns['function'].search(line.strip())
            if match:
                function_content = self._extract_function_body(lines, i)
                functions.append({
                    'name': match.group(1),
                    'line': i + 1,
                    'content': function_content
                })
        return functions
    
    def _extract_function_body(self, lines: List[str], start_line: int) -> str:
        body_lines = []
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                break
            body_lines.append(line)
        
        return '\n'.join(body_lines)
    
    def _extract_function_calls(self, function_content: str) -> List[Dict]:
        calls = []
        lines = function_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                continue
            
            matches = self.patterns['function_call'].finditer(line)
            for match in matches:
                function_call = match.group(1)
                if self._is_valid_function_call(function_call):
                    calls.append({
                        'function': function_call,
                        'line_offset': i,
                        'context': line.strip()
                    })
        
        return calls
    
    def _is_valid_function_call(self, function_call: str) -> bool:
        keywords = {'if', 'for', 'while', 'with', 'try', 'except', 'class', 'def', 'return', 'yield'}
        first_part = function_call.split('.')[0]
        return first_part not in keywords and not function_call.startswith('__')


class JavaScriptCodeParser:
    """Parses JavaScript code to extract relationships"""
    
    def __init__(self):
        self.patterns = {
            'import': re.compile(r'import\s+(?:\{([^}]+)\}|\*\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)|([a-zA-Z_][a-zA-Z0-9_]*))(?:\s*,\s*\{([^}]+)\})?\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'require': re.compile(r'(?:const|let|var)\s+(?:\{([^}]+)\}|([a-zA-Z_][a-zA-Z0-9_]*))?\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'),
            'class': re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+extends\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s*\{'),
            'function': re.compile(r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(?:function\s*\(|async\s+function\s*\(|\([^)]*\)\s*=>))'),
            'function_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\([^)]*\)'),
        }
    
    def parse_file(self, file_path: str, content: str) -> List[CodeRelationship]:
        """Parse a JavaScript file and extract relationships"""
        relationships = []
        lines = content.split('\n')
        
        # Extract imports
        imports = self._extract_imports(content, lines)
        for imp in imports:
            relationships.append(CodeRelationship(
                source_file=file_path,
                target_file=imp['module'],
                relationship_type='import',
                source_element='',
                target_element=imp['name'],
                line_number=imp['line'],
                context=imp['context']
            ))
        
        # Extract class inheritance
        classes = self._extract_classes(content, lines)
        for cls in classes:
            if cls.get('extends'):
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='extends',
                    source_element=cls['name'],
                    target_element=cls['extends'],
                    line_number=cls['line'],
                    context=cls['context']
                ))
        
        # Extract function calls
        functions = self._extract_functions(content, lines)
        for func in functions:
            function_calls = self._extract_function_calls(func['content'])
            for call in function_calls:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='calls',
                    source_element=func['name'],
                    target_element=call['function'],
                    line_number=func['line'] + call['line_offset'],
                    context=call['context']
                ))
        
        return relationships
    
    def _extract_imports(self, content: str, lines: List[str]) -> List[Dict]:
        imports = []
        for i, line in enumerate(lines):
            # ES6 imports
            import_match = self.patterns['import'].search(line.strip())
            if import_match:
                module = import_match.group(5)
                names = []
                if import_match.group(1):  # named imports
                    names.extend([n.strip() for n in import_match.group(1).split(',')])
                if import_match.group(2):  # namespace import
                    names.append(import_match.group(2))
                if import_match.group(3):  # default import
                    names.append(import_match.group(3))
                if import_match.group(4):  # additional named imports
                    names.extend([n.strip() for n in import_match.group(4).split(',')])
                
                for name in names:
                    imports.append({
                        'module': module,
                        'name': name,
                        'line': i + 1,
                        'context': line.strip()
                    })
            
            # CommonJS require
            require_match = self.patterns['require'].search(line.strip())
            if require_match:
                module = require_match.group(3)
                name = require_match.group(2) or require_match.group(1) or module
                imports.append({
                    'module': module,
                    'name': name,
                    'line': i + 1,
                    'context': line.strip()
                })
        
        return imports
    
    def _extract_classes(self, content: str, lines: List[str]) -> List[Dict]:
        classes = []
        for i, line in enumerate(lines):
            match = self.patterns['class'].search(line.strip())
            if match:
                classes.append({
                    'name': match.group(1),
                    'extends': match.group(2),
                    'line': i + 1,
                    'context': line.strip()
                })
        return classes
    
    def _extract_functions(self, content: str, lines: List[str]) -> List[Dict]:
        functions = []
        for i, line in enumerate(lines):
            match = self.patterns['function'].search(line.strip())
            if match:
                name = match.group(1) or match.group(2) or 'anonymous'
                function_content = self._extract_function_body(lines, i)
                functions.append({
                    'name': name,
                    'line': i + 1,
                    'content': function_content
                })
        return functions
    
    def _extract_function_body(self, lines: List[str], start_line: int) -> str:
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
    
    def _extract_function_calls(self, function_content: str) -> List[Dict]:
        calls = []
        lines = function_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
            
            matches = self.patterns['function_call'].finditer(line)
            for match in matches:
                function_call = match.group(1)
                if self._is_valid_function_call(function_call):
                    calls.append({
                        'function': function_call,
                        'line_offset': i,
                        'context': line.strip()
                    })
        
        return calls
    
    def _is_valid_function_call(self, function_call: str) -> bool:
        keywords = {'if', 'for', 'while', 'switch', 'catch', 'new', 'this', 'super', 'return', 'typeof'}
        first_part = function_call.split('.')[0]
        return first_part not in keywords and not first_part.isupper()


class JavaCodeParser:
    """Parses Java code to extract relationships"""
    
    def __init__(self):
        self.patterns = {
            'import': re.compile(r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?);'),
            'package': re.compile(r'package\s+([a-zA-Z_][a-zA-Z0-9_.]*);'),
            'class': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+([a-zA-Z_][a-zA-Z0-9_]*))?\s*(?:implements\s+([a-zA-Z_][a-zA-Z0-9_,\s]*))?\s*\{'),
            'method': re.compile(r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>[\],\s]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[^{]*)?[{;]'),
            'method_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\([^)]*\)'),
            'constructor_call': re.compile(r'new\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'),
            'field_assignment': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*new\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'),
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
            
            # Add constructor calls from method content
            constructor_calls = self._extract_constructor_calls(method['content'])
            for call in constructor_calls:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='instantiates',
                    source_element=f"{class_info.get('name', '')}.{method['name']}",
                    target_element=call['constructor'],
                    line_number=method['line'] + call['line_offset'],
                    context=call['context']
                ))
            
            # Add field assignments from method content
            field_assignments = self._extract_field_assignments(method['content'])
            for assignment in field_assignments:
                relationships.append(CodeRelationship(
                    source_file=file_path,
                    target_file='',
                    relationship_type='assigns',
                    source_element=f"{class_info.get('name', '')}.{assignment['field']}",
                    target_element=assignment['constructor'],
                    line_number=method['line'] + assignment['line_offset'],
                    context=assignment['context']
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
    
    def _extract_constructor_calls(self, method_content: str) -> List[Dict]:
        """Extract constructor calls (new ClassName()) from method content"""
        calls = []
        lines = method_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
                
            matches = self.patterns['constructor_call'].finditer(line)
            for match in matches:
                constructor_name = match.group(1)
                if self._is_valid_constructor_call(constructor_name):
                    calls.append({
                        'constructor': constructor_name,
                        'line_offset': i,
                        'context': line.strip()
                    })
        
        return calls
    
    def _extract_field_assignments(self, method_content: str) -> List[Dict]:
        """Extract field assignments (field = new ClassName()) from method content"""
        assignments = []
        lines = method_content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('//') or line.strip().startswith('/*'):
                continue
                
            matches = self.patterns['field_assignment'].finditer(line)
            for match in matches:
                field_name = match.group(1)
                constructor_name = match.group(2)
                if self._is_valid_constructor_call(constructor_name):
                    assignments.append({
                        'field': field_name,
                        'constructor': constructor_name,
                        'line_offset': i,
                        'context': line.strip()
                    })
        
        return assignments
    
    def _is_valid_constructor_call(self, constructor_name: str) -> bool:
        """Validate constructor call name"""
        keywords = {'if', 'for', 'while', 'switch', 'catch', 'this', 'super'}
        return constructor_name not in keywords and not constructor_name.isupper() and constructor_name[0].isupper()


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
    
    def save_graph(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'file_to_classes': self.file_to_classes,
                'class_to_file': self.class_to_file
            }, f)


# Text file loader configuration
def safe_text_loader(file_path):
    """Safe file loading with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():  # Skip empty files
                print(f"Skipped empty file: {file_path}")
                return None
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except UnicodeDecodeError:
        print(f"Encoding error in file: {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Clean up existing data before starting
print("Cleaning up existing data...")

# Remove ChromaDB directory
chroma_db_path = "./chroma_db"
if os.path.exists(chroma_db_path):
    import shutil
    shutil.rmtree(chroma_db_path)
    print(f"  Removed ChromaDB directory: {chroma_db_path}")

# Remove graph file
graph_path = "./code_graph.pkl"
if os.path.exists(graph_path):
    os.remove(graph_path)
    print(f"  Removed graph file: {graph_path}")

print("Cleanup completed!\n")

# Load files from all specified paths
documents = []
found_files = []
processed_paths = []

for codebase_path in CODEBASE_PATHS:
    if not os.path.exists(codebase_path):
        print(f"Warning: Path {codebase_path} does not exist, skipping.")
        continue
    
    print(f"Processing path: {codebase_path}")
    processed_paths.append(codebase_path)
    
    # Count files in current path
    current_path_files = 0
    current_path_docs = 0
    
    for root, dirs, files in os.walk(codebase_path):
        # Modify dirs in-place to exclude specified folders
        dirs[:] = [d for d in dirs if d not in EXCLUDE_FOLDERS]
        
        for file in files:
            if file.endswith(('.py', '.js', '.cpp', '.java', '.txt', '.properties')):
                file_path = os.path.join(root, file)
                found_files.append(file_path)
                current_path_files += 1
                try:
                    doc = safe_text_loader(file_path)
                    if doc:
                        documents.extend(doc)
                        current_path_docs += len(doc)
                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")
    
    print(f"  Found files: {current_path_files}")
    print(f"  Loaded documents: {current_path_docs}")

# Check that documents were found
if not documents:
    raise ValueError(f"No loaded documents from {len(found_files)} files in {len(processed_paths)} paths. Issues with encoding, content, or access. Check logs above.")

print(f"\nTotal processed paths: {len(processed_paths)}")
print(f"Total found files: {len(found_files)}")
print(f"Total loaded documents: {len(documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save to ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="codebase",
    persist_directory="./chroma_db"
)
vector_store.persist()
print("Vector indexing completed!")

# Build code relationship graph
print("\nBuilding code relationship graph...")

# Initialize parsers
parsers = {
    '.java': JavaCodeParser(),
    '.py': PythonCodeParser(), 
    '.js': JavaScriptCodeParser(),
    '.jsx': JavaScriptCodeParser(),
    '.ts': JavaScriptCodeParser(),
    '.tsx': JavaScriptCodeParser()
}

# Initialize graph
code_graph = CodeGraph()
all_relationships = []

# Parse all documents to extract relationships
for doc in documents:
    source_file = doc.metadata.get('source', '')
    file_ext = None
    
    # Find matching parser by file extension
    for ext in parsers.keys():
        if source_file.endswith(ext):
            file_ext = ext
            break
    
    if file_ext and file_ext in parsers:
        try:
            relationships = parsers[file_ext].parse_file(
                source_file,
                doc.page_content
            )
            all_relationships.extend(relationships)
            print(f"  Parsed {len(relationships)} relationships from {source_file}")
        except Exception as e:
            print(f"  Error parsing {source_file}: {e}")

# Add relationships to graph
code_graph.add_relationships(all_relationships)

# Save graph
graph_path = "./code_graph.pkl"
code_graph.save_graph(graph_path)

print(f"\nGraph building completed!")
print(f"Total relationships extracted: {len(all_relationships)}")
print(f"Graph saved to: {graph_path}")
print("\nIndexing and graph building completed!")