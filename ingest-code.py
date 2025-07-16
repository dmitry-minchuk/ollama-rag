import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Paths to codebases for indexing
CODEBASE_PATHS = [
    "/Users/dmitryminchuk/Projects/eis/openl-tests",
    # Add additional paths here:
    # "/path/to/another/codebase",
    # "/path/to/third/codebase",
]

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
    
    for root, _, files in os.walk(codebase_path):
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
print("Indexing completed!")
