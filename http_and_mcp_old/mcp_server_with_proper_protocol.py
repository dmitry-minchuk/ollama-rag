import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
from fastapi.responses import JSONResponse

# Initialize FastAPI
app = FastAPI()

# ChromaDB path
CHROMA_DB_PATH = "../chroma_db"
COLLECTION_NAME = "codebase"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH
)

# JSON-RPC request model
class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None

# JSON-RPC response model
class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Any] = None
    id: Optional[Any] = None

# MCP protocol endpoint
@app.post("/")
async def handle_mcp_request(request: JsonRpcRequest):
    try:
        method = request.method
        params = request.params or {}

        if method == "tools/list":
            # Return available tools
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
            return JsonRpcResponse(result=result, id=request.id)

        elif method == "tools/call":
            # Call the code_search tool
            tool_name = params.get("name")
            if tool_name != "code_search":
                return JsonRpcResponse(
                    error={"code": -32601, "message": f"Tool {tool_name} not found"},
                    id=request.id
                )

            arguments = params.get("arguments", {})
            query = arguments.get("query")
            limit = arguments.get("limit", 5)

            if not query:
                return JsonRpcResponse(
                    error={"code": -32602, "message": "Missing query parameter in arguments"},
                    id=request.id
                )

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
            return JsonRpcResponse(result=result, id=request.id)

        elif method == "resources/list":
            # Return list of unique files from metadata
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
                return JsonRpcResponse(result=result, id=request.id)

            except Exception as e:
                return JsonRpcResponse(
                    error={"code": -32000, "message": f"Failed to list resources: {str(e)}"},
                    id=request.id
                )

        elif method == "resources/read":
            # Read resource by URI
            uri = params.get("uri")
            if not uri:
                return JsonRpcResponse(
                    error={"code": -32602, "message": "Missing uri parameter"},
                    id=request.id
                )

            # Extract file path from URI
            if uri.startswith("file://"):
                file_path = uri[7:]  # Remove "file://" prefix
            else:
                file_path = uri

            try:
                # Find documents by file path
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
                    return JsonRpcResponse(
                        error={"code": -32602, "message": f"Resource {uri} not found"},
                        id=request.id
                    )

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
                return JsonRpcResponse(result=result, id=request.id)

            except Exception as e:
                return JsonRpcResponse(
                    error={"code": -32000, "message": f"Failed to read resource: {str(e)}"},
                    id=request.id
                )

        else:
            return JsonRpcResponse(
                error={"code": -32601, "message": f"Method {method} not found"},
                id=request.id
            )

    except Exception as e:
        return JsonRpcResponse(
            error={"code": -32000, "message": f"Server error: {str(e)}"},
            id=request.id
        )

# Compatibility with custom /query endpoint
class QueryRequest(BaseModel):
    query: str
    limit: int = 5

@app.post("/query")
async def query_codebase(request: QueryRequest):
    try:
        docs = vector_store.similarity_search(request.query, k=request.limit)
        response = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs
        ]
        return {"chunks": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "vector_store": "connected"}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
