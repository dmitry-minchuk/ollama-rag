import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Инициализация FastAPI
app = FastAPI()

# Путь к базе ChromaDB
CHROMA_DB_PATH = "../chroma_db"
COLLECTION_NAME = "codebase"

# Инициализация эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Загрузка векторной базы
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH
)

# Модель для запроса
class QueryRequest(BaseModel):
    query: str

# Эндпоинт для обработки запросов
@app.post("/query")
async def query_codebase(request: QueryRequest):
    try:
        # Поиск релевантных чанков (k=5 для топ-5 результатов)
        docs = vector_store.similarity_search(request.query, k=5)
        # Формируем ответ с текстом чанков и метаданными
        response = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs
        ]
        return {"chunks": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request handling error: {str(e)}")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
