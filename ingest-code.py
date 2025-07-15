import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Путь к кодовой базе
CODEBASE_PATH = "/Users/dmitryminchuk/Projects/eis/openl-tests"

# Настройка загрузчика для текстовых файлов
def safe_text_loader(file_path):
    """Безопасная загрузка файла с обработкой ошибок."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():  # Пропускаем пустые файлы
                print(f"Пропущен пустой файл: {file_path}")
                return None
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except UnicodeDecodeError:
        print(f"Ошибка кодировки в файле: {file_path}. Пропускаем.")
        return None
    except Exception as e:
        print(f"Ошибка загрузки файла {file_path}: {e}")
        return None

# Настройка загрузчика
loader = DirectoryLoader(
    CODEBASE_PATH,
    glob="**/*.{py,js,cpp,java,txt,properties}",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    use_multithreading=True  # Ускоряем загрузку
)

# Загрузка файлов с фильтрацией
documents = []
found_files = []
for root, _, files in os.walk(CODEBASE_PATH):
    for file in files:
        if file.endswith(('.py', '.js', '.cpp', '.java', '.txt', '.properties')):
            file_path = os.path.join(root, file)
            found_files.append(file_path)
            try:
                doc = safe_text_loader(file_path)
                if doc:
                    documents.extend(doc)
            except Exception as e:
                print(f"Не удалось обработать файл {file_path}: {e}")

# Проверка, что документы найдены
if not documents:
    raise ValueError(f"Нет загруженных документов из {len(found_files)} файлов в {CODEBASE_PATH}. Проблемы с кодировкой, содержимым или доступом. Проверьте логи выше.")

print(f"Найдено и загружено {len(documents)} документов из {len(found_files)} файлов")

# Разбиение на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Создание эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Сохранение в ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="codebase",
    persist_directory="./chroma_db"
)
vector_store.persist()
print("Индексация завершена!")
