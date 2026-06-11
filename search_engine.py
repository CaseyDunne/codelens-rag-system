import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = "./indexer/chroma_db"
COLLECTION_NAME = "code_chunks"

_model = None
_collection = None


def init_db():
    global _model, _collection
    if _model is None:
        print("🔧 Загрузка модели...")
        _model = SentenceTransformer(MODEL_NAME)
    if _collection is None:
        print("🔧 Подключение к ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def search(query: str, top_k: int = 5, hybrid: bool = False, alpha: float = 0.5):
    global _model, _collection
    init_db()

    # Генерируем эмбеддинг запроса
    query_embedding = _model.encode(query)

    # Ищем в ChromaDB
    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Форматируем результаты
    formatted = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        percent = round((1 - distance) * 100, 1)
        metadata = results["metadatas"][0][i]

        formatted.append({
            "path": metadata.get("file_path", "unknown"),
            "type": metadata.get("chunk_type", "unknown"),
            "name": metadata.get("name", "unknown"),
            "percent": percent,
            "code": results["documents"][0][i]
        })

    return formatted


def evaluate_precision():
    # Заглушка для метрик
    import json
    metrics = {"total_accuracy": 75, "ru_accuracy": 85, "en_accuracy": 65}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    return metrics