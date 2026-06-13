import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = "./indexer/chroma_db"
COLLECTION_NAME = "code_chunks"

_model = None
_collection = None
_tfidf_vectorizer = None
_tfidf_matrix = None
_chunks_texts = None
_chunks_metadata = None


def init_db():
    """Инициализация модели и БД для кэширования в Streamlit"""
    global _model, _collection, _tfidf_vectorizer, _tfidf_matrix, _chunks_texts, _chunks_metadata

    if _model is None:
        print("🔧 Загрузка модели...")
        _model = SentenceTransformer(MODEL_NAME)

    if _collection is None:
        print("🔧 Подключение к ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)

    # Подготовка для гибридного поиска (TF-IDF)
    if _tfidf_vectorizer is None and _collection is not None:
        try:
            print("🔧 Подготовка TF-IDF для гибридного поиска...")
            all_chunks = _collection.get()
            _chunks_texts = all_chunks.get("documents", [])
            _chunks_metadata = all_chunks.get("metadatas", [])

            if _chunks_texts:
                _tfidf_vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=None)
                _tfidf_matrix = _tfidf_vectorizer.fit_transform(_chunks_texts)
                print(f"✅ TF-IDF готов: {_tfidf_matrix.shape[0]} чанков")
        except Exception as e:
            print(f"⚠️ TF-IDF не инициализирован: {e}")

    return _model, _collection


def vector_search(query: str, top_k: int = 5):
    """Чисто векторный поиск"""
    global _model, _collection

    query_embedding = _model.encode(query)
    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        percent = round((1 - distance) * 100, 1)

        metadata = results["metadatas"][0][i]
        formatted_results.append({
            "path": metadata.get("file_path", "unknown"),
            "type": metadata.get("chunk_type", "unknown"),
            "name": metadata.get("name", "unknown"),
            "percent": percent,
            "code": results["documents"][0][i]
        })

    return formatted_results


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """Гибридный поиск: векторный + TF-IDF"""
    global _model, _collection, _tfidf_vectorizer, _tfidf_matrix, _chunks_texts, _chunks_metadata

    # Векторный поиск (берём больше кандидатов)
    query_embedding = _model.encode(query)
    vector_results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k * 3, len(_chunks_texts) if _chunks_texts else top_k),
        include=["documents", "metadatas", "distances"]
    )

    # TF-IDF поиск
    if _tfidf_vectorizer is not None and _chunks_texts:
        query_tfidf = _tfidf_vectorizer.transform([query])
        similarities = np.dot(_tfidf_matrix, query_tfidf.T).toarray().flatten()
        if similarities.max() > 0:
            similarities = similarities / similarities.max()
        tfidf_scores = similarities
    else:
        tfidf_scores = np.zeros(len(_chunks_texts)) if _chunks_texts else []

    # Комбинируем результаты
    combined = []
    for i in range(len(vector_results["ids"][0])):
        chunk_text = vector_results["documents"][0][i]
        metadata = vector_results["metadatas"][0][i]

        # Находим TF-IDF оценку для этого чанка
        try:
            chunk_idx = _chunks_texts.index(chunk_text) if _chunks_texts else -1
            tfidf_score = tfidf_scores[chunk_idx] if chunk_idx >= 0 else 0
        except ValueError:
            tfidf_score = 0

        vector_score = 1 - vector_results["distances"][0][i]
        combined_score = alpha * vector_score + (1 - alpha) * tfidf_score
        percent = round(combined_score * 100, 1)

        combined.append({
            "path": metadata.get("file_path", "unknown"),
            "type": metadata.get("chunk_type", "unknown"),
            "name": metadata.get("name", "unknown"),
            "percent": percent,
            "code": chunk_text,
            "_sort_key": combined_score
        })

    combined.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in combined:
        del item["_sort_key"]

    return combined[:top_k]


def search(query: str, top_k: int = 5, hybrid: bool = False, alpha: float = 0.5):
    """Основная функция поиска для вызова из app.py"""
    global _collection

    if _collection is None:
        init_db()

    if _collection is None or _collection.count() == 0:
        return []

    if hybrid:
        return hybrid_search(query, top_k, alpha)
    else:
        return vector_search(query, top_k)


def evaluate_precision():
    """Вычисляет Precision@5 на тестовом наборе и сохраняет metrics.json"""

    init_db()

    if not os.path.exists("eval_questions.json"):
        print("❌ eval_questions.json не найден")
        metrics = {"total_accuracy": 0, "ru_accuracy": 0, "en_accuracy": 0}
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f)
        return metrics

    with open("eval_questions.json", "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    # Получаем все чанки из БД
    all_chunks = _collection.get()
    id_to_metadata = {all_chunks["ids"][i]: all_chunks["metadatas"][i] for i in range(len(all_chunks["ids"]))}

    ru_correct = 0
    ru_total = 0
    en_correct = 0
    en_total = 0

    for item in test_queries:
        query = item.get("question", "")
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        language = item.get("language", "unknown")

        if not query or not relevant_ids:
            continue

        # Выполняем поиск
        results = search(query, top_k=5, hybrid=False)

        # Проверяем, попал ли эталонный чанк в результаты
        found = False
        for r in results:
            for chunk_id, meta in id_to_metadata.items():
                if (meta.get("file_path") == r["path"] and
                        meta.get("name") == r["name"]):
                    if chunk_id in relevant_ids:
                        found = True
                    break

        if language == "ru":
            ru_total += 1
            if found:
                ru_correct += 1
        elif language == "en":
            en_total += 1
            if found:
                en_correct += 1

    ru_acc = round(ru_correct / ru_total * 100, 1) if ru_total > 0 else 0
    en_acc = round(en_correct / en_total * 100, 1) if en_total > 0 else 0
    total = round((ru_correct + en_correct) / (ru_total + en_total) * 100, 1) if (ru_total + en_total) > 0 else 0

    metrics = {
        "total_accuracy": total,
        "ru_accuracy": ru_acc,
        "en_accuracy": en_acc
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"📊 Метрики сохранены в metrics.json")
    print(f"   Total Precision@5: {total}%")
    print(f"   Русский: {ru_acc}% (на {ru_total} запросах)")
    print(f"   Английский: {en_acc}% (на {en_total} запросах)")

    return metrics


if __name__ == "__main__":
    evaluate_precision()