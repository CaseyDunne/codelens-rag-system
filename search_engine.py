"""
CodeLens — поисковый движок и расчёт метрик.

Поиск:
  * векторный — эмбеддинг запроса + ближайшие соседи в ChromaDB (косинус);
  * гибридный — alpha * векторная близость + (1 - alpha) * TF-IDF.

Метрики (python search_engine.py):
  * Precision@K — доля релевантных чанков среди K выданных (строгое определение);
  * Hit Rate@K  — доля вопросов, где хотя бы один эталонный чанк попал в топ-K;
  * Recall@K    — доля эталонных чанков вопроса, попавших в топ-K;
  * MRR         — средний обратный ранг первого правильного ответа;
  * Latency     — время ответа на каждый запрос (среднее, p95, максимум).

Примеры:
  python search_engine.py                       # векторный поиск
  python search_engine.py --hybrid --alpha 0.6  # гибридный поиск
  python search_engine.py --compare             # сравнение режимов (таблица для доклада)
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_chunks"
EVAL_FILE = os.environ.get("CODELENS_EVAL_FILE", "eval_questions.json")
METRICS_FILE = "metrics.json"
COMPARE_FILE = "metrics_compare.json"

_model = None
_collection = None
_ids: List[str] = []
_documents: List[str] = []
_metadatas: List[Dict[str, Any]] = []
_embeddings: Optional[np.ndarray] = None
_tfidf_vectorizer = None
_tfidf_matrix = None


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def _split_identifiers(text: str) -> str:
    """get_current_user / getCurrentUser -> 'get current user' (для TF-IDF)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return text.replace("_", " ").lower()


def _tfidf_text(doc: str, meta: Dict[str, Any]) -> str:
    return " ".join([
        _split_identifiers(meta.get("name", "")),
        _split_identifiers(meta.get("file_path", "")),
        meta.get("docstring", "") or "",
        _split_identifiers(doc),
    ])


def reset():
    """Сбросить кэш (нужно после переиндексации, если приложение уже запущено)."""
    global _collection, _ids, _documents, _metadatas, _embeddings
    global _tfidf_vectorizer, _tfidf_matrix
    _collection = None
    _ids, _documents, _metadatas = [], [], []
    _embeddings = None
    _tfidf_vectorizer = None
    _tfidf_matrix = None


def init_db():
    """Загружает модель, подключается к ChromaDB и готовит TF-IDF."""
    global _model, _collection, _ids, _documents, _metadatas, _embeddings
    global _tfidf_vectorizer, _tfidf_matrix

    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Загрузка модели...")
        _model = SentenceTransformer(MODEL_NAME)

    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            # get_collection, а не get_or_create: иначе при запуске UI до индексации
            # создалась бы пустая коллекция с метрикой L2 вместо косинуса.
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            print("База не найдена. Сначала запустите: python index.py <папка>")
            return _model, None

        data = _collection.get(include=["documents", "metadatas", "embeddings"])
        _ids = list(data["ids"])
        _documents = list(data["documents"])
        _metadatas = list(data["metadatas"])
        emb = data.get("embeddings")
        if emb is not None and len(emb) > 0:
            emb = np.asarray(emb, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            _embeddings = emb / np.clip(norms, 1e-12, None)

        if _documents:
            _tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            _tfidf_matrix = _tfidf_vectorizer.fit_transform(
                [_tfidf_text(d, m) for d, m in zip(_documents, _metadatas)])
            print(f"База загружена: {len(_ids)} чанков")

    return _model, _collection


def db_stats() -> Dict[str, int]:
    """Сколько файлов и чанков в базе."""
    if _collection is None:
        init_db()
    files = {m.get("file_path") for m in _metadatas}
    return {"chunks": len(_ids), "files": len(files)}


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------

def _encode_query(query: str) -> np.ndarray:
    return np.asarray(_model.encode(query, normalize_embeddings=True), dtype=np.float32)


def _format(idx: int, score: float, **extra) -> Dict[str, Any]:
    meta = _metadatas[idx]
    item = {
        "id": _ids[idx],
        "path": meta.get("file_path", "unknown"),
        "type": meta.get("chunk_type", "unknown"),
        "name": meta.get("name", "unknown"),
        "start_line": meta.get("start_line"),
        "end_line": meta.get("end_line"),
        "percent": round(float(score) * 100, 1),
        "code": _documents[idx],
    }
    item.update(extra)
    return item


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Чисто векторный поиск через ChromaDB."""
    q = _encode_query(query)
    res = _collection.query(
        query_embeddings=[q.tolist()],
        n_results=min(top_k, len(_ids)),
        include=["distances"],
    )
    id_to_idx = {cid: i for i, cid in enumerate(_ids)}
    out = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        out.append(_format(id_to_idx[cid], 1 - dist))
    return out


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.6) -> List[Dict[str, Any]]:
    """
    Гибридный поиск по всем чанкам: alpha * cos(вектор) + (1 - alpha) * TF-IDF.
    TF-IDF нормируется на максимум по запросу, чтобы обе оценки были в [0, 1].
    """
    q = _encode_query(query)
    vec_scores = _embeddings @ q

    tfidf_scores = np.zeros(len(_ids), dtype=np.float32)
    if _tfidf_vectorizer is not None:
        q_tfidf = _tfidf_vectorizer.transform([_split_identifiers(query)])
        sims = (_tfidf_matrix @ q_tfidf.T).toarray().ravel()
        if sims.max() > 0:
            tfidf_scores = sims / sims.max()

    combined = alpha * vec_scores + (1 - alpha) * tfidf_scores
    order = np.argsort(-combined)[:top_k]
    return [
        _format(int(i), combined[i],
                vector_score=round(float(vec_scores[i]), 3),
                tfidf_score=round(float(tfidf_scores[i]), 3))
        for i in order
    ]


def search(query: str, top_k: int = 5, hybrid: bool = False, alpha: float = 0.6):
    """Основная функция поиска для app.py."""
    if _collection is None:
        init_db()
    if _collection is None or not _ids:
        return []
    if hybrid and _embeddings is not None:
        return hybrid_search(query, top_k, alpha)
    return vector_search(query, top_k)


# ---------------------------------------------------------------------------
# Оценка качества
# ---------------------------------------------------------------------------

def _parse_ref(ref: str):
    """'app/api/auth.py:login:12' -> ('app/api/auth.py', 'login', 12)."""
    parts = ref.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    if len(parts) >= 2:
        return ":".join(parts[:-1]), parts[-1], None
    return None, ref, None


def _path_match(a: str, b: str) -> bool:
    a, b = a.strip("./"), b.strip("./")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _name_match(db_name: str, ref_name: str) -> bool:
    """'get_name' совпадает с 'User.get_name', но 'User.get_name' не совпадает с 'Admin.get_name'."""
    if db_name == ref_name:
        return True
    if db_name.split(".")[-1] != ref_name.split(".")[-1]:
        return False
    return ("." not in ref_name or "." not in db_name
            or db_name.endswith("." + ref_name) or ref_name.endswith("." + db_name))


def resolve_reference(ref: str) -> set:
    """
    Находит в базе чанк(и), соответствующие эталонному ID.
    Сначала точное совпадение ID, затем путь + имя + строка, затем путь + имя.
    Это нужно, потому что формат ID в эталоне может немного отличаться
    (другой корень пути, метод без имени класса и т.п.).
    """
    if ref in _ids:
        return {ref}
    path, name, line = _parse_ref(ref)
    candidates = []
    for cid, meta in zip(_ids, _metadatas):
        if path and not _path_match(meta.get("file_path", ""), path):
            continue
        if not _name_match(meta.get("name", ""), name):
            continue
        candidates.append(cid)
    if line is not None:
        exact = [c for c in candidates if _parse_ref(c)[2] == line]
        if exact:
            return set(exact)
    return set(candidates)


def _detect_language(text: str) -> str:
    return "ru" if re.search(r"[а-яё]", text, re.I) else "en"


def _load_eval(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # {"questions": [...]}
        data = data.get("questions") or data.get("items") or list(data.values())[0]
    items = []
    for row in data:
        q = row.get("question") or row.get("query") or row.get("text")
        refs = (row.get("relevant_chunk_ids") or row.get("relevant_ids")
                or row.get("expected_chunk_ids") or row.get("chunk_ids")
                or row.get("relevant_chunk_id") or row.get("chunk_id") or [])
        if isinstance(refs, str):
            refs = [refs]
        if q and refs:
            items.append({"question": q, "refs": refs,
                          "language": row.get("language") or _detect_language(q)})
    return items


def evaluate(eval_file: str = EVAL_FILE, k: int = 5, hybrid: bool = False,
             alpha: float = 0.6, verbose: bool = True) -> Dict[str, Any]:
    """Прогоняет тестовые вопросы и считает метрики."""
    init_db()
    if _collection is None or not _ids:
        raise RuntimeError("База пуста — сначала запустите python index.py <папка>")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Не найден файл {eval_file}")

    items = _load_eval(eval_file)
    search(items[0]["question"], top_k=k, hybrid=hybrid, alpha=alpha)  # прогрев модели

    per_q, latencies, unresolved = [], [], []
    for it in items:
        relevant = set()
        for ref in it["refs"]:
            found = resolve_reference(ref)
            if not found:
                unresolved.append(ref)
            relevant |= found

        t0 = time.perf_counter()
        results = search(it["question"], top_k=k, hybrid=hybrid, alpha=alpha)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        got = [r["id"] for r in results]
        hits = [cid for cid in got if cid in relevant]
        first_rank = next((i + 1 for i, cid in enumerate(got) if cid in relevant), None)

        per_q.append({
            "question": it["question"],
            "language": it["language"],
            "precision": len(hits) / k,
            "hit": 1.0 if hits else 0.0,
            "recall": len(hits) / len(relevant) if relevant else 0.0,
            "rr": 1.0 / first_rank if first_rank else 0.0,
            "first_rank": first_rank,
            "latency_s": round(latency, 4),
            "relevant": sorted(relevant),
            "returned": got,
        })

    def agg(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "precision_at_k": round(100 * np.mean([r["precision"] for r in rows]), 1),
            "hit_rate_at_k": round(100 * np.mean([r["hit"] for r in rows]), 1),
            "recall_at_k": round(100 * np.mean([r["recall"] for r in rows]), 1),
            "mrr": round(float(np.mean([r["rr"] for r in rows])), 3),
        }

    lat = np.array(latencies)
    stats = db_stats()
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_file": eval_file,
        "mode": "hybrid" if hybrid else "vector",
        "alpha": alpha if hybrid else None,
        "k": k,
        "db_files": stats["files"],
        "db_chunks": stats["chunks"],
        "overall": agg(per_q),
        "by_language": {lang: agg([r for r in per_q if r["language"] == lang])
                        for lang in sorted({r["language"] for r in per_q})},
        "latency_s": {
            "mean": round(float(lat.mean()), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "max": round(float(lat.max()), 3),
        },
        "unresolved_references": sorted(set(unresolved)),
        "per_question": per_q,
    }

    if verbose:
        o = metrics["overall"]
        print(f"\nРежим: {metrics['mode']}" + (f" (alpha={alpha})" if hybrid else ""))
        print(f"База: {stats['files']} файлов, {stats['chunks']} чанков; вопросов: {o['n']}")
        print(f"  Precision@{k}: {o['precision_at_k']}%   Hit Rate@{k}: {o['hit_rate_at_k']}%   "
              f"Recall@{k}: {o['recall_at_k']}%   MRR: {o['mrr']}")
        for lang, m in metrics["by_language"].items():
            print(f"  [{lang}] n={m['n']}: Precision@{k} {m['precision_at_k']}%, "
                  f"Hit Rate@{k} {m['hit_rate_at_k']}%, MRR {m['mrr']}")
        print(f"  Latency: среднее {metrics['latency_s']['mean']} с, "
              f"p95 {metrics['latency_s']['p95']} с, макс {metrics['latency_s']['max']} с")
        if unresolved:
            print(f"\n  ВНИМАНИЕ: {len(set(unresolved))} эталонных ID не найдено в базе "
                  f"(проиндексирован не тот код или другой формат ID):")
            for ref in sorted(set(unresolved))[:10]:
                print(f"    - {ref}")
    return metrics


def save_metrics(metrics: Dict[str, Any], path: str = METRICS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def evaluate_precision(hybrid: bool = False, alpha: float = 0.6):
    """Обёртка для кнопки на странице метрик: считает и сохраняет metrics.json."""
    metrics = evaluate(EVAL_FILE, hybrid=hybrid, alpha=alpha)
    save_metrics(metrics)
    return metrics


def compare(eval_file: str = EVAL_FILE, k: int = 5, alphas=(0.9, 0.8, 0.7, 0.6, 0.5)):
    """Сравнивает векторный и гибридный поиск с разными alpha."""
    rows = [evaluate(eval_file, k=k, hybrid=False, verbose=False)]
    for a in alphas:
        rows.append(evaluate(eval_file, k=k, hybrid=True, alpha=a, verbose=False))

    print(f"\n{'Режим':<16}{'P@'+str(k):>8}{'Hit@'+str(k):>9}{'R@'+str(k):>8}"
          f"{'MRR':>7}{'ru Hit':>8}{'en Hit':>8}{'ср.время':>10}")
    for m in rows:
        name = m["mode"] + (f" a={m['alpha']}" if m["alpha"] is not None else "")
        o, bl = m["overall"], m["by_language"]
        ru = bl.get("ru", {}) or {}
        en = bl.get("en", {}) or {}
        print(f"{name:<16}{o['precision_at_k']:>7}%{o['hit_rate_at_k']:>8}%"
              f"{o['recall_at_k']:>7}%{o['mrr']:>7}{ru.get('hit_rate_at_k', '-'):>7}%"
              f"{en.get('hit_rate_at_k', '-'):>7}%{m['latency_s']['mean']:>9}с")

    summary = [{k2: v for k2, v in m.items() if k2 != "per_question"} for m in rows]
    with open(COMPARE_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nСводка сохранена в {COMPARE_FILE}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CodeLens — расчёт метрик качества поиска")
    p.add_argument("--eval", default=EVAL_FILE, help="Файл с тестовыми вопросами")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hybrid", action="store_true", help="Гибридный поиск")
    p.add_argument("--alpha", type=float, default=0.6, help="Вес векторной части (0..1)")
    p.add_argument("--compare", action="store_true", help="Сравнить режимы поиска")
    args = p.parse_args()

    if args.compare:
        compare(args.eval, args.k)
    else:
        result = evaluate(args.eval, k=args.k, hybrid=args.hybrid, alpha=args.alpha)
        save_metrics(result)
        print(f"\nМетрики сохранены в {METRICS_FILE}")
