"""
CodeLens — индексатор Python-кода.

Шаги:
  1. Находит все .py-файлы в папке.
  2. Режет каждый файл модулем ast на чанки: функция / класс / метод = один чанк.
  3. Строит эмбеддинг для каждого чанка (sentence-transformers) и сохраняет
     его вместе с метаданными в ChromaDB.

Запуск:  python index.py <папка> [--append]
По умолчанию база пересоздаётся с нуля, чтобы в ней не оставались чанки
от прошлых индексаций. Флаг --append добавляет чанки к существующей базе.
"""

import os
import sys
import ast
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_chunks"
BATCH_SIZE = 64

# Сколько символов кода добавлять в текст для эмбеддинга.
# Модель всё равно читает только начало текста (ограничение max_seq_length),
# поэтому самое важное (имя, docstring) ставим в начало, а код обрезаем.
MAX_CODE_CHARS_FOR_EMBEDDING = 1500

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "env", "node_modules",
             "build", "dist", ".mypy_cache", ".pytest_cache", "chroma_db"}


# ---------------------------------------------------------------------------
# Чанкование
# ---------------------------------------------------------------------------

def _make_chunk(node, lines: List[str], relative_path: str,
                chunk_type: str, name: str,
                parent_class: Optional[str] = None) -> Dict[str, Any]:
    """Собирает чанк из узла AST (функции, класса или метода)."""
    # Учитываем декораторы: они часть определения (@app.get, @property и т.п.)
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min([start] + [d.lineno for d in node.decorator_list])
    end = getattr(node, "end_lineno", None) or start

    source_code = "\n".join(lines[start - 1:end])
    docstring = ast.get_docstring(node) or ""

    metadata = {
        "file_path": relative_path,
        "chunk_type": chunk_type,
        "name": name,
        "start_line": start,
        "end_line": end,
        "docstring": docstring[:500],
    }
    if parent_class:
        metadata["parent_class"] = parent_class

    return {
        # Формат ID: <путь>:<имя>:<номер строки с def/class>
        "id": f"{relative_path}:{name}:{node.lineno}",
        "source_code": source_code,
        "metadata": metadata,
    }


def _walk_body(body, lines, relative_path, chunks, class_prefix: str = ""):
    """
    Обходит тело модуля или класса.

    Функции модуля -> "function", классы -> "class", функции внутри класса ->
    "method". Вложенные классы обрабатываются рекурсивно. Функции, объявленные
    внутри других функций, отдельными чанками не считаются — они уже входят
    в код внешней функции.

    Раньше использовался ast.walk(), из-за чего каждый метод попадал в базу
    дважды (как "function" и как "method").
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if class_prefix:
                chunks.append(_make_chunk(
                    node, lines, relative_path, "method",
                    f"{class_prefix}.{node.name}",
                    parent_class=class_prefix))
            else:
                chunks.append(_make_chunk(
                    node, lines, relative_path, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            class_name = f"{class_prefix}.{node.name}" if class_prefix else node.name
            chunks.append(_make_chunk(
                node, lines, relative_path, "class", class_name))
            _walk_body(node.body, lines, relative_path, chunks, class_name)
        elif isinstance(node, (ast.If, ast.Try)):
            # Определения внутри if/try на уровне модуля
            # (например, if TYPE_CHECKING: ... или try: import ...)
            inner = list(node.body) + list(getattr(node, "orelse", []))
            for handler in getattr(node, "handlers", []):
                inner += handler.body
            inner += list(getattr(node, "finalbody", []))
            _walk_body(inner, lines, relative_path, chunks, class_prefix)


def extract_chunks_from_file(file_path: str, repo_root: Path) -> List[Dict[str, Any]]:
    """Извлекает функции, классы и методы из одного Python-файла."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"  Ошибка чтения {file_path}: {e}")
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"  Синтаксическая ошибка в {file_path}: {e}")
        return []

    lines = source_code.splitlines()
    relative_path = Path(file_path).relative_to(repo_root).as_posix()

    chunks: List[Dict[str, Any]] = []
    _walk_body(tree.body, lines, relative_path, chunks)

    # Страховка от одинаковых ID (например, две функции с одним именем в if/else)
    unique = {}
    for c in chunks:
        unique.setdefault(c["id"], c)
    return list(unique.values())


def build_embedding_text(chunk: Dict[str, Any]) -> str:
    """
    Текст, по которому строится эмбеддинг.

    Кроме кода добавляем путь, тип, имя и docstring. Имена в коде часто
    составные (get_current_user), поэтому дублируем их через пробелы
    (get current user) — так модель лучше сопоставляет их с вопросом
    на естественном языке.
    """
    meta = chunk["metadata"]
    readable_name = meta["name"].replace("_", " ").replace(".", " ")
    readable_path = meta["file_path"].replace("/", " ").replace("_", " ").replace(".py", "")
    parts = [
        f"{meta['chunk_type']} {meta['name']} ({readable_name})",
        f"file: {meta['file_path']} ({readable_path})",
    ]
    if meta.get("docstring"):
        parts.append(meta["docstring"])
    parts.append(chunk["source_code"][:MAX_CODE_CHARS_FOR_EMBEDDING])
    return "\n".join(parts)


def find_python_files(directory: str) -> List[str]:
    """Рекурсивно находит .py-файлы, пропуская служебные папки."""
    root_path = Path(directory)
    files = []
    for p in root_path.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(root_path).parts):
            continue
        files.append(str(p))
    return sorted(files)


# ---------------------------------------------------------------------------
# Индексация
# ---------------------------------------------------------------------------

def index_directory(directory: str, model, collection) -> int:
    """Индексирует все Python-файлы в директории. Возвращает число чанков."""
    repo_root = Path(directory)

    print(f"\n[1/3] Поиск Python-файлов в {directory}...")
    py_files = find_python_files(directory)
    print(f"  Найдено {len(py_files)} .py-файлов")
    if not py_files:
        print("Ошибка: не найдено Python-файлов для индексации.")
        return 0

    print("\n[2/3] Извлечение функций, классов и методов...")
    all_chunks = []
    for file_path in py_files:
        chunks = extract_chunks_from_file(file_path, repo_root)
        all_chunks.extend(chunks)
        if chunks:
            print(f"  {Path(file_path).relative_to(repo_root).as_posix()}: {len(chunks)} чанков")

    counts = {}
    for c in all_chunks:
        counts[c["metadata"]["chunk_type"]] = counts.get(c["metadata"]["chunk_type"], 0) + 1
    print(f"\n  Всего {len(all_chunks)} чанков: " +
          ", ".join(f"{k} — {v}" for k, v in sorted(counts.items())))
    if not all_chunks:
        print("Не найдено чанков для индексации.")
        return 0

    print("\n[3/3] Генерация эмбеддингов и сохранение в ChromaDB...")
    t0 = time.perf_counter()
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        embed_texts = [build_embedding_text(c) for c in batch]
        embeddings = model.encode(embed_texts, show_progress_bar=False,
                                  normalize_embeddings=True)
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c["source_code"] for c in batch],   # показываем в UI исходный код
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  Обработано {min(i + BATCH_SIZE, len(all_chunks))} из {len(all_chunks)}")
    print(f"  Время индексации: {time.perf_counter() - t0:.1f} с")

    return len(all_chunks)


def main():
    parser = argparse.ArgumentParser(description="CodeLens RAG — индексатор кода")
    parser.add_argument("directory", type=str, help="Путь к директории с Python-кодом")
    parser.add_argument("--append", action="store_true",
                        help="Не очищать базу, а добавить чанки к существующим")
    parser.add_argument("--clear", action="store_true",
                        help="(оставлен для совместимости — база и так очищается по умолчанию)")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Ошибка: директория '{args.directory}' не существует")
        sys.exit(1)

    # Тяжёлые импорты — после проверки аргументов, чтобы ошибка ввода показывалась сразу
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("CodeLens — индексатор кода")
    print("Стратегия чанкования: функция / класс / метод = один чанк")
    print(f"Модель эмбеддингов: {MODEL_NAME}")
    print(f"Векторная БД: ChromaDB ({CHROMA_DB_PATH})")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if not args.append:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Старая коллекция удалена — индексируем с нуля")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nЗагрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    chunk_count = index_directory(args.directory, model, collection)

    if chunk_count > 0:
        print(f"\nИндексация завершена! В базе {collection.count()} чанков.")
        print("Запустите поиск: streamlit run app.py")
        print("Посчитать метрики: python search_engine.py")
    else:
        print("\nИндексация не выполнена")
        sys.exit(1)


if __name__ == "__main__":
    main()
