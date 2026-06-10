import os
import sys
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_chunks"
BATCH_SIZE = 50

def _get_node_source(node, lines: List[str]) -> str:
    """Извлекает из файла код только одной функции или класса по номерам строк"""
    start_line = node.lineno - 1
    end_line = node.end_lineno - 1 if node.end_lineno else min(start_line + 100, len(lines))
    return '\n'.join(lines[start_line:end_line + 1]), start_line, end_line

def _extract_function_chunk(node, lines: List[str], relative_path: str) -> Optional[Dict]:
    """Извлекает функцию верхнего уровня"""
    try:
        source_code, start, end = _get_node_source(node, lines)
        docstring = ast.get_docstring(node) or ""
        
        return {
            "id": f"{relative_path}:{node.name}:{node.lineno}",
            "source_code": source_code,
            "metadata": {
                "file_path": relative_path,
                "chunk_type": "function",
                "name": node.name,
                "start_line": node.lineno,
                "end_line": end + 1,
                "docstring": docstring[:500],
            }
        }
    except Exception as e:
        print(f"Ошибка извлечения функции {node.name}: {e}")
        return None

def _extract_class_chunk(node, lines: List[str], relative_path: str) -> Optional[Dict]:
    """Извлекает класс"""
    try:
        source_code, start, end = _get_node_source(node, lines)
        docstring = ast.get_docstring(node) or ""
        
        return {
            "id": f"{relative_path}:{node.name}:{node.lineno}",
            "source_code": source_code,
            "metadata": {
                "file_path": relative_path,
                "chunk_type": "class",
                "name": node.name,
                "start_line": node.lineno,
                "end_line": end + 1,
                "docstring": docstring[:500],
            }
        }
    except Exception as e:
        print(f"Ошибка извлечения класса {node.name}: {e}")
        return None


def _extract_method_chunk(node, parent_class, lines: List[str], relative_path: str) -> Optional[Dict]:
    """Извлекает метод класса"""
    try:
        source_code, start, end = _get_node_source(node, lines)
        docstring = ast.get_docstring(node) or ""
        method_name = f"{parent_class.name}.{node.name}"
        
        return {
            "id": f"{relative_path}:{method_name}:{node.lineno}",
            "source_code": source_code,
            "metadata": {
                "file_path": relative_path,
                "chunk_type": "method",
                "name": method_name,
                "parent_class": parent_class.name,
                "start_line": node.lineno,
                "end_line": end + 1,
                "docstring": docstring[:500],
            }
        }
    except Exception as e:
        print(f"Ошибка извлечения метода {node.name}: {e}")
        return None

def extract_chunks_from_file(file_path: str, repo_root: Path) -> List[Dict[str, Any]]:
    """Извлекает функции, классы и методы из Python-файла с помощью AST"""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except (UnicodeDecodeError, IOError) as e:
        print(f"  Ошибка чтения {file_path}: {e}")
        return chunks
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Синтаксическая ошибка в {file_path}: {e}")
        return chunks
    lines = source_code.splitlines()
    relative_path = Path(file_path).relative_to(repo_root).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _extract_function_chunk(node, lines, relative_path)
            if chunk:
                chunks.append(chunk)
        elif isinstance(node, ast.ClassDef):
            chunk = _extract_class_chunk(node, lines, relative_path)
            if chunk:
                chunks.append(chunk)
            
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = _extract_method_chunk(child, node, lines, relative_path)
                    if chunk:
                        chunks.append(chunk)
    unique_chunks = {c["id"]: c for c in chunks}
    return list(unique_chunks.values())

def find_python_files(directory: str) -> List[str]:
    """Рекурсивно находит все .py-файлы в директории, исключая кэш"""
    root_path = Path(directory)
    return [str(p) for p in root_path.rglob("*.py") if "__pycache__" not in p.parts]

def index_directory(directory: str, model, collection) -> int:
    """Индексирует все Python-файлы в директории"""
    repo_root = Path(directory)
    
    print(f"\n [1/3] Поиск Python-файлов в {directory}...")
    py_files = find_python_files(directory)
    print(f"Найдено {len(py_files)} .py-файлов")
    
    if not py_files:
        print("Ошибка: не найдено Python-файлов для индексации.")
        return 0
    
    print("\n [2/3] Извлечение функций, классов и методов...")
    all_chunks = []
    
    for file_path in py_files:
        chunks = extract_chunks_from_file(file_path, repo_root)
        all_chunks.extend(chunks)
        print(f"{Path(file_path).name}: {len(chunks)} чанков")
    
    print(f"\n Всего извлечено {len(all_chunks)} чанков")
    if not all_chunks:
        print("Не найдено чанков для индексации.")
        return 0
    
    print("\n [3/3] Генерация эмбеддингов и сохранение в ChromaDB...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i+BATCH_SIZE]
        texts = [chunk["source_code"] for chunk in batch]
        
        embeddings = model.encode(texts, show_progress_bar=False)
        
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in batch],
        )
        print(f"Обработано {min(i+BATCH_SIZE, len(all_chunks))} из {len(all_chunks)}")
    
    return len(all_chunks)

def main():
    parser = argparse.ArgumentParser(description="CodeLens RAG - Индексатор кода")
    parser.add_argument("directory", type=str, help="Путь к директории с Python-кодом")
    parser.add_argument("--clear", action="store_true", help="Очистить существующую базу перед индексацией")
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Ошибка: директория '{args.directory}' не существует")
        sys.exit(1)
    
    print("CodeLens — Индексатор кода")
    print("Стратегия чункования: функция/класс/метод = один чанк")
    print(f"Модель эмбеддингов: {MODEL_NAME}")
    print(f"Векторная БД: ChromaDB ({CHROMA_DB_PATH})")
    
    print("\n Инициализация ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    if args.clear:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Коллекция '{COLLECTION_NAME}' очищена")
        except Exception:
            pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"\n Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Модель загружена")
    
    chunk_count = index_directory(args.directory, model, collection)
    
    if chunk_count > 0:
        print(f"\n Индексация завершена! Сохранено {chunk_count} чанков.")
        print(f"База данных: {CHROMA_DB_PATH}")
        print("\n Запустите поиск: streamlit run app.py")
    else:
        print("\n Индексация не завершена")
        sys.exit(1)

if __name__ == "__main__":
    main()