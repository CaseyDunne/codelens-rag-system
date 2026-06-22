
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

os.chdir(ROOT_DIR)

from indexer.indexer import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python index.py <путь_к_папке> [--clear]")
        print("Пример: python index.py test_code")
        sys.exit(1)
    main()