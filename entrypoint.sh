#!/bin/bash
# ==========================================
# CodeLens RAG System - Docker Entrypoint
# ==========================================
set -e

echo "=========================================="
echo "  CodeLens RAG System - Starting..."
echo "=========================================="

DB_PATH="/app/chroma_db"

if [ ! -d "$DB_PATH" ] || [ -z "$(ls -A "$DB_PATH" 2>/dev/null)" ]; then
    echo ""
    echo "[INFO] Database not found or empty."
    echo "[INFO] Indexing test_code directory..."
    echo ""
    python index.py /app/test_code
    echo ""
    echo "[SUCCESS] Indexing completed!"
    echo ""
else
    echo "[INFO] Database found. Skipping indexing."
    echo ""
fi

echo "[INFO] Starting Streamlit UI on port 8501..."
echo "[INFO] Open in browser: http://localhost:8501"
echo "=========================================="
echo ""

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false