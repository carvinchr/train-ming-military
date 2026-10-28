#!/bin/bash
# RAG一键构建和启动
set -e
PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo ">>> 构建向量索引..."
python 02_rag/build_index.py \
    --chunks data/chunks \
    --db-path data/chroma_db \
    --embedding-model BAAI/bge-m3

echo ">>> 启动RAG服务 (端口8080，模型/路径/CORS均从 .env 读取，见 00_setup/.env.example)..."
python 02_rag/rag_server.py --port 8080
