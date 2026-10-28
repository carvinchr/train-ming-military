#!/bin/bash
# 数据管线一键运行脚本
# 用法：bash 01_data_pipeline/run_pipeline.sh
set -e

PROJECT_ROOT="/mnt/d/ollama/train-ming-military"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=========================================="
echo " 明代军事装备大模型 — 数据管线"
echo "=========================================="

echo ""
echo ">>> Step 1: PDF文本提取"
python 01_data_pipeline/01_extract_pdf.py --input data/raw --output data/txt

echo ""
echo ">>> Step 2: OCR处理扫描件"
python 01_data_pipeline/02_ocr_images.py --input data/raw --output data/txt

echo ""
echo ">>> Step 3: 文本清洗"
python 01_data_pipeline/03_clean_text.py --input data/txt --output data/cleaned

echo ""
echo ">>> Step 4: token感知分块 (默认512 tokens，64重叠，可被 .env / CLI 覆盖)"
python 01_data_pipeline/04_chunk_text.py \
    --input data/cleaned \
    --output data/chunks \
    --chunk-tokens "${CHUNK_TOKENS:-512}" \
    --overlap-tokens "${OVERLAP_TOKENS:-64}"

echo ""
echo ">>> Step 5: 自动生成Q&A（需要Ollama运行）"
echo "    确认Ollama已启动且qwen3.6:35b已拉取（4090实测最详尽，见benchmark/report.md）..."
python 01_data_pipeline/05_generate_qa.py \
    --input data/chunks \
    --output data/qa_pairs \
    --model qwen3.6:35b \
    --questions-per-chunk 3

echo ""
echo "✅ 数据管线完成！"
echo "   文本块: data/chunks/all_chunks.jsonl"
echo "   Q&A对:  data/qa_pairs/all_qa.jsonl"
echo ""
echo ">>> 下一步：构建RAG索引"
echo "    bash 02_rag/build_and_serve.sh"
