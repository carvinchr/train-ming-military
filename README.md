# 明朝军事装备专域大模型

基于 **RAG + LoRA微调** 的明代军事史专业问答系统。经 [机器A：RTX 4090 Laptop 16GB 实测](benchmark/reports/rtx4090-laptop-16gb.md) 验证，微调基座与线上推理默认模型均为 **Qwen3.5:9B**（16GB显卡即可训练+部署，速度快），另有 **Qwen3.6:35B MoE** 作为质量优先的RAG部署候选（无需微调，直接挂知识库），详见 [RUNBOOK.md](RUNBOOK.md) 阶段5部署说明。项目还在 [机器B：RTX 3060 12GB](benchmark/reports/rtx3060-12gb.md) 上做了同一批基准测试——模型排名并不相同，12GB显存下 `qwen2.5:7b` 反而是速度最快的选项。

## 快速开始

```bash
# 1. 环境初始化（WSL Ubuntu）+ 配置环境变量
bash 00_setup/wsl_setup.sh
cp 00_setup/.env.example .env    # 按需修改

# （可选）运行单元测试（集成测试在RAG服务不可达时自动跳过）
pip install -r requirements-dev.txt && pytest

# 2. 将书籍PDF放入 data/raw/，然后运行数据管线
bash 01_data_pipeline/run_pipeline.sh

# 3. 构建RAG索引并启动服务
bash 02_rag/build_and_serve.sh

# 4. （可选）微调训练
bash 03_finetune/train.sh

# 5. 一键部署
bash 05_deployment/deploy.sh
```

## 目录结构

```
train-ming-military/
├── RUNBOOK.md                    ← 完整操作手册（主要文档）
├── README.md                     ← 本文件
├── requirements-dev.txt          ← 开发/CI依赖（pytest + ruff）
├── pytest.ini / ruff.toml        ← 测试与lint配置
├── tests/                        ← pytest 单元测试 + RAG集成测试（服务不可达自动跳过）
├── .github/workflows/ci.yml      ← CI：py_compile + ruff + pytest
├── 00_setup/
│   ├── wsl_setup.sh              ← WSL环境一键初始化
│   ├── requirements.txt          ← Python依赖
│   ├── .env.example              ← 环境变量模板（复制为项目根目录 .env）
│   └── docker-compose.yml        ← ChromaDB + Open-WebUI
├── 01_data_pipeline/
│   ├── 01_extract_pdf.py         ← PDF文字提取
│   ├── 02_ocr_images.py          ← OCR扫描识别（PaddleOCR）
│   ├── 03_clean_text.py          ← 文本清洗
│   ├── 04_chunk_text.py          ← 语义分块
│   ├── 05_generate_qa.py         ← LLM自动生成Q&A对
│   └── run_pipeline.sh           ← 一键运行数据管线
├── 02_rag/
│   ├── build_index.py            ← 构建ChromaDB向量索引
│   ├── rag_server.py             ← FastAPI RAG服务
│   ├── test_rag.py               ← 交互式测试
│   └── build_and_serve.sh        ← 一键构建和启动
├── 03_finetune/
│   ├── prepare_sft.py            ← 准备SFT训练数据
│   ├── qwen35_lora.yaml          ← LLaMA-Factory训练配置（Qwen3.5-9B）
│   ├── train.sh                  ← 训练脚本
│   └── merge_lora.sh             ← 合并LoRA权重
├── 04_evaluation/
│   ├── create_testset.py         ← 生成测试集模板
│   ├── eval_model.py             ← 评估模型（ROUGE + LLM-as-Judge）
│   ├── eval_rag.py               ← 评估RAG系统
│   └── compare.py                ← 三路对比报告
├── 05_deployment/
│   ├── Modelfile                 ← Ollama模型配置
│   └── deploy.sh                 ← 一键部署
└── benchmark/
    ├── run_benchmark.py          ← 本地Ollama模型基准测试脚本（跑完后需手动按机型归档，见下）
    ├── reports/                  ← 按机型命名的报告（rtx4090-laptop-16gb.md、rtx3060-12gb.md）
    └── raw/                      ← 各模型原始测试数据，按机型分子目录归档（rtx4090-laptop-16gb/、rtx3060-12gb/）
```

## 核心技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 微调基座 | Qwen3.5-9B-Instruct | 4090（16GB）实测QLoRA训练显存仅需10-13GB，训练与部署统一到同一硬件规格 |
| RAG部署候选（不微调） | Qwen3.6:35B MoE | 见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)：回答最详尽，质量优先场景直接挂知识库使用 |
| 嵌入模型 | BAAI/bge-m3 | 中文检索SOTA，支持长文本 |
| 向量数据库 | ChromaDB | 轻量易用，支持持久化 |
| 微调框架 | LLaMA-Factory | 支持Qwen系列，QLoRA省显存 |
| 推理服务 | Ollama | 本地部署，OpenAI兼容API |
| OCR引擎 | PaddleOCR | 中文识别率高 |

## 机器A：RTX 4090 Laptop（16GB VRAM）候选模型

在 RTX 4090 Laptop（16GB VRAM）上对本地 Ollama 模型做了基准测试（[完整报告](benchmark/reports/rtx4090-laptop-16gb.md)），综合速度、显存占用与回答质量后确定两个部署候选：

| 候选模型 | 生成速度 | VRAM占用 | 适用场景 |
|----------|----------|----------|----------|
| `qwen3.5:9b` | 82.6 tok/s | ~6.6 GB | 微调基座 + 日常问答，速度优先，可与其他模型共存显存 |
| `qwen3.6:35b`（MoE） | 53.6 tok/s | ~23 GB | 深度分析/SFT数据生成，回答更详尽，不参与微调 |

两者的 Open-WebUI 部署配置与差异说明见 [RUNBOOK.md](RUNBOOK.md) 阶段5.2/5.3。

## 机器B：RTX 3060（12GB VRAM）候选模型

在 RTX 3060（12GB VRAM）上对本地已装的5个模型做了基准测试（[完整报告](benchmark/reports/rtx3060-12gb.md)）。**排名与机器A不同**——显存更小，模型越大掉速越明显：

| 候选模型 | 生成速度 | VRAM占用 | 适用场景 |
|----------|----------|----------|----------|
| `qwen2.5:7b` | **65.8 tok/s**（最快） | ~4.7 GB | 速度优先场景，机器A未测试过此模型 |
| `qwen3.5:9b` | 53.9 tok/s | ~6.6 GB | 与机器A一致的微调基座/RAG默认，保持训练管线统一 |
| `qwen3.6:35b`（MoE） | 38.8 tok/s（机器A为53.6） | ~22 GB（CPU offload占比更高） | 质量优先，但比机器A慢约28%，仅建议离线/非交互场景 |
| `gemma4:12b` | 37.8 tok/s（机器A为58.8） | ~7.6 GB（完全放得进显存，无需offload） | 与`qwen3.6:35b`速度几乎打平，但体积小1/3且无offload，可作为质量优先的备选 |

> ⚠️ **微调需注意：** `qwen3.5:9b` QLoRA训练显存估算10-13GB，在12GB显卡上偏紧（上限已超出可用显存），建议减小 `cutoff_len` 或开启梯度检查点，且需实测验证是否OOM——本次基准仅覆盖推理，未覆盖训练。

详见 [RUNBOOK.md](RUNBOOK.md) "多机型基准与选型"一节，两机的正面对比分析见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)（结论：即使模型完全放得进显存，生成速度差距仍有约35%，根源是GPU算力/带宽差异而非显存不足）。

## 详细说明

请阅读 [RUNBOOK.md](RUNBOOK.md) 获取完整的操作步骤、参数说明和故障排除指南。
