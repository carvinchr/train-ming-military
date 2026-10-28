# 明朝军事装备大模型训练 — 完整运行手册

> **目标**：以 **Qwen3.5:9B** 为微调基座，通过 RAG + LoRA 微调，构建一个能精准回答明朝军事装备问题的专域大模型；经机器A（4090 Laptop 16GB，[报告](benchmark/reports/rtx4090-laptop-16gb.md)）与机器B（RTX 3060 12GB，[报告](benchmark/reports/rtx3060-12gb.md)）两台机器实测，9B 模型在两台机器上均可部署，但**具体候选模型排名因机器而异**（12GB显存下 `qwen2.5:7b` 反而最快），另有 **qwen3.6:35b MoE** 作为质量优先的RAG部署候选（不参与微调，两机均可运行但速度不同）；详见"多机型基准与选型"一节。
> **环境**：Windows 11 + WSL2 Ubuntu 22.04 + CUDA（主力训练环境）；推理基准测试覆盖 RTX 4090 Laptop（16GB VRAM）与 RTX 3060（12GB VRAM）两台机器
> **日期**：2026-09

---

## 架构总览

```
原始书籍 (PDF/扫描件)
    │
    ▼
[阶段1] 数据管线
    ├── PDF文本提取
    ├── OCR扫描识别
    ├── 文本清洗分块
    └── LLM自动生成Q&A对
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[阶段2] RAG系统               [阶段3] SFT微调
  ChromaDB向量库                LLaMA-Factory
  bge-m3嵌入模型                QLoRA (4-bit)
  FastAPI服务                   Qwen3.5:9B（微调基座）
    │                                  │
    └──────────┬───────────────────────┘
               ▼
        [阶段4] 评估对比
          RAG vs 微调 vs RAG+微调
               │
               ▼
        [阶段5] 部署上线
          Ollama + Open-WebUI
          候选模型（4090实测）：
          qwen3.5:9b / qwen3.6:35b MoE
```

### 三种方案对比

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| 纯RAG | 无需训练，快速上线，易更新 | 依赖检索质量，推理慢 | 快速验证 |
| 纯LoRA微调 | 推理快，回答风格好 | 无法引用原文，可能幻觉 | 通用增强 |
| **RAG + LoRA** | **最准确，可溯源** | **工程复杂** | **生产推荐** |

**建议路径**：先做RAG（1-2天），验证效果后做LoRA微调（3-5天），最终合并部署。

---

## 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3060 (12GB VRAM) | RTX 4090 (16GB+) |
| RAM | 32GB | 64GB |
| 磁盘 | 100GB SSD | 200GB NVMe |
| CUDA | 12.1+ | 12.4 |

> 微调基座已由 Qwen2.5:14B 改为 **Qwen3.5:9B**（原14B方案QLoRA显存需求18-22GB，需24GB+显卡）。9B QLoRA训练显存估算：9B × 4bit ≈ 4.5GB模型 + 梯度/优化器 ≈ **总计10-13GB**，16GB显卡（如 RTX 4090 Laptop）即可完成训练，无需24GB+显卡；12GB显卡（如 RTX 3060）**推理已验证可行**（见下方基准数据），但**训练尚未实测**，估算区间上限（13GB）已超出可用显存，需调小 `cutoff_len` 或开启梯度检查点，见下方"多机型基准与选型"章节。

### 推理候选模型显存占用（4090 Laptop 16GB 实测）

微调完成后的**线上推理/部署**同样不要求24GB显卡：在16GB显存的 RTX 4090 Laptop 上实测（详见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)），以下两个候选模型均可正常运行：

| 候选模型 | 参数量 | VRAM占用 | 生成速度 | 说明 |
|----------|--------|----------|----------|------|
| `qwen3.5:9b` | 9B（密集，微调基座） | ~6.6 GB | 82.6 tok/s | 剩余显存多，可同时跑其他模型/服务 |
| `qwen3.6:35b`（MoE，不参与微调） | 35B | ~23 GB（部分CPU卸载） | 53.6 tok/s | 16GB显卡需offload，回答更详尽 |

因此 16GB 级别显卡足以承担本项目的**训练+推理全流程**（基座已改为9B）。

## 多机型基准与选型

本项目已在两台硬件不同的机器上做过基准测试，**模型选型需按实际机器的显存/算力单独确认，不能直接照搬另一台机器的推荐**。两台机器的直接对比分析见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)。

### 机器 A — RTX 4090 Laptop（16GB VRAM）

- 完整基准报告：[benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)
- 结论：微调基座 + 日常RAG 用 `qwen3.5:9b`；质量优先RAG（不微调）用 `qwen3.6:35b` MoE
- 详见本文档前面各节

### 机器 B — RTX 3060（12GB VRAM）

- 硬件：CPU Intel i7-12700F（12核20线程）/ RAM 64GB / GPU RTX 3060 12GB VRAM
- 完整基准报告：[benchmark/reports/rtx3060-12gb.md](benchmark/reports/rtx3060-12gb.md)

| 候选模型 | 生成速度 | VRAM占用 | 说明 |
|----------|----------|----------|------|
| `qwen2.5:7b` | **65.8 tok/s**（本机最快） | ~4.7 GB | 机器A未测试过，速度优先场景可用 |
| `qwen3.5:9b` | 53.9 tok/s | ~6.6 GB | 与机器A保持一致的微调基座/RAG默认 |
| `qwen3.6:35b`（MoE） | 38.8 tok/s（较机器A的53.6 tok/s下降约28%） | ~22 GB（`ollama ps`显示58%CPU/42%GPU offload） | 仍可用但明显更慢，只建议离线/非交互场景 |
| `gemma4:12b` | 37.8 tok/s（较机器A的58.8 tok/s下降约36%） | ~7.6 GB（完全放得进显存，无offload） | 后补测试；与`qwen3.6:35b`速度几乎打平但体积小1/3，质量优先场景的备选 |
| `qwen2.5:14b` | 34.7 tok/s | ~9 GB | 机器A未测试过；本机上是5个模型中最慢的（14B密集模型 vs 35B MoE，MoE反而更快） |

**结论：** 排名与机器A**不同**——12GB显存下 `qwen2.5:7b` 反而最快，且模型越大掉速越明显（35B MoE从53.6降到38.8 tok/s，降幅约28%）。RAG部署仍建议 `qwen3.5:9b`（与机器A一致，保持训练管线统一），速度优先可换 `qwen2.5:7b`，质量优先仍用 `qwen3.6:35b`（`gemma4:12b`速度相近且更省显存，可作平替）。

> ⚠️ **微调注意事项：** 本机是否能顺利完成 `qwen3.5:9b` 的 QLoRA 训练**尚未实测**（本次基准只测了推理）。9B QLoRA 训练显存估算10-13GB，而本机仅有12GB，估算区间上限已超出可用显存，实际训练前建议先减小 `qwen35_lora.yaml` 中的 `cutoff_len`（如降到1024）或开启 `gradient_checkpointing: true`，并密切观察 `nvidia-smi` 显存占用，防止OOM。

> 📊 **两机对比：** `qwen3.5:9b`/`gemma4:12b`/`qwen3.6:35b` 三个模型在两台机器上都测过，生成速度分别下降34.8%/35.7%/27.6%。前两个完全放得进两张卡的显存，降幅却和显存不足的`qwen3.6:35b`一样大甚至更大——说明降速**并非**由显存不足导致，根源是两张GPU本身的算力/带宽差异；细节与更多发现见 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)。

---

## 阶段0：环境搭建

### 0.1 启用 WSL2 + Ubuntu

```powershell
# Windows PowerShell (管理员)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

### 0.2 WSL Ubuntu 环境初始化

```bash
# 进入WSL
wsl

# 更新系统
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 poppler-utils tesseract-ocr tesseract-ocr-chi-sim

# 创建项目Python虚拟环境
cd /mnt/d/ollama/train-ming-military
python3.11 -m venv .venv
source .venv/bin/activate

# 安装PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r 00_setup/requirements.txt
```

### 0.3 安装 LLaMA-Factory（微调框架）

```bash
cd /mnt/d/ollama/train-ming-military
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 03_finetune/LLaMA-Factory
cd 03_finetune/LLaMA-Factory
pip install -e ".[torch,metrics,bitsandbytes,qwen]"
```

### 0.4 安装 Ollama（推理服务）

```bash
# WSL 内安装
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &  # 后台启动

# 拉取基座模型（用于Q&A自动生成/RAG基线测试）
ollama pull qwen2.5:14b

# 拉取部署候选模型（机器A 4090实测，见 benchmark/reports/rtx4090-laptop-16gb.md）
ollama pull qwen3.5:9b
ollama pull qwen3.6:35b
```

### 0.5 Docker 服务（ChromaDB + Open-WebUI）

```bash
# 在 Windows PowerShell 中
cd D:\ollama\train-ming-military
docker-compose -f 00_setup/docker-compose.yml up -d
```

---

## 阶段1：数据管线

### 1.1 目录结构

```
data/
├── raw/          # 原始书籍文件 (PDF/扫描图片)
├── txt/          # 提取后的纯文本
├── cleaned/      # 清洗后文本
├── chunks/       # 分块后的JSONL
├── qa_pairs/     # 生成的Q&A对
└── sft/          # 最终SFT训练集
```

```bash
mkdir -p data/{raw,txt,cleaned,chunks,qa_pairs,sft}
```

### 1.2 将书籍放入 data/raw/

支持格式：`.pdf`、`.txt`、`.epub`、扫描图片(`.jpg`/`.png`)

### 1.3 运行数据管线

```bash
source .venv/bin/activate

# Step 1: PDF文本提取
python 01_data_pipeline/01_extract_pdf.py --input data/raw --output data/txt

# Step 2: OCR处理扫描书籍（如有）
python 01_data_pipeline/02_ocr_images.py --input data/raw --output data/txt

# Step 3: 文本清洗
python 01_data_pipeline/03_clean_text.py --input data/txt --output data/cleaned

# Step 4: token感知分块（默认参数来自 .env：CHUNK_TOKENS=512 / OVERLAP_TOKENS=64，CLI可覆盖）
python 01_data_pipeline/04_chunk_text.py \
    --input data/cleaned \
    --output data/chunks \
    --chunk-tokens 512 \
    --overlap-tokens 64

# Step 5: 自动生成Q&A对（需要Ollama在运行；默认模型来自 .env：QA_GEN_MODEL=qwen3.6:35b）
# 默认断点续跑（跳过已处理的chunk）；--overwrite 忽略checkpoint重新生成
python 01_data_pipeline/05_generate_qa.py \
    --input data/chunks \
    --output data/qa_pairs \
    --model qwen3.6:35b \
    --questions-per-chunk 3
```

---

## 阶段2：RAG系统

### 2.1 构建向量索引

```bash
# 构建ChromaDB索引（使用bge-m3中文嵌入；默认值来自 .env，CLI可覆盖）
python 02_rag/build_index.py \
    --chunks data/chunks \
    --db-path data/chroma_db \
    --embedding-model BAAI/bge-m3
```

> 索引构建为原子操作：先写入 `ming_military_new` 集合再原子换名，不会污染旧索引；chunk 元数据（source/chapter）随向量一并持久化。

### 2.2 启动RAG服务

路径、模型、RAG参数、CORS 均从 `.env`（`config.py`）读取，脚本参数只剩 `--port` / `--host`：

```bash
cp 00_setup/.env.example .env   # 首次：按需调整环境变量
python 02_rag/rag_server.py --port 8080
```

- 嵌入模型：`EMBED_MODEL`（默认 `BAAI/bge-m3`）
- 重排序：`RERANKER_MODEL`（默认 `BAAI/bge-reranker-v2-m3`，CrossEncoder，检索候选按 rerank 得分重排）
- LLM：`LLM_MODEL`（默认 `qwen3.5:9b`）+ `LLM_BASE_URL`
- 每次请求可用 `top_k` / `rerank_top_k` 字段覆盖全局默认（`TOP_K=5` / `RERANK_TOP_K=4`）
- `/health` 返回服务状态与集合大小

> 机器A（4090 16GB）环境下也可将 `--model` 换成实测候选模型 `qwen3.5:9b`（速度优先）或 `qwen3.6:35b`（质量优先，需搭配 `/no_think` 与更大 `num_ctx`，见阶段5.2/5.3），详见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)。

### 2.3 测试RAG效果

```bash
python 02_rag/test_rag.py \
    --server http://localhost:8080 \
    --question "明代火绳枪与欧洲同期火枪有何差异？"
```

### 2.4 自动化测试

单元测试位于 `tests/`（分块边界/重叠/ID、问题归一化去重、config默认值与env覆盖），不需要任何外部服务：

```bash
pip install -r requirements-dev.txt
pytest                    # 全量
pytest -m "not integration"  # 仅单元测试（CI用的就是这个）
```

`tests/test_rag_integration.py` 复用 `02_rag/test_rag.py` 的 `ask()`/`search_only()` 做集成测试：服务在 `http://localhost:8080`（可用 `RAG_TEST_SERVER` 覆盖）可达时跑，不可达自动 skip。CI 配置见 `.github/workflows/ci.yml`（py_compile + ruff + pytest）。

---

## 阶段3：SFT微调

### 3.1 准备SFT数据集

```bash
python 03_finetune/prepare_sft.py \
    --qa-dir data/qa_pairs \
    --output-dir data/sft \
    --train-ratio 0.9
```

### 3.2 下载基座模型权重（HuggingFace格式）

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载Qwen3.5-9B-Instruct
huggingface-cli download Qwen/Qwen3.5-9B-Instruct \
    --local-dir /mnt/d/models/Qwen3.5-9B-Instruct \
    --resume-download
```

> 或使用已有的vLLM服务（192.168.1.209）做数据生成，训练仍需本地HF权重。

### 3.3 启动LoRA训练

```bash
cd 03_finetune/LLaMA-Factory

llamafactory-cli train ../qwen35_lora.yaml
```

训练过程监控（另开终端）：

```bash
# 查看GPU使用
watch -n 2 nvidia-smi

# 查看训练日志
tail -f /mnt/d/ollama/train-ming-military/outputs/training.log
```

### 3.4 合并LoRA权重

```bash
llamafactory-cli export \
    --model_name_or_path /mnt/d/models/Qwen3.5-9B-Instruct \
    --adapter_name_or_path outputs/lora_weights \
    --template qwen3 \
    --export_dir /mnt/d/models/ming-military-9b \
    --export_size 4 \
    --export_legacy_format false
```

---

## 阶段4：评估

### 4.1 创建测试集（人工标注20-50题）

```bash
python 04_evaluation/create_testset.py --output data/testset.jsonl
```

手动编辑 `data/testset.jsonl`，填写标准答案。

### 4.2 三路对比评估

```bash
# 评估纯基座模型
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model qwen3.5:9b \
    --testset data/testset.jsonl \
    --output results/baseline.json

# 评估RAG系统
python 04_evaluation/eval_rag.py \
    --rag-server http://localhost:8080 \
    --testset data/testset.jsonl \
    --output results/rag.json

# 评估微调后模型（需先部署）
python 04_evaluation/eval_model.py \
    --model-url http://localhost:11434/v1 \
    --model ming-military \
    --testset data/testset.jsonl \
    --output results/finetuned.json

# 汇总对比报告
python 04_evaluation/compare.py \
    --results results/baseline.json results/rag.json results/finetuned.json
```

---

## 阶段5：部署

> 5.1 是"合并LoRA权重"后的自定义模型部署路径；若不做微调，直接用 RAG + 基座模型上线，可跳到 5.2/5.3 — 两节分别对应机器A（4090）环境实测确定的两个候选基座：`qwen3.6:35b`（质量优先）与 `qwen3.5:9b`（速度优先，推荐日常使用），选型依据见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)。

### 5.1 注册到Ollama

```bash
# 创建Modelfile
cp 05_deployment/Modelfile /mnt/d/models/ming-military-9b/Modelfile

cd /mnt/d/models/ming-military-9b
ollama create ming-military -f Modelfile

# 验证
ollama run ming-military "明朝虎蹲炮的射程和威力如何？"
```

### 5.2 Open-WebUI 配置（qwen3.6:35b MoE + 知识库）

浏览器打开 `http://localhost:3000`

#### 5.2.1 配置 RAG 全局参数

**Admin Panel → Settings → Documents**

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding Model | `BAAI/bge-m3` | 中文检索最佳，首次自动下载 |
| Chunk Size | `1024` | 配合大上下文模型 |
| Chunk Overlap | `128` | |
| Top K | `5` | 先用小值确认稳定，再逐步增加到 20 |

#### 5.2.2 创建知识库

**Workspace → Knowledge → `+` New Knowledge**

```
Name:        明代军事装备知识库
Description: 明朝军事装备、火器、兵制专著文献
```

上传 `data/cleaned/` 中的 TXT 文件，等待向量化完成。

#### 5.2.3 创建专属模型

**Workspace → Models → `+` New Model**

**基本信息：**
```
Model ID:   ming-military-35b
Name:       明代军事助手 (qwen3.6:35b)
Base Model: qwen3.6:35b
```

**Knowledge 选项卡：** 勾选 `明代军事装备知识库`

**System Prompt：**
```
你是一位专精明代军事史的学术助手，擅长明朝军事装备、火器制造、兵制、战术等领域。

回答时请：
1. 优先基于检索到的史料原文，保持学术准确性
2. 引用原文时注明文献来源
3. 对不确定的信息如实说明，不捏造史实
4. 使用规范的历史术语和学术语言

/no_think
```

> ⚠️ **`/no_think` 必须保留**（见调整记录 2026-09-13）

**Advanced Parameters：**
```
num_ctx:         16384
temperature:     0.7
top_p:           0.9
repeat_penalty:  1.1
```

> ⚠️ **`num_ctx = 16384` 必须设置**（见调整记录 2026-09-13）

点击 **Save** 保存，新建对话选择该模型即可使用。

#### 5.2.4 验证连接

```cmd
docker exec ming-webui curl -s http://host.docker.internal:11434/api/tags
```

有 JSON 输出说明 Open-WebUI → Ollama 连接正常。

---

### 5.3 Open-WebUI 配置（qwen3.5:9b 密集模型 — 推荐日常使用）

#### 模型信息

| 参数 | 值 |
|------|-----|
| 架构 | `qwen35`（密集模型，非 MoE） |
| 参数量 | 9.7B |
| 文件大小 | 6.6 GB (Q4_K_M) |
| 最大上下文 | 262,144 tokens |
| VRAM 占用 | ~7 GB，剩余 ~9 GB 可用于 KV Cache |
| 实测速度 | **82.6 tok/s**（benchmark 2026-09-12） |

**与 qwen3.6:35b 的取舍：**

| 项目 | qwen3.5:9b | qwen3.6:35b MoE |
|------|-----------|-----------------|
| 生成速度 | **82 tok/s** | 53 tok/s |
| VRAM 占用 | **6.6 GB** | 23 GB |
| 可用 KV Cache | **~9 GB** | ~1 GB |
| 回答深度 | 一般 | **更详尽** |
| 可同时运行其他模型 | ✅ | ❌ |

> 9B 模型释放了大量显存给 KV Cache，实际可用上下文反而更长、更稳定。

---

#### 5.3.1 RAG 全局参数（与 5.2.1 相同，可直接复用）

**Admin Panel → Settings → Documents** 设置不变：

| 参数 | 值 |
|------|-----|
| Embedding Model | `BAAI/bge-m3` |
| Chunk Size | `1024` |
| Chunk Overlap | `128` |
| Top K | `15` ← 比 35b 可以设更高，VRAM 更宽裕 |

---

#### 5.3.2 创建专属模型

**Workspace → Models → `+` New Model**

**基本信息：**
```
Model ID:   ming-military-9b
Name:       明代军事助手 (qwen3.5:9b)
Base Model: qwen3.5:9b
```

**Knowledge 选项卡：** 勾选 `明代军事装备知识库`

**System Prompt：**
```
你是一位专精明代军事史的学术助手，擅长明朝军事装备、火器制造、兵制、战术等领域。

回答时请：
1. 优先基于检索到的史料原文，保持学术准确性
2. 引用原文时注明文献来源
3. 对不确定的信息如实说明，不捏造史实
4. 使用规范的历史术语和学术语言
```

**Advanced Parameters：**
```
temperature:     0.7
top_p:           0.9
repeat_penalty:  1.1
```

> ✅ qwen3.5:9b 使用默认 num_ctx 即可正常运行，无需像 qwen3.6:35b 那样强制设为 16384。如希望支持更长对话上下文可选填 `num_ctx = 8192`。

---

#### 5.3.3 与 qwen3.6:35b 的配置差异说明

qwen3.5:9b 在 Open-WebUI 挂载知识库后**开箱即用**，不需要 qwen3.6:35b 所需的两项强制修复：

| 配置项 | qwen3.5:9b | qwen3.6:35b MoE |
|--------|-----------|-----------------|
| `/no_think` | 非必须 | **必须** |
| `num_ctx = 16384` | 非必须 | **必须** |
| 知识库挂载后正常输出 | ✅ 开箱即用 | ❌ 需修复后才可用 |

根本原因：qwen3.6:35b 是 MoE 架构，thinking 模式 + RAG 上下文会撑满默认 context 窗口导致无输出；qwen3.5:9b 是密集模型，上下文占用更小，不存在此问题。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| OOM (显存不足) | 减小 `per_device_train_batch_size` 到 1，增大 `gradient_accumulation_steps` |
| 训练loss不下降 | 检查数据格式，降低 `learning_rate` 到 1e-5 |
| OCR中文识别差 | 改用 PaddleOCR 替代 Tesseract |
| ChromaDB慢 | 改用 FAISS 本地索引，或部署 Milvus |
| 中文分词差 | chunk时按句号/换行分割，而非固定字符数 |
| RAG有来源但无输出 | num_ctx 不足，改为 16384；System Prompt 末尾加 `/no_think` |
| Open-WebUI 思考后无答案 | qwen3.6:35b thinking 模式占满上下文，同上两步修复 |
| 模型第一次响应极慢 | 正常，冷启动需 30~40 秒加载；在 System Prompt 中加预热请求 |

---

## 参考资源

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3.5 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B-Instruct)
- [ChromaDB Docs](https://docs.trychroma.com)
- [RAGAS 评估框架](https://docs.ragas.io)
- [BAAI/bge-m3 嵌入](https://huggingface.co/BAAI/bge-m3)
- [机器A（4090 Laptop 16GB）基准测试报告](benchmark/reports/rtx4090-laptop-16gb.md) — 候选模型（qwen3.5:9b / qwen3.6:35b）选型依据
- [机器B（RTX 3060 12GB）基准测试报告](benchmark/reports/rtx3060-12gb.md) — 排名因机器而异，`qwen2.5:7b`在此机器上更快

---

## 调整记录

### 2026-10-28 — 设计审查收尾：测试、CI、文档同步

- `tests/`（pytest）：分块（句子边界/整句重叠/SHA-256 ID/章节跟踪）、问题归一化去重与JSON回退解析、config默认值与env覆盖；`pytest.ini` 注册 `integration` 标记。
- `tests/test_rag_integration.py` 复用 `02_rag/test_rag.py` 的 `ask()`/`search_only()`，服务不可达时自动 skip。
- `.github/workflows/ci.yml`：py_compile 全量语法检查 + ruff（`ruff.toml`，行宽120，E/F/I规则）+ `pytest -m "not integration"`。
- 修复测试暴露的脚本问题：`load_checkpoint` 对非dict JSON行的 `AttributeError`。
- `run_pipeline.sh` 分块参数更名为 `--chunk-tokens/--overlap-tokens`；`build_and_serve.sh` 移除 `rag_server.py` 不支持的 `--db-path/--llm-base-url/--model` 参数（现由 `.env` 提供）。
- 依赖审计：补声明 `pytesseract`、`pydantic`（均被脚本直接import）；`langchain*`、`marker-pdf`、`typer`、`datasets/peft/trl/bitsandbytes`、`pandas`、`scikit-learn`、`pyyaml`、`sentencepiece` 确认保留（为训练管线预留，已在 requirements.txt 注明）。

### 2026-09-13 — 基于4090基准测试确定候选部署模型

**背景：** 在 RTX 4090 Laptop（16GB VRAM，机器A）上对 7 个本地 Ollama 模型做了基准测试（`benchmark/run_benchmark.py`，完整数据见 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)），覆盖生成速度、显存占用与回答质量。

**结论：** 本项目在 4090 环境下的部署候选确定为：

| 候选模型 | 理由 |
|----------|------|
| `qwen3.5:9b` | 实测 82.6 tok/s（次快），仅占 ~6.6GB 显存，剩余显存可用于更大 KV Cache，日常问答**开箱即用**（无需 `/no_think` 或调大 `num_ctx`） |
| `qwen3.6:35b`（MoE） | 实测 53.6 tok/s，回答最详尽（单次输出可达4000 tokens），适合深度分析与 SFT 数据生成，但需 `/no_think` + `num_ctx=16384` 两项修复才能在 Open-WebUI 挂载知识库后正常输出 |

`qwen3.8:latest`、`qwen3.6:27b` 因 thinking 模式拖慢至 12-14 tok/s、`gemma4:26b` 因 OOM 全部失败，均不作为候选。

具体部署步骤见阶段5.2（qwen3.6:35b）与阶段5.3（qwen3.5:9b）。

> **更新（同日）：** 微调基座已进一步由 `Qwen2.5-14B-Instruct` 改为 `Qwen3.5-9B-Instruct`，见下一条记录。

### 2026-09-13 — 微调基座由 Qwen2.5:14B 改为 Qwen3.5:9B

**原因：** 上一条记录中，`qwen3.5:9b` 已被确定为4090环境下的推理部署候选之一，但微调基座当时仍沿用 `Qwen2.5-14B-Instruct`，导致训练（需24GB+显卡）与推理部署（16GB即可）使用两套不同规格的硬件与模型，流程不统一。改为以 `Qwen3.5-9B-Instruct` 作为微调基座后：

- 训练与部署统一到同一个9B模型，同一张16GB显卡（如 RTX 4090 Laptop）即可完成QLoRA训练+合并+部署全流程，无需额外24GB+显卡。
- 9B QLoRA训练显存估算：9B × 4bit ≈ 4.5GB模型 + 梯度/优化器 ≈ **总计10-13GB**。
- `qwen3.6:35b` 保持为不参与微调的RAG-only质量优先候选（35B MoE 微调门槛高，且当前无微调收益验证）。

**受影响文件：** `03_finetune/qwen25_lora.yaml` 重命名为 `qwen35_lora.yaml`（model_name_or_path、template、run_name均已更新）、`train.sh`、`merge_lora.sh`（BASE_MODEL/MERGED_MODEL/template）、`05_deployment/deploy.sh`（MERGED_MODEL路径与基座回退逻辑）、`04_evaluation/eval_model.py`（--model与--judge-model默认值）、`04_evaluation/eval_rag.py`（--judge-model默认值，改为回答最详尽的`qwen3.6:35b`）、`01_data_pipeline/05_generate_qa.py`与`run_pipeline.sh`（QA生成默认模型改为`qwen3.6:35b`，取benchmark"SFT数据生成"推荐）、`02_rag/rag_server.py`与`build_and_serve.sh`（RAG默认模型改为`qwen3.5:9b`）。

### 2026-09-13 — 新增机器B（RTX 3060 12GB）基准测试，模型排名因机器而异

**背景：** 在第二台机器（CPU i7-12700F / RAM 64GB / GPU RTX 3060 12GB VRAM）上安装了 `qwen2.5:7b`，并对本机已安装的4个模型（`qwen2.5:7b`、`qwen3.5:9b`、`qwen3.6:35b`、`qwen2.5:14b`）运行了 `benchmark/run_benchmark.py`。

**处理方式：** `run_benchmark.py` 内部硬编码输出路径为 `benchmark/report.md` 和 `benchmark/raw/*.json`（按模型名命名，不区分机器），运行前先把机器A已有数据移开以免被覆盖：
- 机器A的原始报告移到 [benchmark/reports/rtx4090-laptop-16gb.md](benchmark/reports/rtx4090-laptop-16gb.md)
- 机器A的原始JSON数据移到 `benchmark/raw/rtx4090-laptop-16gb/`

脚本跑完后 `benchmark/report.md`（脚本手动补充Hardware/Key Findings/Recommendations等章节后）与 `benchmark/raw/*.json` 又手动归档为与机器A同样的命名格式：报告移到 `benchmark/reports/rtx3060-12gb.md`，原始JSON移到 `benchmark/raw/rtx3060-12gb/`，两台机器的目录结构完全对称。

**结果：** 12GB显存下模型排名与机器A（16GB）**不同**——`qwen2.5:7b`（65.8 tok/s）比`qwen3.5:9b`（53.9 tok/s）更快，且模型越大掉速越明显：`qwen3.6:35b` 从机器A的53.6 tok/s降到38.8 tok/s（约28%降幅，`ollama ps`显示CPU/GPU offload比例从机器A的多数GPU变为58%CPU/42%GPU）。详见 [benchmark/reports/rtx3060-12gb.md](benchmark/reports/rtx3060-12gb.md)"多机型基准与选型"及"Key Findings"章节。

**已知遗留问题：** `run_benchmark.py` 在Windows GBK控制台下打印`✅`等emoji时会抛出`UnicodeEncodeError`导致脚本在写完报告后以退出码1终止；报告和原始数据在崩溃前已正确写入，不影响结果正确性。已将崩溃的两行 `console.print` 改为纯ASCII文本修复该问题；但脚本硬编码的 `benchmark/report.md`/`benchmark/raw/` 输出路径本身仍需每次跑完后手动归档到按机器命名的子目录，尚未自动化。

### 2026-09-13 — 新增两机对比报告，加测 gemma4:12b

**对比报告：** 新增 [benchmark/reports/rtx4090-vs-rtx3060.md](benchmark/reports/rtx4090-vs-rtx3060.md)，把机器A、机器B两份报告里都测过的模型放在一起算涨跌幅。核心发现：完全放得进显存的模型（不需要CPU offload）反而降速比例更大更一致（约35%），比需要offload的`qwen3.6:35b`（降28%）降得更多——说明降速根源是两张GPU本身的算力/带宽差异，不是显存不够。

**加测 gemma4:12b：** 用户在机器B上另外装了 `gemma4:12b`（机器A之前测过，58.80 tok/s）。为避免重跑其余4个模型引入的随机波动污染已经写好的结论和百分比，没有重新跑整个 `run_benchmark.py`，而是写了个小脚本单独调用其 `benchmark_model()` 函数只测这一个模型，结果单独存档、再合并进各文档：
- 机器B实测 37.83 tok/s（较机器A降35.7%），几乎与`qwen3.6:35b`（38.79 tok/s）打平，但体积只有其1/3且无需offload
- 原始数据：`benchmark/raw/rtx3060-12gb/gemma4_12b.json`，同时补入该目录下的 `_all_results.json`
- 已更新：`benchmark/reports/rtx3060-12gb.md`（新增第5个模型的排名/明细/发现/推荐）、`benchmark/reports/rtx4090-vs-rtx3060.md`（新增第三个"两机都测过"的模型，强化"~35%算力差距"这一发现）、本文档与 README.md 的机器B候选模型表

### 2026-09-13 — Open-WebUI + qwen3.6:35b RAG 无输出问题修复

**现象：** 在 Open-WebUI 中使用 qwen3.6:35b 挂载知识库后，每次请求均显示"Thought for 10 seconds"并找到 4 个来源，但之后没有任何文字输出，仅显示 Follow up 建议。

**根本原因：** 两个因素叠加导致 context 溢出：

```
System Prompt   ≈  400 tokens
4个RAG chunks   ≈ 2000 tokens
Thinking过程    ≈  800 tokens
用户问题         ≈  100 tokens
─────────────────────────────
合计             ≈ 3300 tokens  >  默认 num_ctx 2048
```

模型完成 Thinking 后已无剩余 context 空间输出答案。

**修复措施：**

1. **关闭 Thinking 模式**：在模型 System Prompt 末尾添加 `/no_think`
2. **增大上下文窗口**：Advanced Parameters 中设置 `num_ctx = 16384`

**验证结果：** 两项同时生效后模型正常输出，速度恢复至约 53 tok/s。

**影响范围：** 所有在 Open-WebUI 中使用 Qwen3 系列 MoE 模型（qwen3.6:27b、qwen3.8:latest 等）并挂载知识库的配置，均需应用相同修复。
