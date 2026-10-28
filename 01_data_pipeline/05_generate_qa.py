"""
Step 5: 用LLM自动从文本块生成Q&A对，作为SFT训练数据
支持结构化JSON输出、断点续跑（checkpoint跳过）、问题去重。
用法：python 01_data_pipeline/05_generate_qa.py \
        --input data/chunks --output data/qa_pairs \
        --model qwen3.6:35b --questions-per-chunk 3
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from pathlib import Path as _Path
from typing import Dict, List, Set

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(_Path(__file__).resolve().parent.parent))
from config import settings

SYSTEM_PROMPT = """你是一位专精明代军事史的学术助手。
你的任务是根据给定的史料原文，生成高质量的问答对，用于训练专业大模型。

要求：
1. 问题必须基于原文内容，具体明确，涉及明代军事装备、战术、制度等专业知识
2. 答案必须忠实于原文，用学术语言表述，不要凭空捏造
3. 避免过于简单的是非题，优先生成需要解释、比较、分析的问题
4. 避免与已生成过的重复问题

输出格式：JSON对象 {"qa": [{"question": "...", "answer": "..."}, ...]}
"""

USER_TEMPLATE = """请根据以下明代史料原文，生成 {n} 个高质量问答对：

---原文开始---
{text}
---原文结束---

直接输出JSON对象，形如 {{"qa": [...]}}，不要有其他内容："""


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", "", q.lower().strip())


def parse_qa(content: str) -> List[Dict]:
    """优先解析JSON对象（response_format），回退到JSON数组。"""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("qa"), list):
            return obj["qa"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    return []


def generate_qa_for_chunk(
    client: OpenAI,
    chunk: Dict,
    model: str,
    n_questions: int,
    max_retries: int = 3,
) -> List[Dict]:
    prompt = USER_TEMPLATE.format(text=chunk["text"], n=n_questions)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            qa_list = parse_qa(content)
            if not qa_list:
                logger.warning(f"无法解析JSON（尝试 {attempt+1}）: {content[:100]}")
                time.sleep(1)
                continue

            results = []
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if q and a and len(a) > 20:
                    results.append({
                        "source": chunk["source"],
                        "chunk_id": chunk["id"],
                        "question": q,
                        "answer": a,
                    })
            return results

        except Exception as e:
            logger.error(f"API调用失败（尝试 {attempt+1}）: {e}")
            time.sleep(2)

    return []


def load_checkpoint(output_file: Path) -> Set[str]:
    """从已有输出文件读取已处理的 chunk_id（断点续跑）。"""
    done = set()
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(qa, dict) and qa.get("chunk_id"):
                done.add(qa["chunk_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.chunks_dir))
    parser.add_argument("--output", default=str(settings.qa_dir))
    parser.add_argument("--model", default=settings.qa_gen_model)
    parser.add_argument("--base-url", default=settings.llm_base_url,
                        help="OpenAI兼容API地址（Ollama或vLLM）")
    parser.add_argument("--api-key", default=settings.llm_api_key)
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="限制处理块数（测试用）")
    parser.add_argument("--skip-short", type=int, default=100,
                        help="跳过字符数少于此值的块")
    parser.add_argument("--overwrite", action="store_true",
                        help="忽略checkpoint，重新生成（默认续跑）")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_qa.jsonl"

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    chunks_file = Path(args.input) / "all_chunks.jsonl"
    if chunks_file.exists():
        chunks = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        chunks = []
        for f in Path(args.input).glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunks.append(json.loads(line))

    chunks = [c for c in chunks if c.get("char_count", len(c.get("text", ""))) >= args.skip_short]
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    done = set() if args.overwrite else load_checkpoint(output_file)
    seen: Set[str] = set()
    if not args.overwrite and done:
        logger.info(f"checkpoint: 已处理 {len(done)} 个块，跳过续跑")

    logger.info(
        f"共 {len(chunks)} 个块，每块生成 {args.questions_per_chunk} 问，"
        f"目标 {len(chunks) * args.questions_per_chunk} 条Q&A"
    )

    generated = 0
    mode = "w" if args.overwrite else "a"
    with output_file.open(mode, encoding="utf-8") as f_out:
        for chunk in tqdm(chunks, desc="生成Q&A"):
            if chunk["id"] in done:
                continue

            qa_list = generate_qa_for_chunk(
                client, chunk, args.model,
                n_questions=args.questions_per_chunk,
            )

            # 去重后追加写入（flush保证断点安全）
            new_items = []
            for qa in qa_list:
                key = normalize_question(qa["question"])
                if key in seen:
                    continue
                seen.add(key)
                new_items.append(qa)

            for qa in new_items:
                f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
            f_out.flush()
            generated += len(new_items)
            time.sleep(0.2)  # 避免API限速

    logger.info(f"✅ 本次新增 {generated} 条Q&A → {output_file}")

    from collections import Counter
    final = [json.loads(l) for l in output_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    source_counts = Counter(qa["source"] for qa in final)
    for src, count in sorted(source_counts.items()):
        logger.info(f"  {src}: {count} 条")
    logger.info(f"  总计: {len(final)} 条（去重后）")


if __name__ == "__main__":
    main()
