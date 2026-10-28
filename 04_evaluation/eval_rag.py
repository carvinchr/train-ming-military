"""
评估 Step 3: 评估RAG系统表现
额外评估：检索相关性（Context Recall）、答案忠实度（Faithfulness）
用法：python 04_evaluation/eval_rag.py \
        --rag-server http://localhost:8080 \
        --testset data/testset.jsonl \
        --output results/rag.json
"""
import argparse
import json
import time
from pathlib import Path

import httpx
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


def ask_rag(server: str, question: str, timeout: int = 120) -> dict:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{server}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": question}]},
        )
        resp.raise_for_status()
        return resp.json()


def search_rag(server: str, query: str, top_k: int = 5) -> list:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{server}/v1/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()["results"]


def score_context_relevance(judge_client: OpenAI, judge_model: str,
                             question: str, contexts: list) -> float:
    """评估检索到的上下文与问题的相关性（0-1）"""
    context_texts = "\n\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(contexts))
    prompt = f"""以下是针对问题检索到的文本片段，请评估这些片段与问题的相关程度。

问题：{question}

检索到的文本：
{context_texts}

请输出一个0到1之间的小数，表示相关性（1=高度相关，0=完全无关）。只输出数字，不要解释。"""

    try:
        resp = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=10,
        )
        score_str = resp.choices[0].message.content.strip()
        return float(score_str)
    except Exception:
        return 0.5


def score_faithfulness(judge_client: OpenAI, judge_model: str,
                        answer: str, contexts: list) -> float:
    """评估答案是否忠实于检索到的上下文（0-1）"""
    context_texts = "\n\n".join(f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(contexts))
    prompt = f"""请评估以下答案是否基于给定的上下文内容，是否存在超出上下文的捏造信息。

上下文：
{context_texts}

答案：
{answer[:500]}

请输出0到1之间的小数（1=完全忠实于上下文，0=大量内容不在上下文中）。只输出数字。"""

    try:
        resp = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=10,
        )
        return float(resp.choices[0].message.content.strip())
    except Exception:
        return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-server", default="http://localhost:8080")
    parser.add_argument("--testset", default="data/testset.jsonl")
    parser.add_argument("--output", default="results/rag.json")
    parser.add_argument("--judge-url", default="http://localhost:11434/v1")
    parser.add_argument("--judge-model", default="qwen3.6:35b")
    parser.add_argument("--judge-key", default="ollama")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    judge_client = OpenAI(base_url=args.judge_url, api_key=args.judge_key)

    test_items = [
        json.loads(l) for l in Path(args.testset).read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("reference_answer", "").strip()
    ]
    logger.info(f"评估RAG系统，共 {len(test_items)} 道题")

    results = []
    for item in tqdm(test_items, desc="RAG评估"):
        start = time.time()
        try:
            rag_resp = ask_rag(args.rag_server, item["question"])
            answer = rag_resp["choices"][0]["message"]["content"]
            contexts = rag_resp.get("retrieved_contexts", [])
        except Exception as e:
            logger.error(f"RAG请求失败 [{item['id']}]: {e}")
            answer = ""
            contexts = []

        latency = round(time.time() - start, 2)

        ctx_relevance = score_context_relevance(
            judge_client, args.judge_model, item["question"], contexts
        )
        faithfulness = score_faithfulness(
            judge_client, args.judge_model, answer, contexts
        )

        results.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "question": item["question"],
            "reference": item["reference_answer"],
            "answer": answer,
            "latency_sec": latency,
            "num_contexts": len(contexts),
            "context_relevance": round(ctx_relevance, 3),
            "faithfulness": round(faithfulness, 3),
            "contexts": [{"source": c.get("source"), "score": c.get("score")} for c in contexts],
        })
        time.sleep(0.5)

    if results:
        avg_ctx = sum(r["context_relevance"] for r in results) / len(results)
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_latency = sum(r["latency_sec"] for r in results) / len(results)
        summary = {
            "system": "RAG",
            "rag_server": args.rag_server,
            "total_questions": len(results),
            "avg_context_relevance": round(avg_ctx, 3),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_latency_sec": round(avg_latency, 2),
        }
    else:
        summary = {"system": "RAG", "total_questions": 0}

    output = {"summary": summary, "details": results}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("\n📊 RAG评估结果:")
    logger.info(f"   上下文相关性: {summary.get('avg_context_relevance', 'N/A')}")
    logger.info(f"   答案忠实度:   {summary.get('avg_faithfulness', 'N/A')}")
    logger.info(f"   平均延迟:     {summary.get('avg_latency_sec', 'N/A')} 秒")
    logger.info(f"   结果已保存:   {args.output}")


if __name__ == "__main__":
    main()
