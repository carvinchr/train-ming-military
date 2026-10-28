"""
评估 Step 2: 评估模型（基座或微调后）在测试集上的表现
使用 ROUGE + LLM-as-Judge 两种方法
用法：python 04_evaluation/eval_model.py \
        --model-url http://localhost:11434/v1 \
        --model qwen3.5:9b \
        --testset data/testset.jsonl \
        --output results/baseline.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

try:
    import jieba
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning("rouge-score 未安装，跳过ROUGE评分")

JUDGE_SYSTEM = """你是一位明代军事史专家，负责评估大模型回答的质量。

评分维度（每项0-10分）：
1. 准确性（Accuracy）：事实是否正确，是否存在错误信息
2. 完整性（Completeness）：是否覆盖了问题的主要方面
3. 相关性（Relevance）：回答是否切题，是否包含无关信息
4. 专业性（Expertise）：是否使用了准确的历史术语和专业表述

请输出JSON格式：
{
  "accuracy": 分数,
  "completeness": 分数,
  "relevance": 分数,
  "expertise": 分数,
  "total": 总分(四项平均),
  "reasoning": "简要评分说明"
}
"""

JUDGE_USER = """问题：{question}

参考答案：{reference}

待评估的模型回答：{answer}

请按照要求评分（JSON格式）："""


def get_model_answer(client: OpenAI, model: str, question: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def llm_judge(judge_client: OpenAI, judge_model: str,
              question: str, reference: str, answer: str) -> Dict:
    prompt = JUDGE_USER.format(
        question=question, reference=reference, answer=answer
    )
    try:
        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        import re
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"LLM Judge失败: {e}")
    return {"accuracy": 0, "completeness": 0, "relevance": 0, "expertise": 0, "total": 0, "reasoning": "评分失败"}


def rouge_score(reference: str, hypothesis: str) -> Dict:
    if not HAS_ROUGE:
        return {}
    # 中文ROUGE需要分词
    def tokenize_zh(text):
        return " ".join(jieba.cut(text))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(tokenize_zh(reference), tokenize_zh(hypothesis))
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:9b", help="被评估的模型")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--testset", default="data/testset.jsonl")
    parser.add_argument("--output", default="results/baseline.json")
    parser.add_argument("--judge-model", default="qwen3.6:35b",
                        help="LLM-as-Judge使用的模型（建议用回答最详尽的候选模型，见benchmark/report.md）")
    parser.add_argument("--system", default=None, help="系统提示词文件路径")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.model_url, api_key=args.api_key)

    system_prompt = None
    if args.system:
        system_prompt = Path(args.system).read_text(encoding="utf-8")

    test_items = [
        json.loads(l) for l in Path(args.testset).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # 只评估有参考答案的题目
    test_items = [t for t in test_items if t.get("reference_answer", "").strip()]
    logger.info(f"评估 {len(test_items)} 道题（模型: {args.model}）")

    results = []
    for item in tqdm(test_items, desc="评估中"):
        start = time.time()
        try:
            answer = get_model_answer(client, args.model, item["question"], system_prompt)
        except Exception as e:
            logger.error(f"生成答案失败 [{item['id']}]: {e}")
            answer = ""

        latency = round(time.time() - start, 2)
        rouge = rouge_score(item["reference_answer"], answer)
        judge = llm_judge(client, args.judge_model,
                          item["question"], item["reference_answer"], answer)

        result = {
            "id": item["id"],
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "question": item["question"],
            "reference": item["reference_answer"],
            "answer": answer,
            "latency_sec": latency,
            "rouge": rouge,
            "judge": judge,
        }
        results.append(result)
        time.sleep(0.5)

    # 汇总统计
    if results:
        avg_total = sum(r["judge"].get("total", 0) for r in results) / len(results)
        avg_rouge_l = sum(r["rouge"].get("rougeL_f", 0) for r in results) / len(results)
        summary = {
            "model": args.model,
            "total_questions": len(results),
            "avg_judge_score": round(avg_total, 2),
            "avg_rougeL": round(avg_rouge_l, 4),
        }
    else:
        summary = {"model": args.model, "total_questions": 0}

    output = {"summary": summary, "details": results}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n📊 评估结果 ({args.model}):")
    logger.info(f"   平均LLM评分: {summary.get('avg_judge_score', 'N/A')} / 10")
    logger.info(f"   平均ROUGE-L: {summary.get('avg_rougeL', 'N/A')}")
    logger.info(f"   结果已保存: {args.output}")


if __name__ == "__main__":
    main()
