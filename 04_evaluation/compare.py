"""
评估 Step 4: 汇总对比报告 — 基座 vs RAG vs 微调
用法：python 04_evaluation/compare.py \
        --results results/baseline.json results/rag.json results/finetuned.json
"""
import argparse
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def load_result(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_summary_table(results: list):
    table = Table(title="模型对比汇总", box=box.ROUNDED, show_header=True)
    table.add_column("系统", style="bold cyan")
    table.add_column("LLM评分 (0-10)")
    table.add_column("ROUGE-L")
    table.add_column("上下文相关性")
    table.add_column("答案忠实度")
    table.add_column("平均延迟(秒)")
    table.add_column("题目数")

    for r in results:
        s = r["summary"]
        table.add_row(
            s.get("model") or s.get("system", "?"),
            str(s.get("avg_judge_score", "—")),
            str(s.get("avg_rougeL", "—")),
            str(s.get("avg_context_relevance", "—")),
            str(s.get("avg_faithfulness", "—")),
            str(s.get("avg_latency_sec", "—")),
            str(s.get("total_questions", 0)),
        )

    console.print(table)


def print_per_category(results: list):
    """按类别分析各系统表现"""
    categories = set()
    for r in results:
        for d in r.get("details", []):
            categories.add(d.get("category", ""))

    table = Table(title="按题目类别分析（LLM评分）", box=box.SIMPLE)
    table.add_column("类别")
    for r in results:
        s = r["summary"]
        name = s.get("model") or s.get("system", "?")
        table.add_column(name)

    for cat in sorted(categories):
        row = [cat]
        for r in results:
            items = [d for d in r.get("details", []) if d.get("category") == cat]
            if items:
                scores = [d.get("judge", {}).get("total", 0) for d in items]
                avg = sum(scores) / len(scores)
                row.append(f"{avg:.1f}")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


def find_best_worst(results: list):
    """找出每个系统表现最好和最差的题目"""
    console.print("\n[bold]各系统表现极值分析:[/bold]")
    for r in results:
        s = r["summary"]
        name = s.get("model") or s.get("system", "?")
        details = r.get("details", [])
        if not details:
            continue

        scored = [(d, d.get("judge", {}).get("total", 0)) for d in details]
        scored.sort(key=lambda x: x[1])

        worst = scored[0]
        best = scored[-1]

        console.print(f"\n  [cyan]{name}[/cyan]")
        console.print(f"    最高分 [{best[1]:.1f}]: {best[0]['question'][:60]}")
        console.print(f"    最低分 [{worst[1]:.1f}]: {worst[0]['question'][:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=[
        "results/baseline.json", "results/rag.json", "results/finetuned.json"
    ])
    parser.add_argument("--output", default="results/comparison_report.json")
    args = parser.parse_args()

    results = []
    for path in args.results:
        p = Path(path)
        if not p.exists():
            console.print(f"[yellow]跳过不存在的文件: {path}[/yellow]")
            continue
        results.append(load_result(path))

    if not results:
        console.print("[red]没有找到评估结果文件[/red]")
        return

    console.print("\n" + "=" * 60)
    console.print("  明代军事装备大模型 — 评估对比报告", style="bold")
    console.print("=" * 60 + "\n")

    print_summary_table(results)
    print_per_category(results)
    find_best_worst(results)

    # 保存汇总
    comparison = {
        "systems": [r["summary"] for r in results],
        "recommendation": "根据评分选择最优系统部署",
    }
    Path(args.output).write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n✅ 完整报告已保存: {args.output}")


if __name__ == "__main__":
    main()
