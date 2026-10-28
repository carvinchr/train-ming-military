"""
RAG Step 3: 交互式测试RAG服务（核心函数可被 pytest 集成测试复用，见 tests/test_rag_integration.py）
用法：python 02_rag/test_rag.py --server http://localhost:8080
      python 02_rag/test_rag.py --server http://localhost:8080 --question "虎蹲炮的射程是多少？"
"""
import argparse
import json

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

PRESET_QUESTIONS = [
    "明朝初期的火器发展状况如何？",
    "佛朗机铳是如何传入明朝的？",
    "明代神机营的编制与装备是什么？",
    "明代火绳枪与欧洲同期火枪有何差异？",
    "红夷大炮在明末战争中发挥了什么作用？",
    "明军骑兵的主要武器和装备有哪些？",
    "戚继光的鸳鸯阵使用了哪些武器配合？",
    "明代水军的战船类型和火炮配置如何？",
]


def ask(server: str, question: str) -> dict:
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{server}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()


def search_only(server: str, query: str, top_k: int = 5) -> list:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{server}/v1/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()["results"]


def interactive_mode(server: str):
    console.print(Panel("明代军事装备 RAG 问答系统", style="bold blue"))
    console.print(
        "输入 [bold]quit[/bold] 退出 | [bold]search:[问题][/bold] 仅检索 | "
        "[bold]preset[/bold] 显示预设问题\n"
    )

    while True:
        try:
            question = console.input("[bold green]请输入问题:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "preset":
            for i, q in enumerate(PRESET_QUESTIONS, 1):
                console.print(f"  {i}. {q}")
            continue
        if question.startswith("search:"):
            query = question[7:].strip()
            results = search_only(server, query)
            for r in results:
                console.print(Panel(
                    f"[dim]来源: {r['source']} | 相关度: {r['score']}[/dim]\n\n{r['text']}",
                    title="检索结果",
                ))
            continue

        console.print("[dim]正在检索并生成...[/dim]")
        try:
            result = ask(server, question)
            answer = result["choices"][0]["message"]["content"]
            console.print(Panel(Markdown(answer), title=f"[bold]问：{question[:50]}[/bold]"))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--question", default=None, help="单次提问模式")
    parser.add_argument("--run-preset", action="store_true", help="批量跑所有预设问题")
    args = parser.parse_args()

    if args.question:
        result = ask(args.server, args.question)
        answer = result["choices"][0]["message"]["content"]
        print(answer)
    elif args.run_preset:
        results = []
        for q in PRESET_QUESTIONS:
            console.print(f"\n[bold blue]问：[/bold blue]{q}")
            try:
                res = ask(args.server, q)
                ans = res["choices"][0]["message"]["content"]
                console.print(ans[:500] + ("..." if len(ans) > 500 else ""))
                results.append({"question": q, "answer": ans})
            except Exception as e:
                console.print(f"[red]失败: {e}[/red]")
        with open("results/rag_preset_test.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        console.print("\n✅ 结果已保存到 results/rag_preset_test.json")
    else:
        interactive_mode(args.server)


if __name__ == "__main__":
    main()
