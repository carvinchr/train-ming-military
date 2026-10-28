"""
SFT Step 1: 将Q&A数据转换为LLaMA-Factory标准格式
支持格式: alpaca (默认) / sharegpt
用法：python 03_finetune/prepare_sft.py --qa-dir data/qa_pairs --output-dir data/sft
"""
import argparse
import json
import random
from pathlib import Path

from loguru import logger

SYSTEM_PROMPT = (
    "你是一位专精明代军事史的学术助手，擅长明朝军事装备、兵器制造、军事制度、"
    "战术战法等领域。请基于扎实的历史文献，提供准确、专业的学术解答。"
)


def load_qa_pairs(qa_dir: Path) -> list:
    all_qa = []
    qa_file = qa_dir / "all_qa.jsonl"
    if qa_file.exists():
        for line in qa_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_qa.append(json.loads(line))
    else:
        for f in qa_dir.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_qa.append(json.loads(line))
    return all_qa


def to_alpaca_format(qa: dict) -> dict:
    """LLaMA-Factory alpaca格式"""
    return {
        "instruction": qa["question"],
        "input": "",
        "output": qa["answer"],
        "system": SYSTEM_PROMPT,
    }


def to_sharegpt_format(qa: dict) -> dict:
    """LLaMA-Factory sharegpt格式（多轮对话）"""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": qa["question"]},
            {"from": "gpt", "value": qa["answer"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", default="data/qa_pairs")
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--format", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--min-answer-len", type=int, default=30,
                        help="过滤过短回答")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pairs = load_qa_pairs(Path(args.qa_dir))
    logger.info(f"加载 {len(qa_pairs)} 条Q&A对")

    # 过滤
    qa_pairs = [
        qa for qa in qa_pairs
        if len(qa.get("answer", "")) >= args.min_answer_len
        and len(qa.get("question", "")) >= 5
    ]
    logger.info(f"过滤后 {len(qa_pairs)} 条")

    # 打乱
    random.seed(args.seed)
    random.shuffle(qa_pairs)

    # 转换格式
    converter = to_alpaca_format if args.format == "alpaca" else to_sharegpt_format
    converted = [converter(qa) for qa in qa_pairs]

    # 分割训练/验证集
    split = int(len(converted) * args.train_ratio)
    train_data = converted[:split]
    val_data = converted[split:]

    # 保存
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    train_path.write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_data, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"✅ 训练集: {len(train_data)} 条 → {train_path}")
    logger.info(f"✅ 验证集: {len(val_data)} 条 → {val_path}")

    # 注册到LLaMA-Factory dataset_info.json
    dataset_info_path = Path("03_finetune/LLaMA-Factory/data/dataset_info.json")
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["ming_military_train"] = {
        "file_name": str(train_path.resolve()),
        "formatting": args.format,
        "columns": {"prompt": "instruction", "response": "output"} if args.format == "alpaca" else {},
    }
    dataset_info["ming_military_val"] = {
        "file_name": str(val_path.resolve()),
        "formatting": args.format,
    }

    if dataset_info_path.exists():
        dataset_info_path.write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"已更新 {dataset_info_path}")
    else:
        logger.warning("LLaMA-Factory未安装，手动将以下内容加入 data/dataset_info.json：")
        print(json.dumps({"ming_military_train": dataset_info["ming_military_train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
