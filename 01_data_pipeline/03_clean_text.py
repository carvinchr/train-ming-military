"""
Step 3: 文本清洗 — 去除噪声、页码、页眉页脚、多余空白
用法：python 01_data_pipeline/03_clean_text.py --input data/txt --output data/cleaned
"""
import argparse
import re
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# 需要过滤的行模式（页码、页眉页脚等）
NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                        # 纯数字行（页码）
    r"^\s*第[零一二三四五六七八九十百千]+[页章节]\s*$",  # 中文章节页码
    r"^\s*[-—–·•◆○●▪▸]+\s*\d+\s*[-—–·•◆○●▪▸]+\s*$",  # — 123 — 格式页码
    r"^\s*(版权所有|Copyright|All Rights Reserved|ISBN|CIP).+$",
    r"^\s*www\..+\.(com|cn|org)\s*$",       # 网址行
    r"^\s*[（(]\s*转下页\s*[)）]\s*$",
    r"^\s*[（(]\s*续[)）]\s*$",
]
NOISE_RE = [re.compile(p) for p in NOISE_PATTERNS]


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # 过滤噪声行
        if any(p.match(line) for p in NOISE_RE):
            continue
        # 去除行首尾空白，保留内容
        line = line.strip()
        cleaned.append(line)

    # 合并：连续空行缩为一个空行
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去除OCR常见错误：多余空格（中文字符之间）
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)

    # 修复OCR断字：行末连字符
    text = re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)

    # 统一引号、破折号
    text = text.replace(""", "「").replace(""", "」")
    text = text.replace("——", "—")

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/txt")
    parser.add_argument("--output", default="data/cleaned")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="文件最小字符数，过短文件跳过")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [f for f in input_dir.glob("*.txt") if not f.name.startswith("_")]
    logger.info(f"共 {len(txt_files)} 个文本文件待清洗")

    skipped = 0
    for txt_path in tqdm(txt_files, desc="清洗文本"):
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)

        if len(cleaned) < args.min_chars:
            logger.warning(f"跳过 {txt_path.name}（清洗后仅 {len(cleaned)} 字符）")
            skipped += 1
            continue

        out_path = output_dir / txt_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        logger.debug(f"{txt_path.name}: {len(raw)} → {len(cleaned)} 字符")

    logger.info(f"✅ 清洗完成，跳过 {skipped} 个文件")


if __name__ == "__main__":
    main()
