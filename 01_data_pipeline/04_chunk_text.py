"""
Step 4: token 感知语义分块 — 将长文本切割成适合RAG和SFT的片段
策略：按章节边界跟踪元数据，按中文句子边界组装，不超过token预算；
块ID为内容SHA-256（可跨运行稳定去重）。
用法：python 01_data_pipeline/04_chunk_text.py --input data/cleaned --output data/chunks
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from config import settings

CHAPTER_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千\d]+[章节卷篇]|[一二三四五六七八九十]+、|\d+[\.\、])",
    re.MULTILINE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")

_TOKENIZER = None


def get_tokenizer(model_name: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"无法加载tokenizer（{e}），回退为字符估算")
    return _TOKENIZER


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def detect_chapter(text: str) -> str:
    m = CHAPTER_RE.search(text)
    return m.group(0).strip() if m else ""


def split_long_sentence(sentence: str, chunk_tokens: int, tokenizer) -> List[str]:
    """将超过token预算的单句硬切为多个piece"""
    if tokenizer is None:
        return [sentence[i : i + chunk_tokens] for i in range(0, len(sentence), chunk_tokens)]
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    return [tokenizer.decode(ids[i : i + chunk_tokens]) for i in range(0, len(ids), chunk_tokens)]


def _carry_overlap(sentences: List[str], overlap_tokens: int, tokenizer) -> List[str]:
    """从chunk尾部保留若干句子，使其token总和接近overlap_tokens"""
    if overlap_tokens <= 0 or not sentences:
        return []
    carried = []
    total = 0
    for s in reversed(sentences):
        t = count_tokens(s, tokenizer)
        if total + t > overlap_tokens and carried:
            break
        carried.insert(0, s)
        total += t
    return carried


def _make_chunk(sentences: List[str], source: str, chapter: str, tokenizer) -> Dict:
    text = "".join(sentences).strip()
    uid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return {
        "id": uid,
        "source": source,
        "chapter": chapter,
        "text": text,
        "token_count": count_tokens(text, tokenizer),
        "char_count": len(text),
    }


def chunk_text(
    text: str,
    source: str,
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[Dict]:
    """按句子边界组装chunk，超过token预算时输出；保留尾部句子作为overlap"""
    chunks = []
    buffer: List[str] = []
    buffer_tokens = 0
    current_chapter = detect_chapter(text)

    for para in [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]:
        chapter = detect_chapter(para) or current_chapter
        current_chapter = chapter

        for sentence in split_sentences(para):
            tokens = count_tokens(sentence, tokenizer)

            if tokens > chunk_tokens:
                if buffer:
                    chunks.append(_make_chunk(buffer, source, chapter, tokenizer))
                    buffer = _carry_overlap(buffer, overlap_tokens, tokenizer)
                    buffer_tokens = sum(count_tokens(s, tokenizer) for s in buffer)

                for piece in split_long_sentence(sentence, chunk_tokens, tokenizer):
                    chunks.append(_make_chunk([piece], source, chapter, tokenizer))
                buffer = []
                buffer_tokens = 0
                continue

            if buffer_tokens + tokens > chunk_tokens:
                chunks.append(_make_chunk(buffer, source, chapter, tokenizer))
                buffer = _carry_overlap(buffer, overlap_tokens, tokenizer)
                buffer_tokens = sum(count_tokens(s, tokenizer) for s in buffer)

            buffer.append(sentence)
            buffer_tokens += tokens

    if buffer:
        chunks.append(_make_chunk(buffer, source, current_chapter, tokenizer))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.data_dir / "cleaned"))
    parser.add_argument("--output", default=str(settings.chunks_dir))
    parser.add_argument("--chunk-tokens", type=int, default=settings.chunk_tokens)
    parser.add_argument("--overlap-tokens", type=int, default=settings.overlap_tokens)
    parser.add_argument("--tokenizer", default=settings.embed_model)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*.txt"))
    logger.info(
        f"分块 {len(txt_files)} 个文件，chunk_tokens={args.chunk_tokens}, "
        f"overlap_tokens={args.overlap_tokens}, tokenizer={args.tokenizer}"
    )

    total_chunks = 0
    all_chunks_path = output_dir / "all_chunks.jsonl"

    with all_chunks_path.open("w", encoding="utf-8") as all_f:
        for txt_path in tqdm(txt_files, desc="分块"):
            text = txt_path.read_text(encoding="utf-8")
            chunks = chunk_text(
                text,
                source=txt_path.stem,
                tokenizer=tokenizer,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
            )

            per_book_path = output_dir / f"{txt_path.stem}.jsonl"
            with per_book_path.open("w", encoding="utf-8") as f:
                for chunk in chunks:
                    line = json.dumps(chunk, ensure_ascii=False)
                    f.write(line + "\n")
                    all_f.write(line + "\n")

            total_chunks += len(chunks)
            logger.info(f"  {txt_path.name}: {len(chunks)} 块")

    logger.info(f"✅ 共生成 {total_chunks} 个块 → {all_chunks_path}")


if __name__ == "__main__":
    main()
