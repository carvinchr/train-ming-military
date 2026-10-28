"""
Step 1: 从PDF书籍提取纯文本
支持：数字PDF (fitz快速提取) + 扫描PDF (自动降级到OCR提示)
用法：python 01_data_pipeline/01_extract_pdf.py --input data/raw --output data/txt
"""
import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger
from tqdm import tqdm


def is_scan_pdf(doc: fitz.Document, sample_pages: int = 5) -> bool:
    """检测是否为扫描PDF（文字提取量极少则判定为扫描件）"""
    text_len = 0
    pages = min(sample_pages, len(doc))
    for i in range(pages):
        text_len += len(doc[i].get_text("text").strip())
    avg = text_len / pages if pages else 0
    return avg < 50  # 每页平均字符数<50则认为是扫描件


def extract_pdf(pdf_path: Path, output_dir: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    stem = pdf_path.stem

    if is_scan_pdf(doc):
        logger.warning(f"[扫描PDF] {pdf_path.name} → 请用 02_ocr_images.py 处理")
        return {"file": pdf_path.name, "status": "scan_pdf", "pages": len(doc)}

    pages_text = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages_text.append({"page": page_num, "text": text})

    # 合并为一个txt文件
    out_file = output_dir / f"{stem}.txt"
    full_text = "\n\n".join(p["text"] for p in pages_text)
    out_file.write_text(full_text, encoding="utf-8")

    logger.info(f"✓ {pdf_path.name} → {len(pages_text)} 页，{len(full_text)} 字符")
    return {"file": pdf_path.name, "status": "ok", "pages": len(pages_text), "chars": len(full_text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="原始PDF目录")
    parser.add_argument("--output", default="data/txt", help="输出TXT目录")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    if not pdfs:
        logger.warning(f"在 {input_dir} 中未找到PDF文件")
        return

    logger.info(f"共找到 {len(pdfs)} 个PDF文件")
    results = []
    for pdf_path in tqdm(pdfs, desc="提取PDF"):
        try:
            result = extract_pdf(pdf_path, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"处理 {pdf_path.name} 失败: {e}")
            results.append({"file": pdf_path.name, "status": "error", "error": str(e)})

    # 同时处理data/raw中的txt文件（直接复制）
    for txt_path in input_dir.glob("*.txt"):
        dest = output_dir / txt_path.name
        dest.write_text(txt_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        logger.info(f"✓ 复制 {txt_path.name}")

    report_path = output_dir / "_extraction_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"\n报告已保存到 {report_path}")


if __name__ == "__main__":
    main()
