"""
Step 2: OCR处理扫描书籍（扫描PDF / 图片）
使用 PaddleOCR（中文识别率远高于Tesseract）
用法：python 01_data_pipeline/02_ocr_images.py --input data/raw --output data/txt
"""
import argparse
import json
import tempfile
from pathlib import Path

import fitz
from loguru import logger
from tqdm import tqdm

try:
    from paddleocr import PaddleOCR
    OCR_ENGINE = "paddle"
except ImportError:
    import pytesseract
    from PIL import Image
    OCR_ENGINE = "tesseract"
    logger.warning("PaddleOCR未安装，降级使用Tesseract（中文识别质量较差）")


def init_ocr():
    if OCR_ENGINE == "paddle":
        return PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=True,
            show_log=False,
        )
    return None


def ocr_image_paddle(ocr_engine, image_path: str) -> str:
    result = ocr_engine.ocr(image_path, cls=True)
    if not result or not result[0]:
        return ""
    lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
    return "\n".join(lines)


def ocr_image_tesseract(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="chi_sim+chi_tra")


def process_scan_pdf(pdf_path: Path, output_dir: Path, ocr_engine) -> dict:
    doc = fitz.open(str(pdf_path))
    all_text = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_num, page in enumerate(tqdm(doc, desc=f"OCR {pdf_path.name}", leave=False)):
            # 高分辨率渲染
            mat = fitz.Matrix(2.5, 2.5)  # 250% 缩放，提高OCR精度
            pix = page.get_pixmap(matrix=mat)
            img_path = f"{tmp_dir}/page_{page_num:04d}.png"
            pix.save(img_path)

            if OCR_ENGINE == "paddle":
                text = ocr_image_paddle(ocr_engine, img_path)
            else:
                text = ocr_image_tesseract(img_path)

            if text.strip():
                all_text.append(f"[第{page_num+1}页]\n{text.strip()}")

    full_text = "\n\n".join(all_text)
    out_file = output_dir / f"{pdf_path.stem}.txt"
    out_file.write_text(full_text, encoding="utf-8")

    logger.info(f"✓ OCR {pdf_path.name} → {len(doc)} 页，{len(full_text)} 字符")
    return {"file": pdf_path.name, "status": "ok", "pages": len(doc), "chars": len(full_text)}


def process_image(img_path: Path, output_dir: Path, ocr_engine) -> dict:
    if OCR_ENGINE == "paddle":
        text = ocr_image_paddle(ocr_engine, str(img_path))
    else:
        text = ocr_image_tesseract(str(img_path))

    out_file = output_dir / f"{img_path.stem}.txt"
    out_file.write_text(text, encoding="utf-8")
    logger.info(f"✓ OCR {img_path.name} → {len(text)} 字符")
    return {"file": img_path.name, "status": "ok", "chars": len(text)}


def is_scan_pdf(doc: fitz.Document) -> bool:
    text_len = sum(len(doc[i].get_text("text").strip()) for i in range(min(5, len(doc))))
    return (text_len / min(5, len(doc))) < 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/txt")
    parser.add_argument("--force", action="store_true", help="强制重新OCR已处理文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_engine = init_ocr()
    results = []

    # 处理扫描PDF
    for pdf_path in input_dir.glob("*.pdf"):
        out_file = output_dir / f"{pdf_path.stem}.txt"
        if out_file.exists() and not args.force:
            logger.info(f"跳过 {pdf_path.name}（已存在）")
            continue
        doc = fitz.open(str(pdf_path))
        if is_scan_pdf(doc):
            result = process_scan_pdf(pdf_path, output_dir, ocr_engine)
            results.append(result)

    # 处理独立图片
    img_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in img_suffixes:
            out_file = output_dir / f"{img_path.stem}.txt"
            if out_file.exists() and not args.force:
                continue
            result = process_image(img_path, output_dir, ocr_engine)
            results.append(result)

    if not results:
        logger.info("没有需要OCR处理的文件（或全部已处理）")
    else:
        report = output_dir / "_ocr_report.json"
        report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"OCR完成，报告: {report}")


if __name__ == "__main__":
    main()
