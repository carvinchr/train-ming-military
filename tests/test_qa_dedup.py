"""05_generate_qa.py: question normalization/dedup, JSON parsing fallback, checkpoint."""
import json
from pathlib import Path

from conftest import load_script

qa_mod = load_script("01_data_pipeline/05_generate_qa.py")


def test_normalize_question_dedup_key():
    a = qa_mod.normalize_question("  明朝 的火器 发展状况 如何？ ")
    b = qa_mod.normalize_question("明朝的火器发展状况如何？")
    assert a == b
    assert qa_mod.normalize_question("Test Question") == "testquestion"
    assert a == "明朝的火器发展状况如何？"


def test_parse_qa_object_format():
    content = '{"qa": [{"question": "Q1", "answer": "A1"}]}'
    assert qa_mod.parse_qa(content) == [{"question": "Q1", "answer": "A1"}]


def test_parse_qa_array_fallback():
    content = '[{"question": "Q1", "answer": "A1"}]'
    assert qa_mod.parse_qa(content) == [{"question": "Q1", "answer": "A1"}]


def test_parse_qa_regex_fallback_with_extra_text():
    content = "这是额外文字 [{\"question\": \"Q1\", \"answer\": \"A1\"}] 尾部噪声"
    parsed = qa_mod.parse_qa(content)
    assert parsed == [{"question": "Q1", "answer": "A1"}]


def test_parse_qa_invalid_returns_empty():
    assert qa_mod.parse_qa("完全不是JSON") == []
    assert qa_mod.parse_qa('{"qa": "不是列表"}') == []


def test_dedup_via_normalize():
    seen = set()
    questions = ["火器 发展?", "火器发展?", "火器 发展? "]
    unique = 0
    for q in questions:
        key = qa_mod.normalize_question(q)
        if key not in seen:
            seen.add(key)
            unique += 1
    assert unique == 1


def test_load_checkpoint(tmp_path: Path):
    output = tmp_path / "all_qa.jsonl"
    rows = [
        {"chunk_id": "abc123", "question": "Q", "answer": "A"},
        {"question": "no chunk_id"},          # skipped
        "",                                    # blank line skipped
        "malformed json line",                 # skipped
    ]
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    done = qa_mod.load_checkpoint(output)
    assert done == {"abc123"}


def test_load_checkpoint_missing_file(tmp_path: Path):
    assert qa_mod.load_checkpoint(tmp_path / "nope.jsonl") == set()
