"""04_chunk_text.py: sentence boundaries, token budget, overlap, stable SHA-256 IDs."""
import hashlib

from conftest import load_script

chunk_mod = load_script("01_data_pipeline/04_chunk_text.py")


def test_split_sentences_chinese_boundaries():
    text = "第一句。第二句！第三句？第四句；第五句\n第六句"
    sentences = chunk_mod.split_sentences(text)
    assert sentences == [
        "第一句。", "第二句！", "第三句？", "第四句；", "第五句", "第六句",
    ]


def test_detect_chapter():
    assert chunk_mod.detect_chapter("第二章 火器与装备") == "第二章"
    assert chunk_mod.detect_chapter("三、神机营编制") == "三、"
    assert chunk_mod.detect_chapter("1. 佛朗机铳") == "1."
    assert chunk_mod.detect_chapter("普通段落，无章节标记") == ""


def test_chunk_respects_token_budget_and_boundaries():
    # tokenizer=None → count_tokens == char count; budget acts as char budget
    text = "一二三四五六七八九十。" * 30  # each sentence 11 tokens
    chunks = chunk_mod.chunk_text(text, "book", None, chunk_tokens=50, overlap_tokens=22)
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] <= 50
        # chunk text is assembled from whole sentences, ends on a boundary
        assert c["text"].endswith("。")


def test_overlap_carries_whole_sentences():
    # each sentence is 6 chars; budget 12 fits exactly 2, overlap 6 carries the last one
    text = "句子甲内容。句子乙内容。句子丙内容。句子丁内容。"
    chunks = chunk_mod.chunk_text(text, "book", None, chunk_tokens=12, overlap_tokens=6)
    assert len(chunks) >= 2
    assert chunks[0]["text"] == "句子甲内容。句子乙内容。"
    assert chunks[1]["text"].startswith("句子乙内容。")
    # overlap consists of whole sentences carried from the previous chunk
    assert "句子乙内容。" in chunks[0]["text"]
    assert "句子甲内容。" not in chunks[1]["text"]


def test_no_overlap_when_overlap_tokens_zero():
    text = "句子甲内容。句子乙内容。句子丙内容。"
    chunks = chunk_mod.chunk_text(text, "book", None, chunk_tokens=12, overlap_tokens=0)
    assert chunks[0]["text"] == "句子甲内容。句子乙内容。"
    assert chunks[1]["text"] == "句子丙内容。"
    assert "句子乙内容" not in chunks[1]["text"]


def test_long_sentence_hard_split():
    long_sentence = "字" * 120  # single sentence > budget
    chunks = chunk_mod.chunk_text(long_sentence, "book", None, chunk_tokens=50, overlap_tokens=10)
    assert len(chunks) == 3
    assert sum(c["token_count"] for c in chunks) == 120


def test_chunk_id_is_stable_sha256():
    text = "这是一个测试句子。"
    chunk = chunk_mod._make_chunk([text], "book", "第一章", None)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    assert chunk["id"] == expected
    # same content → same ID across runs
    again = chunk_mod._make_chunk([text], "other_book", "第二章", None)
    assert again["id"] == chunk["id"]


def test_chapter_tracking_across_paragraphs():
    # small budget → one chunk per paragraph so chapters can be checked individually
    text = "第一章 总论。\n首段句子。\n\n普通段落句子。\n\n第二章 装备。\n新章句子。"
    chunks = chunk_mod.chunk_text(text, "book", None, chunk_tokens=7, overlap_tokens=0)
    chapters = [c["chapter"] for c in chunks]
    assert chapters[0] == "第一章"
    assert any(c == "第一章" for c in chapters[1:3])
    assert chapters[-1] == "第二章"


def test_source_and_counts_in_metadata():
    text = "测试句子。"
    chunks = chunk_mod.chunk_text(text, "mybook", None, chunk_tokens=512, overlap_tokens=64)
    c = chunks[0]
    assert c["source"] == "mybook"
    assert c["token_count"] == len(c["text"])
    assert c["char_count"] == len(c["text"])
