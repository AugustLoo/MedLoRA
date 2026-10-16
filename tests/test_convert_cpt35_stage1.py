import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import convert_cpt35_stage1 as C  # noqa: E402


def words(s):  # 测试用的「token」计数: 英文按词, 中文按字
    return len(s.split()) + sum(1 for ch in s if "一" <= ch <= "鿿")


def test_clean_text_drops_page_numbers_and_squeezes_space():
    t = C.clean_text("Line  one\r\n\r\n12\r\nLine\t two\n\n\n\nEnd")
    assert t == "Line one\n\nLine two\n\nEnd"


def test_paragraphs_fall_back_to_lines_when_no_blank_lines():
    text = "\n".join(f"line {i}" for i in range(20))
    assert len(C.paragraphs(text)) == 20


def test_paragraphs_join_wrapped_english_lines_and_chinese_without_space():
    assert C.paragraphs("a wrapped\nline\n\n中文\n段落") == ["a wrapped line", "中文段落"]


def test_pack_respects_limit_and_keeps_all_text():
    paras = [" ".join(["w"] * n) for n in (30, 50, 40, 70, 10)]
    chunks = C.pack(paras, words, limit=100)
    assert all(words(c) <= 100 for c in chunks)
    assert sum(words(c) for c in chunks) == sum(words(p) for p in paras)


def test_pack_splits_overlong_paragraph_by_sentences_then_chars():
    long_en = " ".join(["This is a sentence."] * 60)          # 240 词
    chunks = C.pack([long_en], words, limit=50)
    assert len(chunks) >= 5 and all(words(c) <= 50 for c in chunks)
    long_zh = "字" * 230                                       # 没有句号, 只能按字符切
    chunks = C.pack([long_zh], words, limit=100)
    assert all(words(c) <= 100 for c in chunks) and sum(len(c) for c in chunks) == 230


def test_pack_merges_or_drops_tiny_tail():
    chunks = C.pack([" ".join(["w"] * 95), "tail"], words, limit=100, min_tokens=32)
    assert chunks == [" ".join(["w"] * 95) + "\n\ntail"]
    assert C.pack(["tiny"], words, limit=100, min_tokens=32) == []


def test_take_budget_stops_at_budget_and_zero_means_all():
    docs = [(f"d{i}", " ".join(["w"] * 80)) for i in range(10)]
    got = list(C.take_budget(docs, words, limit=50, budget=200))
    assert sum(n for *_, n in got) <= 200 and sum(n for *_, n in got) >= 150
    assert sum(n for *_, n in C.take_budget(docs, words, limit=50, budget=0)) == 800


def test_split_val_is_by_document_and_deterministic():
    docs = [(f"d{i}", [f"c{i}.{j}" for j in range(3)], 30) for i in range(40)]
    tr1, va1 = C.split_val(docs, 0.05, random.Random(1))
    tr2, va2 = C.split_val(docs, 0.05, random.Random(1))
    assert va1 == va2 and va1
    names_tr, names_va = {d[0] for d in tr1}, {d[0] for d in va1}
    assert not names_tr & names_va and len(names_tr) + len(names_va) == 40


def test_pieces_of_one_paragraph_are_rejoined_without_blank_lines():
    zh = "这是第一句。" * 30                                  # 180 字, 超过 limit, 会按句切开
    chunks = C.pack([zh], words, limit=100)
    assert all("\n" not in c for c in chunks) and "".join(chunks) == zh
    en = " ".join(["Short sentence here."] * 40)
    chunks = C.pack([en], words, limit=50)
    assert all("\n" not in c for c in chunks)


def test_final_check_splits_chunks_that_still_exceed_limit():
    def count(s):  # 分隔符比估计的更「贵」
        return len(s.split()) + 3 * s.count("\n\n")
    paras = [" ".join(["w"] * 20) for _ in range(10)]
    assert all(count(c) <= 60 for c in C.pack(paras, count, limit=60))


def test_looks_like_ad():
    assert C.looks_like_ad("本诊所专业整形美容, 价格优惠, 欢迎咨询电话预约, 地址在市中心。" * 3)
    assert not C.looks_like_ad("肝囊肿一般没有症状, 囊肿长大到一定程度可能压迫胃肠道而引起上腹不适。" * 5)


def test_split_val_takes_only_the_tail_of_a_long_book():
    docs = [(f"book{i}", [f"b{i}.{j}" for j in range(300)], 3000) for i in range(5)]   # 5 本书, 共 1,500 段
    tr, va = C.split_val(docs, 0.01, random.Random(0))
    assert sum(len(c) for _, c, _ in va) == 15                     # 1% , 而不是一整本 300 段
    assert sum(len(c) for _, c, _ in tr) + 15 == 1500 and len(tr) == 5
    (vname, vchunks, _), = va
    book = vname.split("#")[0]
    head = next(c for n, c, _ in tr if n == book)
    assert head + vchunks == next(c for n, c, _ in docs if n == book)  # 前面训练, 末尾验证


def test_strip_attribution_keeps_content_lines():
    text = ("The thalamus relays sensory signals.\n"
            "Gray675. Provided by: Wikipedia. Located at: en.wikipedia.org. License: Public Domain: No Known Copyright\n"
            "CC BY-SA: Attribution-ShareAlike\n"
            "It sits above the midbrain.")
    assert C.strip_attribution(text) == "The thalamus relays sensory signals.\nIt sits above the midbrain."
