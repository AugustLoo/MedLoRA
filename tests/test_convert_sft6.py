import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import convert_sft6 as C  # noqa: E402


def img(color, size=(8, 8)):
    return Image.new("RGB", size, color)


def test_vqa_row_closed_uses_eval_prompt():
    row = C.vqa_row(" Is there a fracture? ", "Yes", "/x/a.png")
    assert row["messages"][0]["content"] == "<image>Is there a fracture?\nAnswer with yes or no only."
    assert row["messages"][1]["content"] == "Yes"
    assert row["images"] == ["/x/a.png"]


def test_vqa_row_open_uses_eval_prompt():
    row = C.vqa_row("What organ is this?", "liver", "/x/b.png")
    assert row["messages"][0]["content"].endswith("\nAnswer with a single word or short phrase.")


def test_medqa_row_matches_eval_format():
    row = C.medqa_row("Q?", {"A": "a", "B": "b", "C": "c", "D": "d"}, " c ")
    assert row["messages"][0]["content"] == (
        "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer with the option's letter from the given choices directly.")
    assert row["messages"][1]["content"] == "C"
    assert "images" not in row


def test_options_dict_accepts_string():
    assert C.options_dict("{'A': 'x', 'B': 'y'}") == {"A": "x", "B": "y"}


def test_pneumonia_row_matches_eval_question():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from medvlm.prompts import SLAKE_CLOSED
    assert C.PNEUMONIA_Q == SLAKE_CLOSED.format(question="Does this chest X-ray show pneumonia?")
    assert C.pneumonia_row(1, "/p.png")["messages"][1]["content"] == "yes"
    assert C.pneumonia_row(0, "/p.png")["messages"][1]["content"] == "no"


def test_image_key_same_pixels_same_key_any_mode():
    gray = Image.new("L", (8, 8), 100)
    assert C.image_key(gray) == C.image_key(gray.convert("RGB"))
    assert C.image_key(img("red")) != C.image_key(img("blue"))


def test_sample_is_deterministic_and_capped():
    xs = list(range(100))
    assert C.sample(xs, 10, 42) == C.sample(xs, 10, 42)
    assert len(C.sample(xs, 10, 42)) == 10
    assert C.sample(xs, 0, 42) == xs and C.sample(xs, 500, 42) == xs


def test_dedupe_keeps_first():
    kept, dup = C.dedupe([("a", 1), ("b", 2), ("a", 3)], key=lambda t: t[0])
    assert kept == [("a", 1), ("b", 2)] and dup == 1


def test_as_int_handles_lists():
    assert C.as_int([1]) == 1 and C.as_int(0) == 0


def test_build_vqa_drops_test_overlap_and_counts_shared_images(tmp_path):
    a, b, c = img("red"), img("green"), img("blue")
    train = [{"image": a, "question": "Is it red?", "answer": "yes"},       # 与测试同图同题 → 剔除
             {"image": a, "question": "What color?", "answer": "red"},      # 同图不同题 → 保留, 计入共用图
             {"image": b, "question": "Is it green?", "answer": "yes"},
             {"image": b, "question": "Is it green?", "answer": "yes"}]     # 训练内重复 → 去重
    test = [{"image": a, "question": "is it  RED?", "answer": "yes"}, {"image": c, "question": "x", "answer": "y"}]
    rows, st = C.build_vqa(train, test, tmp_path, n=0, seed=42)
    assert st["train_total"] == 4 and st["duplicates_dropped"] == 1
    assert st["same_image_and_question_as_test_dropped"] == 1 and st["used"] == 2
    assert st["images_shared_with_test"] == 1 and st["used_rows_on_test_images"] == 1
    assert st["used_closed"] == 1
    for r in rows:
        assert Path(r["images"][0]).is_file()


def test_build_vqa_samples_n(tmp_path):
    train = [{"image": img((i, 0, 0)), "question": f"q{i}", "answer": "a"} for i in range(20)]
    rows, st = C.build_vqa(train, [], tmp_path, n=5, seed=42)
    assert len(rows) == 5 and st["used"] == 5
    assert len(list(tmp_path.iterdir())) == 5  # 只为抽中的样本存图


def test_build_medqa_drops_test_questions():
    opts = {"A": "1", "B": "2", "C": "3", "D": "4"}
    train = [{"question": "Q1", "options": opts, "answer_idx": "A"},
             {"question": "Q2", "options": opts, "answer_idx": "B"}]
    test = [{"question": " q1 ", "options": opts, "answer_idx": "A"}]
    rows, st = C.build_medqa(train, test, n=0, seed=42)
    assert st["same_question_as_test_dropped"] == 1 and st["used"] == 1
    assert rows[0]["messages"][1]["content"] == "B"


def test_build_pneumonia_balanced_and_no_test_images(tmp_path):
    train = ([{"image": img((i, 1, 1)), "label": [1]} for i in range(10)]
             + [{"image": img((i, 2, 2)), "label": [0]} for i in range(4)])
    test_images = [img((0, 1, 1))]  # 与训练第一张肺炎片同图 → 剔除
    rows, st = C.build_pneumonia(train, test_images, tmp_path, n=6, seed=42)
    assert st["same_image_as_test_dropped"] == 1
    assert st["used_pneumonia"] == 3 and st["used_normal"] == 3
    answers = sorted(r["messages"][1]["content"] for r in rows)
    assert answers == ["no", "no", "no", "yes", "yes", "yes"]
