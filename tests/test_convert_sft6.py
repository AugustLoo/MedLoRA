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
