import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import make_short_answer_controls as M  # noqa: E402

CLOSED = "\nAnswer with yes or no only."
OPEN = "\nAnswer with a single word or short phrase."


def s(q, a, img="/x/imgs/xmlab1/source.jpg", suffix=None):
    suffix = suffix if suffix is not None else (CLOSED if a.lower() in ("yes", "no") else OPEN)
    return {"messages": [{"role": "user", "content": "<image>" + q + suffix}, {"role": "assistant", "content": a}],
            "images": [img]}


def test_strip_format_removes_only_the_instruction_line():
    out = M.strip_format(s("Is the lung healthy?", "No"))
    assert out["messages"][0]["content"] == "<image>Is the lung healthy?"
    assert out["messages"][1]["content"] == "No" and out["images"] == ["/x/imgs/xmlab1/source.jpg"]


def test_strip_format_refuses_a_sample_without_the_instruction():
    with pytest.raises(ValueError):
        M.strip_format(s("Is the lung healthy?", "No", suffix=""))


def test_pick_is_reproducible_sorted_and_sized():
    a, b = M.pick(4919, 0.2, 42), M.pick(4919, 0.2, 42)
    assert a == b == sorted(a) and len(a) == 984 and len(set(a)) == 984
    assert M.pick(4919, 0.2, 43) != a


def test_rewrite_ok_needs_the_original_answer_kept():
    assert M.rewrite_ok("No", "No, the lung does not look healthy in this image.")
    assert not M.rewrite_ok("No", "The lung looks unhealthy in this image.")          # 丢了 No
    assert M.rewrite_ok("Left Lung", "The abnormality is located in the left lung.")
    assert not M.rewrite_ok("Left Lung", "The abnormality is in the right lung.")
    assert not M.rewrite_ok("CT", "CT")                                               # 没改成句子
    assert not M.rewrite_ok("CT", "This is a CT.\nIt shows the abdomen and the liver.")  # 多于一行


def test_rewrite_ok_matches_whole_words_only():
    assert not M.rewrite_ok("CT", "The picture shows the liver and its ducts clearly.")   # ducts 里的 ct 不算


def test_apply_rewrites_changes_prompt_and_answer_only_for_valid_ones():
    rows = [s("Is the lung healthy?", "No"), s("What modality is used?", "CT"), s("Which organ?", "Liver")]
    out, stats = M.apply_rewrites(rows, {0: "No, the lung does not look healthy.", 1: "It is an MRI scan."})
    assert out[0]["messages"][0]["content"] == "<image>Is the lung healthy?\nAnswer in one complete sentence."
    assert out[0]["messages"][1]["content"] == "No, the lung does not look healthy."
    assert out[1] == rows[1] and out[2] == rows[2]                # 改错的保持原样, 没点到的不动
    assert stats == {"requested": 2, "applied": 1, "rejected": 1}
    assert rows[0]["messages"][1]["content"] == "No"              # 不改传进来的数据


def test_rewrite_requests_carry_the_question_without_image_tag_or_instruction():
    rows = [s("Is the lung healthy?", "No")] * 5
    reqs = M.rewrite_requests(rows, 0.2, 42)
    assert len(reqs) == 1 and reqs[0]["question"] == "Is the lung healthy?" and reqs[0]["answer"] == "No"
    assert "Is the lung healthy?" in reqs[0]["prompt"] and "<image>" not in reqs[0]["prompt"]


def test_one_description_request_per_image_in_first_seen_order():
    rows = [s("q1", "a", "/x/1.jpg"), s("q2", "b", "/x/2.jpg"), s("q3", "c", "/x/1.jpg")]
    assert M.description_requests(rows) == [{"image": "/x/1.jpg"}, {"image": "/x/2.jpg"}]


def test_description_samples_use_the_fixed_prompt_and_skip_empty_text():
    out = M.description_samples([{"image": "/x/1.jpg", "text": "An axial CT of the abdomen. "}, {"image": "/x/2.jpg", "text": " "}])
    assert out == [{"messages": [{"role": "user", "content": "<image>" + M.DESCRIBE},
                                 {"role": "assistant", "content": "An axial CT of the abdomen."}], "images": ["/x/1.jpg"]}]


def test_sample_extra_is_reproducible_and_never_repeats():
    extra = [{"id": i} for i in range(50)]
    a = M.sample_extra(extra, 10, 42)
    assert a == M.sample_extra(extra, 10, 42) and len({x["id"] for x in a}) == 10
    with pytest.raises(ValueError):
        M.sample_extra(extra, 51, 42)


def test_generation_resumes_and_maps_image_paths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gen_control_text as G
    reqs = [{"idx": 3, "prompt": "a"}, {"idx": 9, "prompt": "b"}]
    assert G.todo(reqs, [{"idx": 3, "text": "x"}], "idx") == [{"idx": 9, "prompt": "b"}]
    assert G.map_path("/workspace/a/imgs/1.jpg", [("/workspace/a", "/home/b")]) == "/home/b/imgs/1.jpg"
    assert G.map_path("/other/1.jpg", [("/workspace/a", "/home/b")]) == "/other/1.jpg"
