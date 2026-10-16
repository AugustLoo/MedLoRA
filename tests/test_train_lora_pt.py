"""train_lora.py 新加的预训练模式与 --init-adapter 检查 (要 torch, 在服务器的 s2train 环境里跑)。"""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "interns2"))
import train_lora as T  # noqa: E402


class FakeTok:
    unk_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

    def convert_tokens_to_ids(self, t):
        return 9 if t == "<|endoftext|>" else 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([[10 + (len(w) % 50) for w in text.split()]])
        return {"input_ids": ids}


class FakeProc:
    tokenizer = FakeTok()


def write(tmp_path, rows):
    f = tmp_path / "d.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    return str(f)


def test_pretrain_rows_get_full_loss_and_endoftext(tmp_path):
    d = T.PretrainData([write(tmp_path, [{"text": "a bb ccc"}])], FakeProc(), cutoff=1024)
    ex = d[0]
    assert ex["input_ids"].tolist() == [11, 12, 13, 9]          # 三个词 + <|endoftext|>
    assert torch.equal(ex["labels"], ex["input_ids"])           # 每个 token 都算损失
    assert d.render(d.rows[0]) == ("", "a bb ccc", None)


def test_pretrain_truncates_to_cutoff_keeping_eos(tmp_path):
    d = T.PretrainData([write(tmp_path, [{"text": " ".join(["w"] * 50)}])], FakeProc(), cutoff=8)
    ids = d[0]["input_ids"]
    assert len(ids) == 8 and ids[-1].item() == 9


def test_pretrain_rejects_chat_format(tmp_path):
    with pytest.raises(SystemExit):
        T.PretrainData([write(tmp_path, [{"messages": []}])], FakeProc(), cutoff=16)


def test_collate_pads_labels_with_ignore_index(tmp_path):
    d = T.PretrainData([write(tmp_path, [{"text": "a"}, {"text": "a b c"}])], FakeProc(), cutoff=64)
    batch = T.make_collate(1)([d[0], d[1]])
    assert batch["input_ids"].shape == batch["labels"].shape
    short = 0 if (batch["attention_mask"][0] == 0).any() else 1
    assert (batch["labels"][short][batch["attention_mask"][short] == 0] == -100).all()


def test_adapter_mismatch():
    target = T.LORA_TARGETS["attn_shared"]
    ok = {"r": 16, "lora_alpha": 32, "target_modules": target}
    assert T.adapter_mismatch(ok, 16, 32, target) == []
    assert len(T.adapter_mismatch({**ok, "r": 8}, 16, 32, target)) == 1
    assert len(T.adapter_mismatch(ok, 16, 32, T.LORA_TARGETS["attn"])) == 1


def test_existing_6ds_adapter_matches_default_targets():
    cfg = Path("outputs/interns2_6ds/adapter_config.json")
    if not cfg.exists():
        pytest.skip("只在服务器上有")
    assert T.adapter_mismatch(json.loads(cfg.read_text()), 16, 32, T.LORA_TARGETS["attn_shared"]) == []
