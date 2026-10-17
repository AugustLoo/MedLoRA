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


class CharTok:
    """一个字符一个 token 的假 tokenizer, 用来核对多轮对话的损失掩码。"""
    unk_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

    def convert_tokens_to_ids(self, t):
        return 7 if t == "<|im_end|>" else 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class ChatProc:
    """假的对话模板: <|im_start|>{role}\n{text}<|im_end|>\n, 生成提示加 <|im_start|>assistant\n。"""
    tokenizer = CharTok()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content'][0]['text']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def multi_turn_data(tmp_path, msgs, cutoff=1024):
    return T.SFTData([write(tmp_path, [{"messages": msgs}])], ChatProc(), cutoff, 262144, [])


def test_multi_turn_trains_every_assistant_turn_only(tmp_path):
    msgs = [{"role": "user", "content": "头痛怎么办"}, {"role": "assistant", "content": "多休息"},
            {"role": "user", "content": "还要吃药吗"}, {"role": "assistant", "content": "先观察两天"}]
    d = multi_turn_data(tmp_path, msgs)
    ex = d[0]
    tok = CharTok()
    learned = tok.decode(ex["input_ids"][ex["labels"] != -100])
    assert learned == "多休息<|im_end|>先观察两天<|im_end|>"
    assert "头痛怎么办" in tok.decode(ex["input_ids"]) and "头痛" not in learned


def test_multi_turn_keeps_only_turns_that_fit(tmp_path):
    msgs = [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答" * 10},
            {"role": "user", "content": "再问"}, {"role": "assistant", "content": "答" * 200}]
    ex = multi_turn_data(tmp_path, msgs, cutoff=80)[0]
    assert len(ex["input_ids"]) <= 80
    assert CharTok().decode(ex["input_ids"][ex["labels"] != -100]) == "答" * 10 + "<|im_end|>"


def test_single_turn_path_unchanged(tmp_path):
    msgs = [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]
    d = multi_turn_data(tmp_path, msgs)
    prompt, target, img = d.render(d.rows[0])
    assert prompt.endswith("<|im_start|>assistant\n") and target == "回答<|im_end|>" and img is None


def test_reset_rope_cache_clears_stale_deltas():
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope_deltas = torch.zeros(4, 1)   # 上一个带图批次 (4 条) 留下的
    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.lin = torch.nn.Linear(2, 2)
    m = Outer()
    holders = T.rope_cache_holders(m)
    assert holders == [m.inner]
    T.reset_rope_cache(holders)
    assert m.inner.rope_deltas is None
