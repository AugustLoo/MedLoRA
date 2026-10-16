"""35B 第②③步数据脚本共用的东西: 文本归一、个人信息 / 广告清洗、跨数据集去重、测试集隔离 (docs/PLAN_35B_CPT.md 第七节)。

测试集隔离的做法: 把所有评测题 (SLAKE 中英文测试题、MedQA 测试集、PubMedQA 考卷半边、CMExam 验证 / 测试集、MedMCQA
验证 / 测试集) 归一化 (去掉空白和标点、转小写) 后切成 20 个字符的滑动窗口建索引。一条训练样本的题干若有一半以上的窗口
出现在索引里, 就当作「同一道题或改写」剔除; 题干太短 (不足 20 个字符) 时只查完全相同。这样能抓住原题、加了几个字的改写,
又不会因为病例题常见的开头 (「A 45-year-old man presents with」) 误伤。
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_cpt35_stage1 import looks_like_ad  # noqa: E402  (与第①步同一套推广用语过滤)

RAW = Path("/home/ubuntu/chunqian/data/sft35_raw")
WIN = 20

_NONWORD = re.compile(r"[\W_]+", re.UNICODE)       # 去掉空白和标点, 保留字母、数字、汉字
_PII = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"           # 手机号
                  r"|(?<!\d)\d{17}[\dXx](?!\d)"            # 身份证号
                  r"|[\w.+-]+@[\w-]+\.[\w.]+"              # 邮箱
                  r"|[Qq]{2}[:：号]?\s*\d{5,}"              # QQ 号
                  r"|https?://|www\.", re.I)               # 网址


# 问答 / 对话里的医院推广: 第①步按「推广用语密度」的规则会误伤正常问答 (2026-10-16 抽 3,000 条看过: 被拦的四成多是
# 「建议咨询医生」「预约挂号」这类正常用语, 或者病人本来就在问人流、不孕不育、双眼皮)。问答只认医院自我推销的说法,
# 且要出现两种以上不同的词。
_PROMO_QA = re.compile(r"我院|本院|来院|热线|优惠|名额|专家在线|在线解答|在线咨询|网站|微信|QQ|电话")


def looks_like_promo_qa(text: str, min_kinds: int = 2) -> bool:
    return len(set(_PROMO_QA.findall(text))) >= min_kinds


_HTML = re.compile(r"<\s*/?\s*(br|p|div|span|b|strong|em|i|font)\b[^>]*>|^\s*/?\s*br\s*/?\s*>|&nbsp;", re.I)


def strip_html(text: str) -> str:
    """去掉网页残留: <br> / <p> 等标签、被截断的「BR>」开头、&nbsp;。"""
    return _HTML.sub(" ", str(text)).strip()


def norm(s: str) -> str:
    return _NONWORD.sub("", str(s)).lower()


def has_pii(text: str) -> bool:
    return bool(_PII.search(text))


def load_any(path: Path) -> list[dict]:
    """json 数组或 jsonl 都认。"""
    t = Path(path).read_text(encoding="utf-8")
    return json.loads(t) if t.lstrip().startswith("[") else [json.loads(l) for l in t.splitlines() if l.strip()]


class EvalIndex:
    """评测题索引: contains(text) 判断一条训练题是不是某道评测题 (或其改写)。"""

    def __init__(self):
        self.windows: set[int] = set()
        self.exact: set[str] = set()
        self.sources: dict[str, int] = {}

    def add(self, name: str, texts: Iterable[str]):
        n = 0
        for t in texts:
            s = norm(t)
            if not s:
                continue
            n += 1
            self.exact.add(s)
            for i in range(0, max(0, len(s) - WIN + 1)):
                self.windows.add(hash(s[i:i + WIN]))
        self.sources[name] = n

    def contains(self, text: str, frac: float = 0.5) -> bool:
        s = norm(text)
        if not s:
            return False
        if s in self.exact:
            return True
        if len(s) < WIN:
            return False
        wins = {hash(s[i:i + WIN]) for i in range(len(s) - WIN + 1)}
        return sum(w in self.windows for w in wins) >= frac * len(wins)


def build_eval_index(log: Callable[[str], None] = print) -> EvalIndex:
    """在服务器上建索引。哪一份评测数据读不到会报出来 (不静默跳过), 由调用方决定是否继续。"""
    idx = EvalIndex()
    slake = REPO / "data" / "raw" / "SLAKE" / "test.json"
    idx.add("SLAKE test (中英文)", (r["question"] for r in json.loads(slake.read_text(encoding="utf-8"))))
    from datasets import load_dataset

    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", data_files={"test": "phrases_no_exclude_test.jsonl"},
                         split="test", verification_mode="no_checks")
    idx.add("MedQA test", (r["question"] for r in medqa))
    held = set(json.loads((REPO / "data" / "pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
    pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    idx.add("PubMedQA 考卷半边", (r["question"] for r in pq if int(r["pubid"]) in held))
    for split in ("valid", "test"):
        idx.add(f"CMExam {split}", (r["Question"] for r in load_any(RAW / "cmexam" / f"{split}.json")))
    import pandas as pd

    for split in ("validation", "test"):
        m = pd.read_parquet(RAW / "medmcqa" / "data" / f"{split}-00000-of-00001.parquet")
        idx.add(f"MedMCQA {split}", m["question"].astype(str))
    for k, v in idx.sources.items():
        log(f"  评测索引 {k}: {v} 题")
    log(f"  评测索引共 {len(idx.windows):,} 个窗口")
    return idx


def sample(rows: list, n: int, seed: int) -> list:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n > 0 else rows


def token_counter(tokenizer_path: str | None) -> Callable[[str], int]:
    """计 token: 给了模型目录就用模型自己的 tokenizer; 否则 (单元测试) 粗估为 字符数 / 2。"""
    if not tokenizer_path:
        return lambda s: max(1, len(s) // 2)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def fits(messages: list[dict], count: Callable[[str], int], limit: int, per_turn: int = 12) -> bool:
    """对话套上模板后大约多少 token: 各条内容 + 每条约 per_turn 个模板 token。"""
    return sum(count(m["content"]) + per_turn for m in messages) <= limit
