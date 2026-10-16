"""评估指标。全部基于归一化后的字符串, 与 LLaVA-Med / SLAKE 常用口径一致。"""
import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def yes_no(s: str) -> str:
    """把自由回答折成 yes / no / other, 用于 CLOSED 题。"""
    n = normalize(s)
    first = n.split()[0] if n else ""
    if first in {"yes", "y", "true"}:
        return "yes"
    if first in {"no", "n", "false"}:
        return "no"
    if n.startswith("yes"):
        return "yes"
    if n.startswith("no"):
        return "no"
    return "other"


def closed_score(pred: str, gold: str) -> float:
    """CLOSED 题评分。

    SLAKE 的 CLOSED 不等于 yes/no: 416 道里有 61 道的标准答案是 Lung / Liver / T2 /
    Coronal Plane 这类封闭词表。对这些题只能按字面比, 若沿用 yes_no() 折叠,
    任何非 yes/no 的回答都会折成 "other" 并与标准答案相等, 白送分 (2026-09-19 修正)。
    """
    g = yes_no(gold)
    if g == "other":
        return exact_match(pred, gold)
    return float(yes_no(pred) == g)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_recall(pred: str, gold: str) -> float:
    """gold 的 token 有多少出现在 pred 里 (LLaVA-Med 的 open-ended recall)。"""
    g = normalize(gold).split()
    p = Counter(normalize(pred).split())
    if not g:
        return 0.0
    hit = sum(min(c, p[t]) for t, c in Counter(g).items())
    return hit / len(g)


def token_f1(pred: str, gold: str) -> float:
    g, p = normalize(gold).split(), normalize(pred).split()
    if not g or not p:
        return float(g == p)
    common = sum((Counter(g) & Counter(p)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def vqa_accuracy(pred: str, golds: list[str]) -> float:
    """VQAv2 / TextVQA 口径: min(#匹配的标注/3, 1)。"""
    n = normalize(pred)
    hits = sum(normalize(g) == n for g in golds)
    return min(hits / 3.0, 1.0)


# ---- 中文 (SLAKE 中文题) ----
# 中文封闭题的标准答案有多种说法: 是的/不是、包含/不包含、有/没有、可以/不可以、正常/不正常 ……
# 先按「否定 / 肯定」折成 no / yes; 两边都折不了 (如 肺、T1、横断面) 就按字面比 (与英文 closed_score 同一思路)。
_ZH_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》【】,.;:?!\"'()\[\]<>]+")
_ZH_NO = ("不是", "不包含", "没有", "不可以", "不正常", "不存在", "不能", "否", "无", "异常")
_ZH_YES = ("是的", "是", "包含", "有", "可以", "正常", "健康", "存在", "能", "对")


def normalize_zh(s: str) -> str:
    return _ZH_PUNCT.sub("", str(s)).lower()


def yes_no_zh(s: str) -> str:
    n = normalize_zh(s)
    if not n:
        return "other"
    if n.startswith(("yes", "no")):
        return "yes" if n.startswith("yes") else "no"
    if n.startswith(_ZH_NO):
        return "no"
    if n.startswith(_ZH_YES):
        return "yes"
    return "other"


def closed_score_zh(pred: str, gold: str) -> float:
    g = yes_no_zh(gold)
    if g == "other":
        return float(normalize_zh(pred) == normalize_zh(gold))
    return float(yes_no_zh(pred) == g)


def exact_match_zh(pred: str, gold: str) -> float:
    return float(normalize_zh(pred) == normalize_zh(gold))


def char_recall_zh(pred: str, gold: str) -> float:
    """中文没有空格, 按字算: 标准答案的字有多少出现在回答里 (多重集合)。"""
    g, p = Counter(normalize_zh(gold)), Counter(normalize_zh(pred))
    total = sum(g.values())
    return sum(min(c, p[ch]) for ch, c in g.items()) / total if total else 0.0


def char_f1_zh(pred: str, gold: str) -> float:
    g, p = Counter(normalize_zh(gold)), Counter(normalize_zh(pred))
    common = sum((g & p).values())
    if not g or not p or common == 0:
        return float(not g and not p)
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)
