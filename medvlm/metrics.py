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
