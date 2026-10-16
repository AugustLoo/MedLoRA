"""35B 第①步「知识预训练」语料: 医学教材 + 临床指南 + 中文医学百科 / 语料, 切成约 1,000 token 一段的纯文字。

来源 (服务器上已下载到 /home/ubuntu/chunqian/data/cpt_corpora/, 见 docs/PLAN_35B_CPT.md 第四节):
    openmedtext/Med-Textbooks/<许可>/<书名>.txt   ywchoi/OpenMedText 的 29 本英文开源教材 (默认全用)
    guidelines/open_guidelines.jsonl               epfl-llm/guidelines 3.7 万篇临床指南 (按 token 预算抽篇)
    mmedc/Chinese.zip                              Henrychur/MMedC 中文部分: SmallScaleCorpora_ZH/baike_txt (医学百科)
                                                   (默认 800 万 token, 先用粗规则跳过医院 / 医美推广条目, 约 10%);
                                                   官方包里没有中文教材文件夹。FilteredCorpus_Chinese (医学网页语料)
                                                   抽样一半以上是医院 SEO 推广页, 默认不用 (--zh-filtered-tokens -1)
输出 (--out, 默认 data/processed/):
    cpt35_stage1.json        训练段 [{"text", "source", "lang"}]   (train_lora.py --mode pt 读 "text")
    cpt35_stage1_val.json    每个来源留出约 1% 的段落做验证集, 用来比较学习率 (与训练段不重叠, 按整篇划分)
    cpt35_stage1_stats.json  每个来源的篇数 / 段数 / token 数 / 许可证
切段: 先按空行分段落, 段落太长再按句号分句, 还太长按字符硬切; 累加到 --chunk-tokens 为止。token 数用 35B 模型自己的
tokenizer 计 (--tokenizer); 抽样与划分全部固定种子, 重跑结果相同。

用法 (服务器, 不占卡):
    python data/convert_cpt35_stage1.py --tokenizer /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO = Path(__file__).resolve().parents[1]
CORPORA = Path("/home/ubuntu/chunqian/data/cpt_corpora")
LICENSES = {
    "openmedtext": "按书: CC BY / CC BY-SA / CC BY-NC / CC BY-NC-SA (见目录名)",
    "guidelines": "Common Crawl Foundation Terms of Use",
    "mmedc_baike": "CC BY-NC-SA 4.0",
    "mmedc_filtered": "CC BY-NC-SA 4.0",
}
_SENT = re.compile(r"(?<=[。！？；!?;])|(?<=[.!?])\s+")
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.I)

Count = Callable[[str], int]


# ---------------------------------------------------------------- 清洗与切段 (纯函数, 有单元测试)
def clean_text(text: str) -> str:
    """统一换行; 去掉单独一行的页码; 行内空白压成一个; 三个以上空行压成一个空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ").replace("　", " ")
    lines = []
    for line in text.split("\n"):
        if _PAGE_NO.match(line):
            continue
        lines.append(re.sub(r"[ \t\f\v]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def paragraphs(text: str) -> list[str]:
    """按空行分段落; 整篇没有空行时 (常见于 PDF 转出的书) 退回按行分。"""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) <= 1 and text.count("\n") > 10:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return [re.sub(r"\s*\n\s*", " ", p) if not re.search(r"[一-鿿]", p) else p.replace("\n", "") for p in parts]


def _split_long(piece: str, count: Count, limit: int) -> list[str]:
    """超过 limit 的段落: 先按句子切, 单句还超长就按字符等分硬切。"""
    if count(piece) <= limit:
        return [piece]
    out: list[str] = []
    for sent in [s for s in _SENT.split(piece) if s and s.strip()]:
        n = count(sent)
        if n <= limit:
            out.append(sent.strip())
            continue
        k = -(-n // limit)  # 向上取整
        step = -(-len(sent) // k)
        out += [sent[i:i + step].strip() for i in range(0, len(sent), step) if sent[i:i + step].strip()]
    return out


def _joiner(text: str) -> str:
    """同一段落切开的几片接回去时用的连接符: 中文直接接, 英文用空格。"""
    return "" if re.search(r"[一-鿿]", text) else " "


def pack(paras: Iterable[str], count: Count, limit: int, min_tokens: int = 32) -> list[str]:
    """把段落依次累加成不超过 limit token 的段。不同段落之间空一行; 同一段落因太长切开的几片, 落在同一段里时直接接回去
    (中文不加空格, 英文加一个空格)。计数包含分隔符; 最后再整段核对一次, 万一仍超长就从中间拆开。
    最后一段不足 min_tokens 时并进前一段 (仍不超过 limit) 或丢掉; 整篇都不足 min_tokens 时返回空。"""
    sep_n = count("\n\n") or 1
    chunks: list[str] = []
    cur, cur_n = "", 0          # 当前段的文字和估计 token 数
    for para in paras:
        pieces = _split_long(para, count, limit)
        join = _joiner(para)
        for k, piece in enumerate(pieces):
            n = count(piece)
            glue = "" if not cur else (join if k > 0 else "\n\n")
            add = n + (sep_n if glue == "\n\n" else (1 if glue else 0))
            if cur and cur_n + add > limit:
                chunks.append(cur)
                cur, cur_n, glue, add = "", 0, "", n
            cur, cur_n = cur + glue + piece, cur_n + add
    if cur:
        if cur_n >= min_tokens:
            chunks.append(cur)
        elif chunks and count(chunks[-1]) + sep_n + cur_n <= limit:
            chunks[-1] = chunks[-1] + "\n\n" + cur
    out: list[str] = []
    for c in chunks:            # 整段核对: 估计和真实计数在拼接处可能差几个 token
        stack = [c]
        while stack:
            s = stack.pop(0)
            if count(s) <= limit or len(s) < 2:
                out.append(s)
            else:
                mid = len(s) // 2
                cut = s.rfind("\n\n", 0, mid)
                cut = cut if cut > 0 else mid
                stack[:0] = [s[:cut].strip(), s[cut:].strip()]
    return [s for s in out if s]


_AD = re.compile(r"整形|美容|医美|诊所|价格|多少钱|收费|优惠|咨询|电话|热线|地址|预约|微信|QQ|热玛吉|玻尿酸|双眼皮|隆鼻|祛斑|脱毛"
                 r"|无痛人流|人流|不孕不育|白癜风医院|牛皮癣医院|癫痫医院|男科医院|妇科医院|专科医院|妇科网|在线解答|专家在线"
                 r"|免费|名额|援助|来院|挂号|网站|本院|我院|医学研究院|公益|活动期间")


def looks_like_ad(text: str, per_kchar: float = 2.0) -> bool:
    """医院 / 医美推广条目: 推广用语每千字超过 per_kchar 次就跳过 (粗规则; 2026-10-16 抽 1,500 篇百科看过, 阈值 2 拦下约 10%,
    多是公司简介、医生个人介绍和医院推广, 正常的疾病条目分数接近 0)。"""
    return len(_AD.findall(text)) * 1000 / max(1, len(text)) > per_kchar


def take_budget(docs: Iterable[tuple[str, str]], count: Count, limit: int, budget: int) -> Iterator[tuple[str, list[str], int]]:
    """按顺序取文档 (name, text), 切段, 直到累计 token 达到 budget (budget == 0 表示全取, < 0 表示不取)。产出 (name, 段落列表, token 数)。"""
    if budget < 0:
        return
    used = 0
    for name, text in docs:
        chunks = pack(paragraphs(clean_text(text)), count, limit)
        if not chunks:
            continue
        ns = [count(c) for c in chunks]
        if budget > 0 and used + sum(ns) > budget:  # 最后一篇只取到预算为止
            keep, acc = [], 0
            for c, n in zip(chunks, ns):
                if used + acc + n > budget:
                    break
                keep.append(c)
                acc += n
            if keep:
                yield name, keep, acc
            return
        used += sum(ns)
        yield name, chunks, sum(ns)
        if budget > 0 and used >= budget:
            return


def split_val(docs: list[tuple[str, list[str], int]], frac: float, rng: random.Random, max_chunks: int = 100):
    """划出验证集, 约为总段数的 frac, 最多 max_chunks 段。短篇整篇划入 (同一篇不会一半训练一半验证);
    碰到篇幅超过还缺段数两倍的长篇 (比如一整本教材), 只把它最后若干段 (末尾章节) 划入验证, 前面的仍用于训练,
    免得来源只有几十本书时一下子拿走一整本。"""
    total = sum(len(c) for _, c, _ in docs)
    target = min(max_chunks, max(1, round(total * frac))) if total >= 20 else 0
    order = list(range(len(docs)))
    rng.shuffle(order)
    train, val, got = list(docs), [], 0
    for i in order:
        need = target - got
        if need <= 0 or len(val) >= len(docs) - 1:
            break
        name, chunks, n = docs[i]
        if len(chunks) <= max(need * 2, 3):
            val.append(docs[i])
            train[i] = None
            got += len(chunks)
        else:
            head, tail = chunks[:-need], chunks[-need:]
            val.append((f"{name}#末尾{need}段", tail, round(n * need / len(chunks))))
            train[i] = (name, head, n - val[-1][2])
            got += need
    return [d for d in train if d is not None], val


_ATTRIBUTION = re.compile(r"Provided by:|Located at:|License:|Authored by:|creativecommons\.org|Public Domain:|"
                          r"\bCC BY(-[A-Z]+)*(\s\d\.\d)?\s*:", re.I)


def strip_attribution(text: str) -> str:
    """去掉开源教材里的图片 / 内容出处与许可说明行 (Provided by … Located at … License …)。
    2026-10-16 统计: 这类行占教材文字 4.7%, 主要在 AandP for STEM Educators 一本里; 其他来源几乎没有。"""
    return "\n".join(l for l in text.split("\n") if not _ATTRIBUTION.search(l))


# ---------------------------------------------------------------- 读三个来源
def iter_openmedtext(root: Path, rng: random.Random):
    books = sorted(root.glob("Med-Textbooks/*/*.txt"))
    rng.shuffle(books)
    for b in books:
        yield f"{b.parent.name}/{b.stem}", strip_attribution(b.read_text(encoding="utf-8", errors="ignore"))


def iter_guidelines(path: Path, rng: random.Random):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rng.shuffle(rows)
    for r in rows:
        text = (r.get("clean_text") or "").strip()
        if len(text) < 200:
            continue
        title = (r.get("title") or "").strip()
        yield f"{r.get('source', '?')}/{r.get('id', '?')}", (title + "\n\n" + text) if title else text


SKIPPED_ADS: dict[str, int] = defaultdict(int)
ZIP_PREFIX = {"mmedc_baike": "Chinese/SmallScaleCorpora_ZH/baike_txt/", "mmedc_filtered": "Chinese/FilteredCorpus_Chinese/"}


def iter_zip(path: Path, prefix: str, rng: random.Random, drop_ads: bool = False):
    z = zipfile.ZipFile(path)
    names = sorted(i.filename for i in z.infolist() if not i.is_dir() and i.filename.startswith(prefix) and i.filename.endswith(".txt"))
    rng.shuffle(names)
    for n in names:
        text = z.read(n).decode("utf-8", errors="ignore")
        if drop_ads and looks_like_ad(text):
            SKIPPED_ADS[prefix] += 1
            continue
        yield n.rsplit("/", 1)[-1], text


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="35B 模型目录 (用它的 tokenizer 计 token)")
    ap.add_argument("--corpora", default=str(CORPORA))
    ap.add_argument("--out", default=str(REPO / "data" / "processed"))
    ap.add_argument("--chunk-tokens", type=int, default=1000, help="每段上限; 训练截断 1,024, 留出结束符与拼接余量")
    ap.add_argument("--en-books-tokens", type=int, default=0, help="OpenMedText 教材预算, 0 = 全部")
    ap.add_argument("--guidelines-tokens", type=int, default=4_000_000)
    ap.add_argument("--zh-baike-tokens", type=int, default=8_000_000, help="中文医学百科")
    ap.add_argument("--zh-filtered-tokens", type=int, default=-1, help="中文网页语料; 默认 -1 不用 (多为医院推广页)")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    root = Path(args.corpora)
    sources = [
        ("openmedtext", "en", lambda r: iter_openmedtext(root / "openmedtext", r), args.en_books_tokens),
        ("guidelines", "en", lambda r: iter_guidelines(root / "guidelines" / "open_guidelines.jsonl", r), args.guidelines_tokens),
        ("mmedc_baike", "zh", lambda r: iter_zip(root / "mmedc" / "Chinese.zip", ZIP_PREFIX["mmedc_baike"], r, drop_ads=True),
         args.zh_baike_tokens),
        ("mmedc_filtered", "zh", lambda r: iter_zip(root / "mmedc" / "Chinese.zip", ZIP_PREFIX["mmedc_filtered"], r, drop_ads=True),
         args.zh_filtered_tokens),
    ]
    train, val, stats = [], [], {}
    for i, (name, lang, it, budget) in enumerate(sources):
        rng = random.Random(args.seed + i)
        docs = list(take_budget(it(rng), count, args.chunk_tokens, budget))
        tr, va = split_val(docs, args.val_frac, random.Random(args.seed + 100 + i))
        for split, part in (("train", tr), ("val", va)):
            for doc, chunks, _ in part:
                rows = [{"text": c, "source": f"{name}/{doc}", "lang": lang} for c in chunks]
                (train if split == "train" else val).extend(rows)
        stats[name] = {"lang": lang, "license": LICENSES[name], "budget_tokens": budget,
                       "docs": len(docs), "chunks": sum(len(c) for _, c, _ in docs), "tokens": sum(n for *_, n in docs),
                       "train_chunks": sum(len(c) for _, c, _ in tr), "val_chunks": sum(len(c) for _, c, _ in va),
                       "skipped_as_ads": SKIPPED_ADS.get(ZIP_PREFIX.get(name, ""), 0)}
        print(name, json.dumps(stats[name], ensure_ascii=False), flush=True)
    random.Random(args.seed).shuffle(train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpt35_stage1.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (out / "cpt35_stage1_val.json").write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    stats["_total"] = {"train_chunks": len(train), "val_chunks": len(val),
                       "tokens": sum(s["tokens"] for s in stats.values() if isinstance(s, dict) and "tokens" in s),
                       "chunk_tokens": args.chunk_tokens, "seed": args.seed}
    (out / "cpt35_stage1_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("total", json.dumps(stats["_total"], ensure_ascii=False))


if __name__ == "__main__":
    main()
