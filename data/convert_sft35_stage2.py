"""35B 第②步「问答微调」的中文医学问答 / 对话部分: HuatuoGPT-sft-data-v1 + DISC-Med-SFT, 清洗后各抽 6,000 条。
六个多模态数据集 (与 S2-6datasets 相同) 不在这里处理, 训练时与本脚本的输出一起读 (train/interns2/run_sft35.sh)。

来源 (服务器 /home/ubuntu/chunqian/data/sft35_raw/, scripts/dl_sft35.sh 下载):
    huatuo_sft/HuatuoGPT_sft_data_v1.jsonl   22.6 万条, {"data": ["问：…", "答：…", …]}, Apache-2.0
    disc_med_sft/DISC-Med-SFT_released.jsonl 46.5 万条, {"conversation": [{role, content}…], "source"}, Apache-2.0;
                                             source = meddial / CMeKG / cMedQA。cMedQA 部分不用 (上游 cMedQA2 限非商用研究)
清洗 (docs/PLAN_35B_CPT.md 第七节):
    问答必须一问一答交替; 最多保留前 3 轮; 问题 ≥ 4 字、每个回答 ≥ 10 字;
    含手机号 / 身份证号 / 邮箱 / QQ / 网址的整条丢掉; 医院推广 (「我院」「热线」「在线咨询」等两种以上) 的丢掉;
    DISC 里用户说「这是图片」「上传了照片」之类的整条丢掉 (图片不在数据里, 回答会对不上);
    第一句问题归一化后跨两个数据集去重; 和所有评测题比对 (sft35_common.EvalIndex), 重复或改写的剔除;
    套模板后超过 --max-tokens 的从后往前删轮, 删到只剩第一轮还超长就丢掉。
输出 (--out, 默认 data/processed/):
    sft35_zh_dialog.json         [{"messages": [{"role", "content"}…], "source", "id"}]  (train_lora.py 读 messages, 多轮每轮回答都算损失)
    sft35_zh_dialog_stats.json   每一步丢了多少条

用法 (服务器, 不占卡):
    python data/convert_sft35_stage2.py --tokenizer /home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sft35_common import RAW, REPO, build_eval_index, fits, has_pii, load_any, looks_like_promo_qa, norm, sample, strip_html, token_counter

_IMG_REF = re.compile(r"图片|照片|上传|附件|看图|这张图|截图|传图")


def huatuo_messages(row: dict) -> list[dict] | None:
    turns = row.get("data") or []
    if len(turns) < 2 or len(turns) % 2:
        return None
    msgs = []
    for k, t in enumerate(turns):
        role, prefix = ("user", "问：") if k % 2 == 0 else ("assistant", "答：")
        t = str(t).strip()
        if t.startswith(prefix):
            t = t[len(prefix):]
        t = strip_html(t)          # 去前缀之后再清: 原文常是「问：BR>…」
        msgs.append({"role": role, "content": t})
    return msgs


def disc_messages(row: dict) -> list[dict] | None:
    conv = row.get("conversation") or []
    if len(conv) < 2 or len(conv) % 2:
        return None
    msgs = []
    for k, m in enumerate(conv):
        want = "user" if k % 2 == 0 else "assistant"
        if m.get("role") != want:
            return None
        msgs.append({"role": want, "content": strip_html(m.get("content", ""))})
    return msgs


def clean(msgs: list[dict] | None, *, max_rounds: int, image_refs: bool) -> tuple[list[dict] | None, str]:
    """返回 (清洗后的对话 或 None, 丢弃原因)。"""
    if not msgs:
        return None, "格式不对 (不是一问一答交替)"
    msgs = msgs[: 2 * max_rounds]
    if any(len(m["content"]) < (4 if m["role"] == "user" else 10) for m in msgs):
        return None, "问题太短或回答太短"
    text = "\n".join(m["content"] for m in msgs)
    if has_pii(text):
        return None, "含联系方式 / 个人信息 / 网址"
    if looks_like_promo_qa(text):
        return None, "医院推广 (两种以上自我推销用语)"
    if image_refs and any(_IMG_REF.search(m["content"]) for m in msgs if m["role"] == "user"):
        return None, "提到图片但数据里没有图"
    return msgs, ""


def trim_to_fit(msgs: list[dict], count, limit: int) -> list[dict] | None:
    """从后往前一轮一轮删, 直到套模板后不超过 limit; 第一轮就超长返回 None。"""
    while msgs and not fits(msgs, count, limit):
        msgs = msgs[:-2]
    return msgs or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="35B 模型目录 (计 token 用); 不给就按 字符数/2 粗估")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(REPO / "data" / "processed"))
    ap.add_argument("--huatuo", type=int, default=6000)
    ap.add_argument("--disc", type=int, default=6000)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1000, help="训练截断 1,024, 留余量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample", type=float, default=3.0, help="先随机取 目标×此倍数 条再清洗, 免得把几十万条全过一遍")
    args = ap.parse_args()
    raw = Path(args.raw)
    count = token_counter(args.tokenizer)
    print("建评测题索引")
    idx = build_eval_index()

    sources = [
        ("huatuo_sft", args.huatuo, raw / "huatuo_sft" / "HuatuoGPT_sft_data_v1.jsonl", huatuo_messages, False),
        ("disc_med_sft", args.disc, raw / "disc_med_sft" / "DISC-Med-SFT_released.jsonl", disc_messages, True),
    ]
    seen: set[str] = set()
    out, stats = [], {}
    for k, (name, target, path, to_msgs, image_refs) in enumerate(sources):
        rows = load_any(path)
        total = len(rows)
        if name == "disc_med_sft":
            rows = [r for r in rows if r.get("source") != "cMedQA"]
        drop = Counter({"cMedQA 来源 (限非商用)": total - len(rows)}) if name == "disc_med_sft" else Counter()
        pool = sample(rows, int(target * args.oversample), args.seed + k)
        kept = []
        for i, r in enumerate(pool):
            msgs, why = clean(to_msgs(r), max_rounds=args.max_rounds, image_refs=image_refs)
            if msgs is None:
                drop[why] += 1
                continue
            key = norm(msgs[0]["content"])
            if key in seen:
                drop["第一句问题与已收的重复 (跨数据集去重)"] += 1
                continue
            if any(idx.contains(m["content"]) for m in msgs if m["role"] == "user"):
                drop["与评测题重复或改写"] += 1
                continue
            msgs = trim_to_fit(msgs, count, args.max_tokens)
            if msgs is None:
                drop["第一轮就超长"] += 1
                continue
            seen.add(key)
            src = f"{name}/{r.get('source')}" if r.get("source") else name
            kept.append({"messages": msgs, "source": src, "id": str(r.get("_id", f"{name}-{i}"))})
            if len(kept) >= target:
                break
        stats[name] = {"total": total, "pool": len(pool), "kept": len(kept), "target": target,
                       "rounds": dict(Counter(len(x["messages"]) // 2 for x in kept)),
                       "by_source": dict(Counter(x["source"] for x in kept)), "dropped": dict(drop)}
        print(name, json.dumps(stats[name], ensure_ascii=False), flush=True)
        if len(kept) < target:
            print(f"!! {name} 只收到 {len(kept)} / {target} 条, 可调大 --oversample", flush=True)
        out += kept
    stats["_eval_index"] = idx.sources
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "sft35_zh_dialog.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (o / "sft35_zh_dialog_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("total", len(out))


if __name__ == "__main__":
    main()
