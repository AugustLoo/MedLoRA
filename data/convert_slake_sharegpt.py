"""把 SLAKE 转成 LLaMA-Factory 的 sharegpt 多模态格式, 并生成 dataset_info.json。

输出: data/processed/slake_train.json, slake_validation.json, dataset_info.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.prompts import slake_prompt  # noqa: E402
from medvlm.slake import load_split  # noqa: E402

OUT = REPO / "data" / "processed"


def convert(split: str) -> Path:
    rows = load_split(split, lang="en")
    samples = []
    for r in rows:
        if not r.get("image_path"):
            raise SystemExit("HF 格式没有 image_path, 请用官方 json+imgs 布局做训练。")
        samples.append({
            "messages": [
                {"role": "user", "content": "<image>" + slake_prompt(r["question"], r["answer_type"])},
                {"role": "assistant", "content": r["answer"]},
            ],
            "images": [r["image_path"]],
        })
    fn = OUT / f"slake_{split}.json"
    fn.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(split, len(samples), "->", fn)
    return fn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info_fn = OUT / "dataset_info.json"
    info = json.loads(info_fn.read_text(encoding="utf-8")) if info_fn.exists() else {}
    for split in ["train", "validation"]:
        fn = convert(split)
        info[f"slake_{split}"] = {
            "file_name": fn.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    info_fn.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("dataset_info ->", info_fn)


if __name__ == "__main__":
    main()
