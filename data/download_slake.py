"""下载 SLAKE 到 data/raw/SLAKE 并解压所有 zip。

数据源: HF 镜像 BoKelvin/SLAKE (原始发布在 Google Drive, 见官方 repo)。
如果 HF 镜像布局变了, 手动把官方的 train.json / validate.json / test.json / imgs/ 放进
data/raw/SLAKE 即可, medvlm/slake.py 会自动识别。
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "SLAKE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BoKelvin/SLAKE")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(RAW))
    for z in RAW.rglob("*.zip"):
        print("解压", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)
    print("文件列表:")
    for p in sorted(RAW.rglob("*"))[:40]:
        print("  ", p.relative_to(RAW))
    print("如果没看到 train.json / imgs, 请按 docstring 手动放置。")


if __name__ == "__main__":
    main()
