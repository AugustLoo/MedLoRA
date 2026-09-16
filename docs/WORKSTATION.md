# 在朋友的 32 GB 显卡上跑实验

Kaggle T4 (16 GB) 只能 4bit + fp16, 每周 30 h 配额, 一轮实验 6–8 h。
32 GB 单卡 (RTX 5090 / RTX 5000 Ada / V100 32G) 可以不量化直接 bf16 LoRA, 一轮约 1–1.5 h, 没有配额。

## 需要的环境
- Linux, 或 Windows 上的 **WSL2 (Ubuntu)**。Windows 原生 Python 也能跑, 但 bitsandbytes / flash-attn 常出问题, bf16 路线不依赖它们, 所以原生也可以试。
- NVIDIA 驱动 ≥ 570 (RTX 50 系列需要 CUDA 12.8)。
- 磁盘约 30 GB: 基座权重 7.5 GB, SLAKE 1 GB, IU X-Ray 14 GB (只 B2 需要), 其余是 adapter 和缓存。
- 能访问 HuggingFace 和 GitHub。国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 步骤
```bash
git clone https://github.com/AugustLoo/MedLoRA && cd MedLoRA
bash scripts/run_workstation.sh setup        # 一次性: venv + torch cu128 + LLaMA-Factory + 数据
bash scripts/run_workstation.sh baseline     # 约 30 min
bash scripts/run_workstation.sh A            # 约 1 h
bash scripts/run_workstation.sh B1           # 约 1.5 h
bash scripts/run_workstation.sh B2 ~/iu_xray # 约 1.5 h; 目录里要有 indiana_reports.csv / indiana_projections.csv / images/
```
IU X-Ray 从 Kaggle 下: `kaggle datasets download -d raddar/chest-xrays-indiana-university --unzip -p ~/iu_xray`。

## 跑完以后
把 `outputs/eval/*.json` 和 `*_preds.jsonl` 拷回主仓库的 `results/` 与 `outputs/eval/`, 数字直接和 Kaggle 那几轮对比。
`configs/bf16/` 里的 yaml 是从 `configs/` 自动派生的 (不量化、bf16、batch 4×4), 有效 batch、学习率、轮数、seed 全部一致, 所以结果可以和 T4 上的直接比; 唯一差别是 4bit 量化本身带来的少量噪声, 报告里要注明。

## 常见问题
- `CUDA error: no kernel image`: torch 的 CUDA 版本和显卡不匹配, RTX 50 系列必须 cu128。
- 显存不够 (不该发生): 把 `configs/bf16/*.yaml` 里 `per_device_train_batch_size` 改 2、`gradient_accumulation_steps` 改 8。
- 想再快一点: 在 yaml 里加 `flash_attn: fa2` (需 `pip install flash-attn`), 或把 `image_max_pixels` 保持 262144 不要调大。
