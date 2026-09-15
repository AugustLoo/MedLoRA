# 本机 (RTX 3050, 4GB) 冒烟测试: 用 2B 模型 4bit + 小图, 只跑 10 条, 验证代码能跑通。
# 不是拿来出成绩的, 正式数字去云端跑。
Set-Location (Split-Path $PSScriptRoot -Parent)
python eval/eval_slake.py `
  --model Qwen/Qwen2-VL-2B-Instruct `
  --load-4bit `
  --max-pixels 200704 `
  --limit 10 `
  --tag smoke
