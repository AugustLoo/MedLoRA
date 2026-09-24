"""模型加载与推理。同一份代码用于 zero-shot 基线、加载 LoRA adapter 后的评估, 以及远程接口评估。

两种后端, 由环境变量 MEDVLM_API_BASE 决定 (详见 medvlm/remote.py):
  - 没设: 本地加载 (transformers + peft), 与此前所有实验完全相同
  - 设了: 通过 OpenAI 兼容接口调用远程服务, model_id 当作服务端的模型名
四个评估脚本只调用 load_model / generate / backend_info, 不关心是哪种后端。
torch 在函数内导入, 远程模式下不需要装 torch。
"""
from __future__ import annotations

from PIL import Image

from medvlm.remote import RemoteModel, api_base_from_env


def pick_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16  # T4 / RTX 30 笔记本走这里


def load_model(model_id: str, adapter: str | None = None, load_4bit: bool = False,
               max_pixels: int = 512 * 28 * 28, min_pixels: int = 64 * 28 * 28):
    if api_base_from_env():
        if adapter:
            raise SystemExit(
                "远程模式不能加载本地 adapter 路径。请让服务端把 adapter 挂成一个模型名 "
                "(vLLM: --enable-lora --lora-modules <名字>=<路径>), 然后用 --model <名字> 调用。")
        return RemoteModel.from_env(model_id, max_pixels=max_pixels), None

    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(model_id)
    mtype = getattr(cfg, "model_type", "")
    if mtype == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    elif mtype == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForVision2Seq as Cls

    dtype = pick_dtype()
    kwargs = dict(torch_dtype=dtype, device_map="auto")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = Cls.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    # min/max_pixels 控制视觉 token 数量: 4GB 显存调小, 云端可放大
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def backend_info(model) -> dict:
    """写进每张表的汇总 JSON, 事后能分清一个数字来自本地推理还是远程接口。"""
    if isinstance(model, RemoteModel):
        return model.describe()
    return {"backend": "local"}


def generate(model, processor, prompt: str, image: Image.Image | None = None,
             max_new_tokens: int = 32) -> str:
    if isinstance(model, RemoteModel):
        return model.chat(prompt, image, max_new_tokens)
    return _generate_local(model, processor, prompt, image, max_new_tokens)


def _generate_local(model, processor, prompt: str, image: Image.Image | None, max_new_tokens: int) -> str:
    import torch
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image is not None:
        inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()
