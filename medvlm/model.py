"""模型加载与推理。同一份代码用于 zero-shot 基线和加载 LoRA adapter 后的评估。"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16  # T4 / RTX 30 笔记本走这里


def load_model(model_id: str, adapter: str | None = None, load_4bit: bool = False,
               max_pixels: int = 512 * 28 * 28, min_pixels: int = 64 * 28 * 28):
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


@torch.inference_mode()
def generate(model, processor, prompt: str, image: Image.Image | None = None,
             max_new_tokens: int = 32) -> str:
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
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()
