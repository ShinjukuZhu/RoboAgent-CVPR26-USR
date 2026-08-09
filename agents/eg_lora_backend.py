"""Lightweight EG backend — FT Qwen + EG-specific LoRA adapter.

Consumes prompt_eg(target, observed, explored) and outputs a valid
`in|on|target <object>` direction. Loaded ONCE (module-level cache), used by
agent.py's exploration_guidance branch (ROBOAGENT_EG_BACKEND=eg_lora).
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Callable, List, Optional, Sequence

from agents.eg_llm_backend import (  # type: ignore
    legal_objects,
    parse_eg_response,
    max_retries,
)

_LOCK = threading.Lock()
_MODEL = None
_PROC = None
_ADAPTER_PATH = "/mnt/autodl_tmp2/zhuyanhao/training/checkpoints/eg_backend"
_QWEN_FT = "/mnt/autodl_tmp1/zhuyanhao/ckpt/RoboAgent_CVPR26"

PROMPT_EG = (
    "Suppose you are a helpful robotic agent in an indoor environment. Your task "
    "is to find '{target}' in the house, based on common house layouts and object "
    "placements. Currently, you can observe the following objects in the house: "
    "{observed}. You need to output an exploration direction in the exact form of "
    "<relation> <object>, where <relation> is chosen from [target, in, on, near], "
    "and <object> is an object from the given object list. Previously, you have "
    "tried the following exploration directions: {explored}. Do not output the "
    "directions in this list again since they all failed."
)


def _load():
    global _MODEL, _PROC
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        path = os.environ.get("ROBOAGENT_EG_ADAPTER", _ADAPTER_PATH)
        qwen = os.environ.get("ROBOAGENT_EG_BASE", _QWEN_FT)
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen, torch_dtype=torch.bfloat16, device_map="auto")
        _MODEL = PeftModel.from_pretrained(_MODEL, path)
        _MODEL.eval()
        _PROC = AutoProcessor.from_pretrained(qwen)


def _infer_text(prompt: str) -> str:
    _load()
    import torch
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    t = _PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = _PROC(text=[t], return_tensors="pt").to(_MODEL.device)
    with torch.no_grad():
        out = _MODEL.generate(**inp, max_new_tokens=48)
    gen = out[0][inp.input_ids.shape[1]:]
    return _PROC.decode(gen, skip_special_tokens=True).strip()


def propose_eg_lora(
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    *,
    env_name: str,
    qwen_infer: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Lightweight EG: FT Qwen + EG LoRA -> validated direction."""
    target = (target_obj or "").split(" (hint")[0].split(" (except")[0].strip()
    prompt = PROMPT_EG.format(
        target=target,
        observed=", ".join(str(x) for x in observed_objects) or "(none)",
        explored=", ".join(str(x) for x in explored) or "(none)",
    )
    allowed = legal_objects(observed_objects, explored, env_name)
    infer = qwen_infer or _infer_text
    tries = max_retries()
    for i in range(tries):
        try:
            raw = infer(prompt)
        except Exception:
            return None
        raw = re.sub(r'<\|[^|]*\|>', '', raw or '').strip()
        ok, why = parse_eg_response(raw, allowed, env_name, explored=explored)
        if ok is not None:
            return ok
    return None


ground_eg_lora = propose_eg_lora
