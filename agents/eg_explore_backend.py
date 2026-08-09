"""ExploreVLM-style EG backend — independent spatial-reasoning VLM for EG.

Uses a SEPARATE model (Base Qwen2.5-VL-3B-Instruct, not the fine-tuned Brain) as
the exploration-guidance reasoner. Goal: produce `in|on|target <object>` that the
existing EG validator (eg_llm_backend.parse_eg_response) accepts.

Variants:
  naive:   generic prompt, no strict format instruction
  aligned: strict prompt — output exactly "in|on|target <object>" from observed list

Records per-call: raw output, parsed, validator reason, retries.
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


PROMPT_NAIVE = (
    "You are a helpful robotic agent exploring a household. Your task is to find "
    "'{target}'. Currently you can observe objects: {observed}. You have already "
    "tried: {explored}. Where should you look next? Output an exploration direction."
)


PROMPT_ALIGNED = (
    "You are a helpful robotic agent in an indoor environment. Your task is to "
    "find '{target}' in the house, based on common house layouts and object "
    "placements. Currently, you can observe the following objects in the house: "
    "{observed}. You need to output an exploration direction in the EXACT form of "
    "<relation> <object>, where <relation> is one of [target, in, on, near], and "
    "<object> is an object EXACTLY from the given object list (keep the exact "
    "capitalization and number, e.g. 'Sink 1'). Previously, you have tried: "
    "{explored}. Do NOT output a tried direction. Reply with ONLY the single "
    "direction line, nothing else."
)


def _load_model():
    global _MODEL, _PROC
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        path = os.environ.get("ROBOAGENT_EG_MODEL_PATH",
                              "/mnt/autodl_tmp2/zhuyanhao/Qwen2.5-VL-3B-Instruct.git")
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto")
        _PROC = AutoProcessor.from_pretrained(path)


def _infer_text(prompt: str) -> str:
    _load_model()
    import torch
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    t = _PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = _PROC(text=[t], return_tensors="pt").to(_MODEL.device)
    with torch.no_grad():
        out = _MODEL.generate(**inp, max_new_tokens=48)
    gen = out[0][inp.input_ids.shape[1]:]
    return _PROC.decode(gen, skip_special_tokens=True).strip()


def propose_explore_eg(
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    *,
    env_name: str,
    variant: str = "aligned",
    qwen_infer: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """ExploreVLM-style EG: Base Qwen + prompt variant, validated by existing validator.

    Returns None if no valid direction after retries (caller drains buffer -> pass).
    """
    prompt = (PROMPT_NAIVE if variant == "naive" else PROMPT_ALIGNED).format(
        target=target_obj,
        observed=", ".join(str(x) for x in observed_objects) or "(none)",
        explored=", ".join(str(x) for x in explored) or "(none)",
    )
    allowed = legal_objects(observed_objects, explored, env_name)
    infer = qwen_infer or _infer_text
    tries = max_retries()
    last_reason = None
    for i in range(tries):
        try:
            raw = infer(prompt)
        except Exception as e:
            return None
        # strip chat wrapper if the model echoes <|...|>
        raw = re.sub(r'<\|[^|]*\|>', '', raw or '').strip()
        ok, why = parse_eg_response(raw, allowed, env_name, explored=explored)
        if ok is not None:
            return ok
        last_reason = why
    return None


ground_explore = propose_explore_eg
