"""EG Adapter — convert ExploreVLM (Base Qwen) raw output to valid `in|on|target <obj>`.

Skill Logic: Base Qwen proposes the most likely receptacle/surface; Adapter maps
it to the contract format by:
  1. extracting the FIRST observed-object mention from raw output
  2. prefixing with "target " (or relation if raw contains in/on near target)
  3. falling back to "target <target-normalized>" if no observed hit
Validated by existing EG validator (legal_objects + parse_eg_response).
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
    _normalize_obj,
)

_LOCK = threading.Lock()
_MODEL = None
_PROC = None


PROMPT_EG_ADAPTER = (
    "You are a robotic agent exploring a household to find '{target}'. Based on "
    "common house layouts, which of these observed objects is the MOST likely "
    "place to find '{target}' (a container to look in, a surface to look on, or "
    "the object itself)? Observed objects: {observed}. Already tried: {explored}. "
    "Reply with ONLY the single most likely object name from the list, nothing else."
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


def _extract_observed(raw: str, observed: Sequence[str], env_name: str) -> Optional[str]:
    """Find the first observed-object mention in raw output."""
    if not raw:
        return None
    # exact match first (case-insensitive)
    for obj in observed:
        o = str(obj)
        if re.search(r'(?<![a-z0-9])' + re.escape(o) + r'(?![a-z0-9])', raw, re.I):
            return o
    # token-level fallback: normalize and check presence
    raw_norm = re.sub(r'[^a-z0-9 ]', ' ', raw.lower())
    for obj in observed:
        o = str(obj)
        key = _normalize_obj(o, env_name)
        if key and re.search(r'(?<![a-z0-9])' + re.escape(key.lower()) + r'(?![a-z0-9])', raw_norm):
            return o
    return None


def propose_eg_adapter(
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    *,
    env_name: str,
    qwen_infer: Optional[Callable[[str], str]] = None,
    fallback_target: bool = True,
) -> Optional[str]:
    """ExploreVLM + Skill Logic + Adapter -> valid EG direction.

    Base Qwen names the likely place; Adapter wraps into contract format.
    """
    target_clean = (target_obj or "").split(" (hint")[0].split(" (except")[0].strip()
    prompt = PROMPT_EG_ADAPTER.format(
        target=target_clean,
        observed=", ".join(str(x) for x in observed_objects) or "(none)",
        explored=", ".join(str(x) for x in explored) or "(none)",
    )
    allowed = legal_objects(observed_objects, explored, env_name)
    infer = qwen_infer or _infer_text
    tries = max_retries()
    for i in range(tries):
        raw = infer(prompt)
        raw = re.sub(r'<\|[^|]*\|>', '', raw or '').strip()
        obj = _extract_observed(raw, observed_objects, env_name)
        if obj is None and fallback_target and allowed:
            # fallback: use target's own class if it's in observed
            tgt = _normalize_obj(target_clean, env_name)
            for o in observed_objects:
                if _normalize_obj(str(o), env_name) == tgt or tgt in _normalize_obj(str(o), env_name):
                    obj = str(o)
                    break
        if obj is None:
            continue
        # try target first, then in/on, until validator accepts
        for cand in [f"target {obj}", f"on {obj}", f"in {obj}"]:
            ok, why = parse_eg_response(cand, allowed, env_name, explored=explored)
            if ok is not None:
                return ok
    return None


ground_eg_adapter = propose_eg_adapter
