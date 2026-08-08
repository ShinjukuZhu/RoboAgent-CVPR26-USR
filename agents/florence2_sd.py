"""Florence-2 backed Scene Description (SD) with pluggable backends.

Backends (env ROBOAGENT_SD_BACKEND):
  - qwen:             original fine-tuned Qwen SD (default)
  - florence2_naive:  Florence-2 caption directly, NO adapter (to test interface mismatch)
  - florence2_adapter: Florence-2 caption -> Qwen-enriched spatial description (interface restored)

The Learned Interface Alignment of SD: original training produced descriptions rich in
spatial/state info (from scene graph); a generic captioner (Florence-2) breaks that
distribution. The adapter re-aligns by asking Qwen to re-describe with spatial relations.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None
_MODEL_ID = None
_DEVICE = None

TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def _ensure_model() -> None:
    global _MODEL, _PROCESSOR, _MODEL_ID, _DEVICE
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = os.environ.get("ROBOAGENT_FLORENCE2_MODEL", "microsoft/Florence-2-large")
        local = (os.environ.get("ROBOAGENT_FLORENCE2_PATH") or os.environ.get("ROBOAGENT_FLORENCE_PATH") or "").strip()
        if not local:
            for cand in (
                "/mnt/autodl_tmp2/zhuyanhao/ckpts/Florence-2-large-ft",
                "/mnt/autodl_tmp1/zhuyanhao/ckpt/Florence-2-large-ft",
            ):
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model.safetensors")):
                    # only use if looks complete (>=1GB)
                    try:
                        if os.path.getsize(os.path.join(cand, "model.safetensors")) >= 1_000_000_000:
                            local = cand
                            break
                    except OSError:
                        pass
        load_id = local if local else model_id
        device = os.environ.get(
            "ROBOAGENT_FLORENCE2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        _PROCESSOR = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
        # transformers>=4.5x expects class-attr _supports_sdpa; Florence remote code
        # exposes it as a @property that touches language_model before init → AttributeError.
        # Force eager attention to skip SDPA dispatch path.
        attn_impl = os.environ.get("ROBOAGENT_FLORENCE_ATTN", "eager")
        try:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                load_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        except TypeError:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                load_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        # Hardening: ensure class-level flags exist for any late checks
        try:
            type(_MODEL)._supports_sdpa = False
            type(_MODEL)._supports_flash_attn_2 = False
        except Exception:
            pass
        _MODEL.to(device)
        _MODEL.eval()
        _MODEL_ID = load_id
        _DEVICE = device


def florence2_caption(image_path: str, prompt: str = TASK_PROMPT) -> str:
    """Run Florence-2 and return raw caption text."""
    import torch
    from PIL import Image

    _ensure_model()
    image = Image.open(image_path).convert("RGB")
    inputs = _PROCESSOR(text=prompt, images=image, return_tensors="pt")
    model_dtype = next(_MODEL.parameters()).dtype
    casted = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            casted[k] = v
            continue
        v = v.to(_DEVICE)
        if k in ("pixel_values", "pixel_values_2") and v.is_floating_point():
            v = v.to(dtype=model_dtype)
        casted[k] = v
    inputs = casted

    with torch.no_grad():
        # use_cache=False: Florence remote prepare_inputs_for_generation breaks on
        # transformers 4.57 DynamicCache (past_key_values[0][0] is None).
        generated_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False,
        )
    generated_text = _PROCESSOR.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    # Strip the prompt prefix if present
    parsed = _PROCESSOR.post_process_generation(
        generated_text, task=TASK_PROMPT, image_size=image.size
    )
    return parsed[TASK_PROMPT] if TASK_PROMPT in parsed else str(parsed)


def describe_naive(image_path: str, label: str, invent: str = "") -> str:
    """Direct Florence-2 caption. NO interface adapter. Tests mismatch."""
    cap = florence2_caption(image_path)
    # Strip the first sentence that repeats the object, keep raw
    cap = cap.strip()
    if not cap:
        cap = f"The {label} is visible in the scene."
    return cap


ADAPTER_PROMPT = (
    "This is an egocentric image observed by a robotic household agent. "
    "A detector says the target object is '{label}'."
    " {invent_hint}"
    "Please describe the '{label}' and its spatial relations to nearby "
    "objects/receptacles (e.g. on the table, next to the fridge, inside the "
    "drawer). Include object state if visible (open/closed, clean/dirty)."
    " Base your description on the image, not on any caption below. "
    " A draft caption is provided as auxiliary only.\n"
    "Draft caption: {caption}"
)


def describe_adapter(
    image_path: str,
    label: str,
    qwen_infer,
    invent: str = "",
    max_new_tokens: int = 512,
) -> str:
    """Florence-2 caption -> Qwen re-description (interface restored)."""
    cap = florence2_caption(image_path)
    invent_hint = (
        f" Note that the agent is holding {invent}, shown at the bottom of the image. "
        "Ignore it in the description. "
        if invent and invent != "nothing"
        else ""
    )
    prompt = ADAPTER_PROMPT.format(
        label=label, invent_hint=invent_hint, caption=cap
    )
    res = qwen_infer(prompt)
    return str(res).strip() if res else cap
